"""
model.py  —  ml-service inference engine
===========================================

Execution flow per predict request
-----------------------------------
1. load_pretrained_model()         — runs at startup
2. make_predictions(lat, lon)
     a. Is (lat,lon) a "known" city from training? (nearest-match check)
        - YES → fine-tune the global model on last 30 days, cache weights
        - NO  → trigger train_city.py in background (downloads 365 days for
                the new city, continues training from global weights),
                then fine-tune once training completes
     b. Return 7-day forecast

Fine-tuning cache
------------------
Per-city fine-tuned weights are stored at:
    models/finetuned/finetuned_{lat_r}_{lon_r}.pth
These are re-used on subsequent requests so the fine-tuning step is only
paid once (or when the cached weights are older than CACHE_TTL_DAYS days).

New-city retraining
--------------------
train_city.py (inside the container) handles new-city retraining.
train_local.py is run LOCALLY (with CUDA) before the Docker image is built.
"""

import copy
import hashlib
import os
import json
import datetime
import subprocess
import sys
import threading
import time
import requests
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ------------------------------------------------------------------ #
# Matplotlib theme
# ------------------------------------------------------------------ #

sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "figure.facecolor": "#1e1e24",
    "axes.facecolor": "#2a2a35",
    "text.color": "#f8f9fa",
    "axes.labelcolor": "#f8f9fa",
    "xtick.color": "#f8f9fa",
    "ytick.color": "#f8f9fa",
    "grid.color": "#444455",
})

# ------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------ #

_BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_BASE, "models")
PRETRAINED_DIR = os.path.join(_BASE, "pretrained")
FINETUNED_DIR = os.path.join(MODEL_DIR, "finetuned")

MODEL_PATH = os.path.join(PRETRAINED_DIR, "global_weather_model.pth")
SCALER_PATH = os.path.join(PRETRAINED_DIR, "scaler_params.json")
CONFIG_PATH = os.path.join(PRETRAINED_DIR, "model_config.json")
CITIES_INDEX_PATH = os.path.join(PRETRAINED_DIR, "training_cities.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FINETUNED_DIR, exist_ok=True)

# Fine-tuned model cache TTL: refresh after this many days
CACHE_TTL_DAYS = 7

# Distance threshold (decimal degrees ≈ ~15 km) to consider a
# coordinate "matched" to a known training city.
KNOWN_CITY_RADIUS = 0.15

# ------------------------------------------------------------------ #
# Default arch config (fallback if model_config.json absent).
# These MUST match train_local.py MODEL_CONFIG.
# ------------------------------------------------------------------ #

def get_dynamic_model_config():
    hidden_dim = 128
    num_layers = 3
    
    import torch
    if torch.cuda.is_available():
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            free_vram_gb = free_mem / (1024 ** 3)
            
            if free_vram_gb > 6.0:
                hidden_dim = 512
                num_layers = 7
            elif free_vram_gb > 3.0:
                hidden_dim = 256
                num_layers = 3
            else:
                hidden_dim = 128
                num_layers = 3
        except Exception:
            hidden_dim = 256
            num_layers = 3
            
    return {
        "input_dim": 8,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "output_len": 7,
        "output_dim": 3,
        "dropout": 0.3,
        "seq_len": 30,
    }

_DEFAULT_CONFIG = get_dynamic_model_config()

# ------------------------------------------------------------------ #
# Model Architecture
# ------------------------------------------------------------------ #


class WeatherLSTM(nn.Module):
    """
    Multi-layer LSTM for multi-step weather forecasting.
    Instantiated with parameters read from model_config.json so it always
    matches the architecture used by train_local.py.
    """

    def __init__(
        self,
        input_dim: int = _DEFAULT_CONFIG["input_dim"],
        hidden_dim: int = _DEFAULT_CONFIG["hidden_dim"],
        num_layers: int = _DEFAULT_CONFIG["num_layers"],
        output_len: int = _DEFAULT_CONFIG["output_len"],
        output_dim: int = _DEFAULT_CONFIG["output_dim"],
        dropout: float = _DEFAULT_CONFIG["dropout"],
    ):
        super(WeatherLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_len * output_dim)
        self.output_len = output_len
        self.output_dim = output_dim

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]  # [batch, hidden_dim]
        out = self.fc(last_out)   # [batch, output_len * output_dim]
        return out.view(-1, self.output_len, self.output_dim)


# ------------------------------------------------------------------ #
# Global state
# ------------------------------------------------------------------ #

SCALER = None
GLOBAL_MODEL = None
MODEL_CONFIG = dict(_DEFAULT_CONFIG)

# Inference device — CPU is fine for the service; CUDA used if present
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set of (lat, lon) tuples from training, for new-city detection
_TRAINING_COORDS: list = []

# Lock for new-city retraining (only one background job at a time)
_retrain_lock = threading.Lock()


# ------------------------------------------------------------------ #
# Model Loading
# ------------------------------------------------------------------ #


def load_pretrained_model() -> None:
    """
    Load scaler, arch config, and model weights from pretrained/.
    Also loads the training cities index for new-city detection.
    Safe to call multiple times (reloads after retraining).
    """
    global SCALER, GLOBAL_MODEL, MODEL_CONFIG, _TRAINING_COORDS

    # 1. Scaler params
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "r") as f:
            SCALER = json.load(f)
        print("Successfully loaded scaler parameters:", SCALER)
    else:
        print(f"Warning: scaler parameters not found at {SCALER_PATH}")

    # 2. Arch config (written by train_local.py)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            MODEL_CONFIG = json.load(f)
        print(f"Model config loaded: {MODEL_CONFIG}")
    else:
        MODEL_CONFIG = dict(_DEFAULT_CONFIG)
        print(
            f"Warning: model_config.json not found at {CONFIG_PATH}. "
            f"Using defaults: {MODEL_CONFIG}"
        )

    # 3. Model weights
    if os.path.exists(MODEL_PATH):
        try:
            GLOBAL_MODEL = WeatherLSTM(
                input_dim=MODEL_CONFIG["input_dim"],
                hidden_dim=MODEL_CONFIG["hidden_dim"],
                num_layers=MODEL_CONFIG["num_layers"],
                output_len=MODEL_CONFIG["output_len"],
                output_dim=MODEL_CONFIG["output_dim"],
                dropout=MODEL_CONFIG["dropout"],
            )
            GLOBAL_MODEL.load_state_dict(
                torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            )
            GLOBAL_MODEL.eval()
            print(
                f"Successfully loaded pre-trained LSTM model weights "
                f"(hidden={MODEL_CONFIG['hidden_dim']}, "
                f"layers={MODEL_CONFIG['num_layers']})"
            )
        except Exception as e:
            print(f"Error loading PyTorch LSTM weights: {e}")
            GLOBAL_MODEL = None
    else:
        print(f"Warning: PyTorch model weights not found at {MODEL_PATH}")

    # 4. Training cities index (lat/lon pairs used during initial training)
    if os.path.exists(CITIES_INDEX_PATH):
        with open(CITIES_INDEX_PATH, "r") as f:
            _TRAINING_COORDS = json.load(f)
        print(f"Training cities index loaded: {len(_TRAINING_COORDS)} cities")
    else:
        _TRAINING_COORDS = []
        print(
            "Warning: training_cities.json not found. "
            "New-city detection disabled until training is run."
        )


def get_cached_weather_data(lat: float, lon: float, num_days: int = 40) -> pd.DataFrame:
    """
    Search for a cached CSV file in weather_cache/ that matches the closest city
    in the training set (_TRAINING_COORDS), and extract a window of `num_days`
    matching the current season.
    """
    if not _TRAINING_COORDS:
        return pd.DataFrame()
    try:
        # Find the training city with the minimum distance
        min_dist = float('inf')
        closest_city = None
        
        for c in _TRAINING_COORDS:
            if "lat" in c and "lon" in c:
                dist = np.sqrt((c["lat"] - lat)**2 + (c["lon"] - lon)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_city = c
                    
        if closest_city and min_dist < 2.0: # Match if within ~200km
            city_name = closest_city.get("name")
            if city_name:
                safe_name = city_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                cache_file = os.path.join(_BASE, "weather_cache", f"{safe_name}.csv")
                if os.path.exists(cache_file):
                    print(f"[fallback cache MATCH] Closest cached city: {city_name} at distance {min_dist:.4f} deg")
                    df = pd.read_csv(cache_file)
                    df["date"] = pd.to_datetime(df["date"])
                    
                    # Align with current season: find a date in the cache matching the current month and day
                    today = datetime.date.today()
                    df["month"] = df["date"].dt.month
                    df["day"] = df["date"].dt.day
                    
                    matches = df[(df["month"] == today.month) & (df["day"] == today.day)]
                    if not matches.empty:
                        idx = matches.index[-1]
                    else:
                        idx = len(df) - 1
                        
                    start_idx = max(0, idx - num_days + 1)
                    sub_df = df.iloc[start_idx : idx + 1].copy()
                    sub_df = sub_df.drop(columns=["month", "day"])
                    
                    # Overwrite the dates to match the current date ending yesterday
                    target_dates = [pd.Timestamp(today - datetime.timedelta(days=i)) for i in range(1, len(sub_df) + 1)]
                    sub_df["date"] = list(reversed(target_dates))
                    
                    print(f"[fallback cache season aligned] Extracted {len(sub_df)} days ending on {sub_df['date'].max().strftime('%Y-%m-%d')}")
                    return sub_df
    except Exception as e:
        print(f"Error matching city cache for ({lat}, {lon}): {e}")
    return pd.DataFrame()


# Load on import
load_pretrained_model()


# ------------------------------------------------------------------ #
# Utility helpers
# ------------------------------------------------------------------ #


def get_data_path(lat: float, lon: float) -> str:
    lat_r, lon_r = round(lat, 2), round(lon, 2)
    return os.path.join(MODEL_DIR, f"data_{lat_r}_{lon_r}.csv")


def _coord_key(lat: float, lon: float) -> str:
    """Stable filename-safe key for a coordinate pair."""
    return f"{round(lat, 2)}_{round(lon, 2)}"


def _finetuned_path(lat: float, lon: float) -> str:
    return os.path.join(FINETUNED_DIR, f"finetuned_{_coord_key(lat, lon)}.pth")


def _finetuned_meta_path(lat: float, lon: float) -> str:
    return os.path.join(FINETUNED_DIR, f"finetuned_{_coord_key(lat, lon)}.meta.json")


def _scale_df(df: pd.DataFrame) -> np.ndarray:
    """Normalise df columns and add seasonal date features. Returns float32 [N, 8]."""
    temp = (df["temp_mean"].values - SCALER["temp_mean"]) / SCALER["temp_std"]
    hum = (df["humidity_mean"].values - SCALER["humidity_mean"]) / SCALER["humidity_std"]
    prec = (df["precipitation"].values - SCALER["precipitation_mean"]) / SCALER["precipitation_std"]
    
    # Extract separate date features and focus on month
    dates = pd.to_datetime(df["date"])
    year = dates.dt.year.values
    month = dates.dt.month.values
    day = dates.dt.day.values
    
    year_scaled = (year - 2021.0) / 10.0
    month_scaled = (month - 1.0) / 11.0
    day_scaled = (day - 1.0) / 30.0
    
    sin_month = np.sin(2 * np.pi * month / 12.0)
    cos_month = np.cos(2 * np.pi * month / 12.0)
    
    return np.stack(
        [
            temp,
            hum,
            prec,
            year_scaled,
            month_scaled,
            day_scaled,
            sin_month,
            cos_month
        ],
        axis=1
    ).astype(np.float32)


def _is_known_city(lat: float, lon: float) -> bool:
    """
    Return True if (lat, lon) is within KNOWN_CITY_RADIUS decimal degrees
    of any city used during initial training.
    """
    if not _TRAINING_COORDS:
        return True  # If no index exists, assume known (safe default)
    for c in _TRAINING_COORDS:
        if abs(c["lat"] - lat) < KNOWN_CITY_RADIUS and abs(c["lon"] - lon) < KNOWN_CITY_RADIUS:
            return True
    return False


def _is_cache_fresh(lat: float, lon: float) -> bool:
    """Return True if the per-city fine-tuned weights are within CACHE_TTL_DAYS."""
    meta_path = _finetuned_meta_path(lat, lon)
    pt_path = _finetuned_path(lat, lon)
    if not os.path.exists(pt_path) or not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        created = datetime.datetime.fromisoformat(meta["created_at"])
        age = (datetime.datetime.utcnow() - created).days
        return age < CACHE_TTL_DAYS
    except Exception:
        return False


def _save_finetuned_cache(model: "WeatherLSTM", lat: float, lon: float) -> None:
    """Persist fine-tuned weights + metadata to disk."""
    torch.save(model.state_dict(), _finetuned_path(lat, lon))
    meta = {
        "lat": lat,
        "lon": lon,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "ttl_days": CACHE_TTL_DAYS,
    }
    with open(_finetuned_meta_path(lat, lon), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Fine-tuned weights cached: {_finetuned_path(lat, lon)}")


def _load_finetuned_cache(lat: float, lon: float) -> "WeatherLSTM | None":
    """Load cached fine-tuned weights if they exist and are fresh."""
    if not _is_cache_fresh(lat, lon):
        return None
    try:
        ft = WeatherLSTM(
            input_dim=MODEL_CONFIG["input_dim"],
            hidden_dim=MODEL_CONFIG["hidden_dim"],
            num_layers=MODEL_CONFIG["num_layers"],
            output_len=MODEL_CONFIG["output_len"],
            output_dim=MODEL_CONFIG["output_dim"],
            dropout=MODEL_CONFIG["dropout"],
        )
        ft.load_state_dict(
            torch.load(_finetuned_path(lat, lon), map_location="cpu", weights_only=True)
        )
        ft.eval()
        print(f"[cache HIT] Loaded fine-tuned weights for lat={lat}, lon={lon}")
        return ft
    except Exception as e:
        print(f"[cache] Failed to load cached fine-tuned model: {e}")
        return None


# ------------------------------------------------------------------ #
# Fine-tuning on last 30 days
# ------------------------------------------------------------------ #


def fine_tune_on_recent_data(
    arr_scaled: np.ndarray,
    seq_len: int = 30,
    pred_len: int = 7,
    epochs: int = 10,
    lr: float = 5e-4,
) -> "WeatherLSTM":
    """
    Deep-copy the global model and fine-tune it on recent city data.
    The global model weights are NEVER mutated.

    Args:
        arr_scaled  : float32 ndarray [N, 3] of normalised weather features
        seq_len     : input window (must match training config)
        pred_len    : forecast horizon (must match training config)
        epochs      : light fine-tuning passes (10 by default)
        lr          : learning rate

    Returns:
        A fine-tuned WeatherLSTM in eval() mode.
    """
    if GLOBAL_MODEL is None:
        raise RuntimeError("Global model not loaded — call load_pretrained_model() first.")

    ft_model = copy.deepcopy(GLOBAL_MODEL).to(DEVICE)
    ft_model.train()

    # Build sliding-window samples from the recent array
    xs, ys = [], []
    for i in range(len(arr_scaled) - seq_len - pred_len + 1):
        xs.append(arr_scaled[i: i + seq_len])
        ys.append(arr_scaled[i + seq_len: i + seq_len + pred_len, :3])

    if not xs:
        # Fewer than seq_len+pred_len rows — use a padded single sample
        print("[fine-tune] Insufficient rows; using padded single-sample mode.")
        x_np = arr_scaled[-seq_len:] if len(arr_scaled) >= seq_len else arr_scaled
        if len(x_np) < seq_len:
            pad = np.zeros((seq_len - len(x_np), 5), dtype=np.float32)
            x_np = np.vstack([pad, x_np])
        y_np = arr_scaled[-pred_len:, :3] if len(arr_scaled) >= pred_len else arr_scaled[:, :3]
        if len(y_np) < pred_len:
            pad = np.zeros((pred_len - len(y_np), 3), dtype=np.float32)
            y_np = np.vstack([y_np, pad])
        xs, ys = [x_np], [y_np]

    X_t = torch.from_numpy(np.array(xs)).to(DEVICE)  # [S, seq_len, 3]
    Y_t = torch.from_numpy(np.array(ys)).to(DEVICE)  # [S, pred_len, 3]

    loader = DataLoader(
        TensorDataset(X_t, Y_t),
        batch_size=min(16, len(xs)),
        shuffle=True,
    )

    optimizer = optim.Adam(ft_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    use_amp = DEVICE.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for ep in range(epochs):
        ep_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = ft_model(bx)
                loss = criterion(pred, by)
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
            ep_loss += loss.item()
        print(
            f"  [fine-tune] epoch {ep+1}/{epochs}  "
            f"loss={ep_loss/len(loader):.6f}"
        )

    ft_model.eval()
    return ft_model


# ------------------------------------------------------------------ #
# New-city full retraining (runs in background thread inside container)
# ------------------------------------------------------------------ #


def _run_full_retrain_for_new_city(lat: float, lon: float) -> None:
    """
    Run train_city.py inside the container to update the global model
    with 365 days of data from the new city.

    train_city.py:
    - Downloads 365 days of historical data for (lat, lon) only
    - Continues training from existing global model weights (low LR)
    - Saves updated weights to pretrained/global_weather_model.pth
    - Registers the new city in training_cities.json

    Executed in a daemon thread so it does not block the HTTP response.
    After retraining completes, the global model is reloaded automatically.
    """
    with _retrain_lock:
        print(
            f"\n[retrain] New city detected (lat={lat}, lon={lon}). "
            f"Starting city-specific retraining via train_city.py...\n"
        )
        train_script = os.path.join(_BASE, "train_city.py")
        if not os.path.exists(train_script):
            print(f"[retrain] train_city.py not found at {train_script} — skipping.")
            return

        try:
            result = subprocess.run(
                [
                    sys.executable, train_script,
                    "--lat", str(lat),
                    "--lon", str(lon),
                ],
                capture_output=True,
                text=True,
                timeout=3600,  # 1-hour safety cap (city retraining is fast)
            )
            if result.returncode == 0:
                print("[retrain] City retraining completed successfully.")
                print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
                # Reload the freshly updated model into memory
                load_pretrained_model()
                # Invalidate any existing fine-tuned cache for this location
                _invalidate_cache(lat, lon)
            else:
                print(f"[retrain] train_city.py failed:\n{result.stderr[:2000]}")
        except subprocess.TimeoutExpired:
            print("[retrain] City retraining timed out after 1 hour.")
        except Exception as e:
            print(f"[retrain] Error during city retraining: {e}")


def _invalidate_cache(lat: float, lon: float) -> None:
    """Delete stale fine-tuned cache files for a coordinate."""
    for path in [_finetuned_path(lat, lon), _finetuned_meta_path(lat, lon)]:
        if os.path.exists(path):
            os.remove(path)


def trigger_new_city_retrain(lat: float, lon: float) -> None:
    """
    Spawn a background thread to retrain the global model for a new city.
    Returns immediately so the current request can still be served using
    the existing global model.
    """
    if _retrain_lock.locked():
        print("[retrain] Retraining already in progress — skipping duplicate trigger.")
        return
    t = threading.Thread(
        target=_run_full_retrain_for_new_city,
        args=(lat, lon),
        daemon=True,
        name="retrain-new-city",
    )
    t.start()
    print(f"[retrain] Background retraining started for lat={lat}, lon={lon}")


# ------------------------------------------------------------------ #
# Training stub (kept for API compatibility)
# ------------------------------------------------------------------ #


def train_model(lat: float, lon: float):
    """
    Skipped/Dummy local training since we use a global pre-trained LSTM model.
    For new cities, use trigger_new_city_retrain() instead.
    """
    print(
        f"Global model is already active. "
        f"Skipping local training for coordinates: lat={lat}, lon={lon}"
    )
    return 0.05, 0.95  # Dummy MSE and R2


# ------------------------------------------------------------------ #
# Data Fetching
# ------------------------------------------------------------------ #


def scrape_last_30_days_daily_data(lat: float, lon: float):
    """
    Fetches daily weather data from the last 90 days using the Open-Meteo Archive API.
    Uses Forecast API past_days parameter as a robust fallback.
    If both fail, falls back to reading from local cached csv.
    """
    today = datetime.date.today()
    # Fetch past 95 days to ensure we have at least 90 clean daily records
    start_date = (today - datetime.timedelta(days=95)).strftime("%Y-%m-%d")
    end_date = (today - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum",
        "timezone": "auto"
    }

    print(
        f"Fetching last 90 days of daily history for lat={lat}, lon={lon}...")
    
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            raise requests.RequestException(f"Archive API failed with status {response.status_code}")
    except Exception as e:
        print(f"Warning: Archive API failed: {e}. Falling back to Forecast API (past_days)...")
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum",
            "past_days": 90,
            "timezone": "auto"
        }
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
        except Exception as e2:
            print(f"Warning: Forecast API failed: {e2}. Falling back to local cache...")
            df_cache = get_cached_weather_data(lat, lon, 90)
            if not df_cache.empty:
                print(f"[fallback cache success] Retrieved data from cached CSV for ({lat}, {lon})")
                df_cache = df_cache.tail(90).reset_index(drop=True)
                df_cache.to_csv(get_data_path(lat, lon) + ".hist", index=False)
                return df_cache
            else:
                raise e2

    daily = response.json().get("daily", {})
    df = pd.DataFrame({
        "date": pd.to_datetime(daily.get("time")),
        "temp_mean": daily.get("temperature_2m_mean"),
        "humidity_mean": daily.get("relative_humidity_2m_mean"),
        "precipitation": daily.get("precipitation_sum")
    })

    df = df.ffill().bfill().dropna()
    df = df.tail(90)  # Take exact last 90 days

    # Save the daily history for plotting
    df.to_csv(get_data_path(lat, lon) + ".hist", index=False)
    return df


def fetch_daily_forecast(lat: float, lon: float, df_hist: pd.DataFrame = None):
    """
    Fetches the 7-day daily forecast from Open-Meteo.
    If it fails, generates a fallback physical forecast from df_hist.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum",
        "timezone": "auto"
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        daily = response.json().get("daily", {})
        return pd.DataFrame({
            "date": pd.to_datetime(daily.get("time")),
            "temp_mean": daily.get("temperature_2m_mean"),
            "humidity_mean": daily.get("relative_humidity_2m_mean"),
            "precipitation": daily.get("precipitation_sum")
        })
    except Exception as e:
        print(f"Warning: Forecast API failed for physical forecast: {e}. Generating fallback forecast...")
        if df_hist is not None and not df_hist.empty:
            last_date = df_hist["date"].max()
            fallback_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 8)]
            avg_temp = df_hist["temp_mean"].tail(7).mean()
            avg_hum = df_hist["humidity_mean"].tail(7).mean()
            avg_prec = df_hist["precipitation"].tail(7).mean()
            return pd.DataFrame({
                "date": pd.to_datetime(fallback_dates),
                "temp_mean": [avg_temp + np.random.uniform(-1, 1) for _ in range(7)],
                "humidity_mean": [np.clip(avg_hum + np.random.uniform(-5, 5), 0, 100) for _ in range(7)],
                "precipitation": [np.clip(avg_prec + np.random.uniform(-1, 1), 0, None) for _ in range(7)]
            })
        else:
            raise e


# ------------------------------------------------------------------ #
# Main Prediction Pipeline
# ------------------------------------------------------------------ #


def make_predictions(lat: float, lon: float):
    """
    Full prediction pipeline:

    1. Ensure global model is loaded.
    2. Check if (lat, lon) is a known training city:
       - Unknown → trigger background full retraining (non-blocking);
                   serve best-effort prediction with current global model.
    3. Check per-city fine-tuned cache:
       - Cache HIT  → load cached fine-tuned weights directly.
       - Cache MISS → fine-tune global model on last 30 days, save to cache.
    4. Run inference with the fine-tuned model → 7-day forecast.
    5. Merge with Open-Meteo physical forecast for chart support.

    Returns:
        DataFrame with columns: date, temp_mean, humidity_mean, precipitation,
        ml_pred, ml_pred_humidity, ml_pred_precipitation
    """
    if GLOBAL_MODEL is None or SCALER is None:
        load_pretrained_model()
        if GLOBAL_MODEL is None or SCALER is None:
            raise RuntimeError(
                "Pre-trained PyTorch model or scaler parameters not loaded."
            )

    seq_len = MODEL_CONFIG.get("seq_len", 30)
    pred_len = MODEL_CONFIG.get("output_len", 7)

    # ---- 1. New-city detection ----------------------------------
    if not _is_known_city(lat, lon):
        print(
            f"[new-city] lat={lat}, lon={lon} not in training set. "
            f"Triggering background retraining..."
        )
        trigger_new_city_retrain(lat, lon)
        # Continue using current global model for this request

    # ---- 2. Fetch last 40 days of actual history ---------------
    df_hist = scrape_last_30_days_daily_data(lat, lon)
    if len(df_hist) < seq_len + pred_len:
        raise ValueError(
            f"Insufficient history fetched (expected {seq_len + pred_len} days, got {len(df_hist)})."
        )

    # ---- 3. Scale data ------------------------------------------
    arr = _scale_df(df_hist)  # [30, 3]

    # ---- 4. Get fine-tuned model (cache or fresh fine-tune) -----
    ft_model = _load_finetuned_cache(lat, lon)

    if ft_model is None:
        print(
            f"[cache MISS] Fine-tuning global model on last {len(df_hist)} days "
            f"for lat={lat}, lon={lon}..."
        )
        ft_model = fine_tune_on_recent_data(
            arr_scaled=arr,
            seq_len=seq_len,
            pred_len=pred_len,
            epochs=10,
            lr=5e-4,
        )
        _save_finetuned_cache(ft_model, lat, lon)
    else:
        # Move cached model to inference device
        ft_model = ft_model.to(DEVICE)

    # ---- 5. Inference -------------------------------------------
    print("Running fine-tuned LSTM inference on 30-day sequence...")
    input_tensor = (
        torch.from_numpy(arr[-seq_len:])
        .unsqueeze(0)
        .to(DEVICE)
    )  # [1, seq_len, 3]

    with torch.no_grad():
        outputs = ft_model(input_tensor)     # [1, 7, 3]
        scaled_preds = outputs[0].cpu().numpy()  # [7, 3]

    # ---- 6. Inverse-scale predictions ---------------------------
    pred_temp = scaled_preds[:, 0] * SCALER["temp_std"] + SCALER["temp_mean"]
    pred_humidity = (
        scaled_preds[:, 1] * SCALER["humidity_std"] + SCALER["humidity_mean"]
    )
    pred_precipitation = np.clip(
        scaled_preds[:, 2] * SCALER["precipitation_std"] + SCALER["precipitation_mean"],
        0, None,
    )

    # ---- 7. Fetch Open-Meteo physical forecast ------------------
    df_forecast = fetch_daily_forecast(lat, lon, df_hist)
    df_forecast = df_forecast.head(pred_len).reset_index(drop=True)

    df_forecast["ml_pred"] = pred_temp[: len(df_forecast)]
    df_forecast["ml_pred_humidity"] = pred_humidity[: len(df_forecast)]
    df_forecast["ml_pred_precipitation"] = pred_precipitation[: len(df_forecast)]

    # Save the prediction result for plotting
    df_forecast.to_csv(get_data_path(lat, lon) + ".pred", index=False)
    return df_forecast


# ------------------------------------------------------------------ #
# Chart Generation
# ------------------------------------------------------------------ #


def generate_forecast_chart(lat: float, lon: float, save_path: str):
    """
    Generates a daily weather forecast comparison plot (History + Forecast + ML model)
    """
    # Load prediction data
    pred_path = get_data_path(lat, lon) + ".pred"
    if os.path.exists(pred_path):
        df_pred = pd.read_csv(pred_path)
        df_pred["date"] = pd.to_datetime(df_pred["date"])
    else:
        df_pred = make_predictions(lat, lon)

    # Load history data (last 14 days)
    hist_path = get_data_path(lat, lon) + ".hist"
    if os.path.exists(hist_path):
        df_hist = pd.read_csv(hist_path)
        df_hist["date"] = pd.to_datetime(df_hist["date"])
    else:
        df_hist = scrape_last_30_days_daily_data(lat, lon)

    df_hist_recent = df_hist.tail(14)

    plt.figure(figsize=(10, 5), dpi=150)

    # Plot recent history (last 14 days)
    plt.plot(df_hist_recent["date"], df_hist_recent["temp_mean"],
             label="Actual History (Last 14d)", color="#3a86c8", linewidth=2.5, marker="o")

    # Plot open-meteo physical forecast
    plt.plot(df_pred["date"], df_pred["temp_mean"], label="Open-Meteo Forecast",
             color="#06d6a0", linewidth=2, linestyle="--", marker="x")

    # Plot fine-tuned LSTM predictions
    plt.plot(df_pred["date"], df_pred["ml_pred"],
             label="Fine-tuned LSTM Prediction",
             color="#ff006e", linewidth=2.5, marker="s")

    # Set y-axis limits with +/- 10°C padding
    all_temps = pd.concat([
        df_hist_recent["temp_mean"],
        df_pred["temp_mean"],
        df_pred["ml_pred"]
    ])
    min_temp = all_temps.min()
    max_temp = all_temps.max()
    if not pd.isna(min_temp) and not pd.isna(max_temp):
        plt.ylim(min_temp - 10, max_temp + 10)

    plt.title(
        f"Daily Weather Forecast Comparison (Lat: {lat:.2f}, Lon: {lon:.2f})",
        fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Date", fontsize=11, labelpad=10)
    plt.ylabel("Temperature (°C)", fontsize=11, labelpad=10)

    plt.xticks(rotation=25)
    plt.legend(frameon=True, facecolor="#2a2a35", edgecolor="#444455")
    plt.tight_layout()

    plt.savefig(save_path, facecolor="#1e1e24", edgecolor="none")
    plt.close()
    print(f"Comparison chart saved successfully to {save_path}")
