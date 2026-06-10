"""
train_city.py  —  runs INSIDE the container / Kubernetes pod.
================================================================

Used exclusively for retraining the global model when a NEW city
(not in the original 1000) is requested.

Behaviour
---------
1. Download 365 days of daily weather data for the new city only.
2. Load the existing global model weights as a starting point.
3. Fine-tune (continue training) on the new city's data.
4. Save the updated weights back to pretrained/global_weather_model.pth.
5. Add the new city to training_cities.json so it is not flagged again.

This script is intentionally lightweight — it runs on CPU inside
the service container (no GPU required) and completes in minutes.
Full 1000-city training with CUDA is done LOCALLY via train_local.py
before the Docker image is built.
"""
import pandas as pd
import os
import sys
import json
import time
import datetime
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ------------------------------------------------------------------ #
# Paths  (same layout as model.py)
# ------------------------------------------------------------------ #

_BASE = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_DIR = os.path.join(_BASE, "pretrained")
MODEL_PATH = os.path.join(PRETRAINED_DIR, "global_weather_model.pth")
SCALER_PATH = os.path.join(PRETRAINED_DIR, "scaler_params.json")
CONFIG_PATH = os.path.join(PRETRAINED_DIR, "model_config.json")
CITIES_INDEX_PATH = os.path.join(PRETRAINED_DIR, "training_cities.json")

# Training hyperparams for new-city adaptation
NEW_CITY_EPOCHS = 20    # more epochs than the 10-epoch fine-tune in model.py
NEW_CITY_LR = 1e-4      # conservative LR to avoid forgetting
NEW_CITY_BATCH_SIZE = 32
NEW_CITY_DATA_DAYS = 365  # 1 full year of history for the new city

# Default arch (fallback if model_config.json absent)
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
# Model definition (must stay in sync with train_local.py)
# ------------------------------------------------------------------ #


class WeatherLSTM(nn.Module):

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
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_len * output_dim)
        self.output_len = output_len
        self.output_dim = output_dim

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).view(-1, self.output_len, self.output_dim)


# ------------------------------------------------------------------ #
# Data fetching for new city
# ------------------------------------------------------------------ #


def fetch_city_data(lat: float, lon: float, days: int = NEW_CITY_DATA_DAYS):
    """
    Download `days` of historical daily weather for (lat, lon) from
    Open-Meteo Archive API.
    """
    today = datetime.date.today()
    start_date = (today - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = (today - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum",
        "timezone": "auto",
    }

    print(
        f"[train_city] Downloading {days} days of data for lat={lat}, lon={lon}...")
    resp = requests.get(url, params=params, timeout=120)

    if resp.status_code != 200:
        print("[train_city] Archive API failed — falling back to forecast past_days.")
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum",
            "past_days": min(days, 92),  # forecast API caps at 92 past days
            "timezone": "auto",
        }
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()

    daily = resp.json().get("daily", {})

    df = pd.DataFrame({
        "date": pd.to_datetime(daily.get("time")),
        "temp_mean": daily.get("temperature_2m_mean"),
        "humidity_mean": daily.get("relative_humidity_2m_mean"),
        "precipitation": daily.get("precipitation_sum"),
    }).ffill().bfill().dropna()

    print(f"[train_city] Downloaded {len(df)} rows.")
    return df


# ------------------------------------------------------------------ #
# Main retrain routine
# ------------------------------------------------------------------ #


def retrain_for_new_city(lat: float, lon: float) -> None:
    """
    Continue training the global model on 365 days of data from a new city.
    Updates global_weather_model.pth and training_cities.json in-place.
    """
    print(
        f"\n[train_city] ===== Retraining for new city lat={lat}, lon={lon} =====")

    # ---- 1. Load config ------------------------------------------
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
    else:
        cfg = dict(_DEFAULT_CONFIG)
        print("[train_city] Warning: model_config.json not found — using defaults.")

    if not os.path.exists(SCALER_PATH):
        print("[train_city] ERROR: scaler_params.json not found — cannot scale data.")
        sys.exit(1)

    with open(SCALER_PATH) as f:
        scaler = json.load(f)

    # ---- 2. Download new city data --------------------------------
    df = fetch_city_data(lat, lon, days=NEW_CITY_DATA_DAYS)
    if len(df) < cfg["seq_len"] + cfg["output_len"]:
        print(f"[train_city] Insufficient data ({len(df)} rows) — aborting.")
        sys.exit(1)

    # Save downloaded 365-day weather data to a persistent city file under models directory
    models_dir = os.path.join(os.path.dirname(PRETRAINED_DIR), "models")
    os.makedirs(models_dir, exist_ok=True)
    city_file_path = os.path.join(
        models_dir, f"data_{round(lat, 2)}_{round(lon, 2)}_365.csv")
    df.to_csv(city_file_path, index=False)
    print(
        f"[train_city] Saved downloaded 365-day city data with dates to: {city_file_path}")

    # ---- 3. Scale --------------------------------------------------
    temp_s = (df["temp_mean"].values - scaler["temp_mean"]) / \
        scaler["temp_std"]
    hum_s = (df["humidity_mean"].values -
             scaler["humidity_mean"]) / scaler["humidity_std"]
    prec_s = (df["precipitation"].values -
              scaler["precipitation_mean"]) / scaler["precipitation_std"]

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

    arr = np.stack(
        [
            temp_s,
            hum_s,
            prec_s,
            year_scaled,
            month_scaled,
            day_scaled,
            sin_month,
            cos_month
        ],
        axis=1
    ).astype(np.float32)

    # ---- 4. Build sequences ----------------------------------------
    seq_len = cfg["seq_len"]
    pred_len = cfg["output_len"]
    xs, ys = [], []
    for i in range(len(arr) - seq_len - pred_len + 1):
        xs.append(arr[i: i + seq_len])
        # Predict only the 3 weather outcomes (temp, humidity, precipitation)
        ys.append(arr[i + seq_len: i + seq_len + pred_len, :3])

    print(
        f"[train_city] Built {len(xs)} training sequences from {len(df)} rows.")

    X_t = torch.from_numpy(np.array(xs))
    Y_t = torch.from_numpy(np.array(ys))

    loader = DataLoader(
        TensorDataset(X_t, Y_t),
        batch_size=NEW_CITY_BATCH_SIZE,
        shuffle=True,
    )

    # ---- 5. Load existing global model ----------------------------
    device = torch.device("cpu")  # CPU inside container
    model = WeatherLSTM(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        output_len=cfg["output_len"],
        output_dim=cfg["output_dim"],
        dropout=cfg["dropout"],
    )

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        )
        print("[train_city] Loaded existing global model weights.")
    else:
        print("[train_city] Warning: no existing model found — training from scratch.")

    model = model.to(device)
    model.train()

    # ---- 6. Optimise with low LR (avoid catastrophic forgetting) --
    optimizer = optim.Adam(model.parameters(), lr=NEW_CITY_LR)
    criterion = nn.MSELoss()

    print(
        f"[train_city] Fine-tuning for {NEW_CITY_EPOCHS} epochs on new city data...\n")

    best_loss = float("inf")
    for epoch in range(NEW_CITY_EPOCHS):
        epoch_loss = 0.0
        t0 = time.time()
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * bx.size(0)
        epoch_loss /= len(X_t)
        elapsed = time.time() - t0
        print(
            f"  Epoch [{epoch+1:>2}/{NEW_CITY_EPOCHS}] | "
            f"Loss: {epoch_loss:.6f} | Time: {elapsed:.1f}s"
        )
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # Save updated weights immediately (atomic on success)
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\n[train_city] Best loss: {best_loss:.6f}")
    print(f"[train_city] Updated model saved → {MODEL_PATH}")

    # ---- 7. Register new city in training_cities.json -------------
    coords = []
    if os.path.exists(CITIES_INDEX_PATH):
        with open(CITIES_INDEX_PATH) as f:
            coords = json.load(f)

    new_entry = {"lat": round(lat, 4), "lon": round(lon, 4)}
    if new_entry not in coords:
        coords.append(new_entry)
        with open(CITIES_INDEX_PATH, "w") as f:
            json.dump(coords, f)
        print(
            f"[train_city] City registered in training_cities.json ({len(coords)} total).")
    else:
        print("[train_city] City already in training_cities.json.")

    print("[train_city] ===== Done =====\n")


# ------------------------------------------------------------------ #
# CLI entry point
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Retrain global model for a single new city (CPU, runs in container)"
    )
    parser.add_argument("--lat", type=float, required=True, help="Latitude")
    parser.add_argument("--lon", type=float, required=True, help="Longitude")
    args = parser.parse_args()
    retrain_for_new_city(args.lat, args.lon)
