from datetime import timedelta
from datetime import datetime
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import platform
import time
import json
import sys
import os
import zipfile

# Try importing third-party libraries. If any fail, run ensure_dependencies() first, then import them.
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import pandas as pd
    import numpy as np
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print("Dependencies missing, triggering automatic dependency check and installation...")
    # On Windows, DataLoader workers use the 'spawn' start method, which reimports
    # this module in every worker process. Calling pip installs from workers causes
    # hangs and race conditions. But child processes will only be spawned AFTER
    # the main process has successfully installed everything and imported them.
    # If imports fail in a worker, it is abnormal, but we guard by checking if we are '__main__'.
    if __name__ == "__main__":
        libs = [
            "requests",
            "pandas",
            "numpy",
            "scikit-learn"
        ]

        for lib in libs:
            try:
                __import__(lib.replace("-", "_"))
            except ImportError:
                print(f"Installing missing library: {lib}")
                subprocess.check_call([
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    lib
                ])

        try:
            import torch  # noqa: F401
            if not torch.cuda.is_available():
                print("CUDA not available in current PyTorch installation.")
                raise ImportError
        except ImportError:
            print("Installing CUDA-enabled PyTorch...")
            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch",
                "--index-url",
                "https://download.pytorch.org/whl/cu121"
            ])

    # Retry imports after installation
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import pandas as pd
    import numpy as np
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

# Global lock + rate limiter

request_lock = Lock()
last_request_time = [0]


def safe_str(s):
    return str(s).encode("ascii", "replace").decode("ascii")


# ============================================================
# 1. Automatic Dependency Check and Installer
# ============================================================


def ensure_dependencies():
    print("Dependencies already checked and resolved.")


# ============================================================
# 2. Imports
# ============================================================


# ============================================================
# 3. Random Seeds
# ============================================================

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ============================================================
# 4. Directories
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PRETRAINED_DIR = os.path.join(BASE_DIR, "pretrained")
CACHE_DIR = os.path.join(BASE_DIR, "weather_cache")

os.makedirs(PRETRAINED_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================================
# Model Configuration — single source of truth.
# Written to pretrained/model_config.json so model.py in the
# service can instantiate an identical architecture.
# ============================================================

def get_dynamic_model_config(quiet=True):
    hidden_dim = 128
    num_layers = 3
    
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
                
            if not quiet:
                print(f"CUDA Free VRAM: {free_vram_gb:.2f} GB. Selected: hidden_dim={hidden_dim}, num_layers={num_layers}")
        except Exception:
            hidden_dim = 256
            num_layers = 3
    else:
        if not quiet:
            print("CUDA not available. Selected: hidden_dim=128, num_layers=3")
            
    return {
        "input_dim": 8,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "output_len": 7,
        "output_dim": 3,
        "dropout": 0.3,
        "seq_len": 30,
    }

MODEL_CONFIG = get_dynamic_model_config(quiet=True)

# ============================================================
# 5. Load Top Cities
# ============================================================


def download_worldcities_dataset():

    csv_path = os.path.join(BASE_DIR, "worldcities.csv")

    if os.path.exists(csv_path):
        print("worldcities.csv already exists.")
        return csv_path

    print("Downloading world cities dataset...")

    zip_url = (
        "https://simplemaps.com/static/data/"
        "world-cities/basic/simplemaps_worldcities_basicv1.76.zip"
    )

    zip_path = os.path.join(BASE_DIR, "worldcities.zip")

    response = requests.get(zip_url, stream=True)

    response.raise_for_status()

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Extracting dataset...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(BASE_DIR)

    os.remove(zip_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "worldcities.csv not found after extraction."
        )

    print("worldcities.csv downloaded successfully.")

    return csv_path


def load_top_cities(limit=10000):

    csv_path = download_worldcities_dataset()

    print("Loading city dataset...")

    df = pd.read_csv(csv_path)

    required_cols = [
        "city",
        "lat",
        "lng",
        "population"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df[required_cols]

    df = df.dropna(subset=["population"])

    df = df.sort_values(
        by="population",
        ascending=False
    )

    df = df.head(limit)

    cities = []

    for _, row in df.iterrows():

        cities.append({
            "name": str(row["city"]),
            "lat": float(row["lat"]),
            "lon": float(row["lng"])
        })

    print(f"Loaded top {len(cities)} cities.")

    return cities

# ============================================================
# 6. Weather Data Scraper + Cache + Retry Logic
# ============================================================


# ------------------------------------------------------------
# Global session with retry support
# ------------------------------------------------------------

session = requests.Session()

retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[
        500,
        502,
        503,
        504
    ],
    allowed_methods=["GET"]
)

adapter = HTTPAdapter(
    max_retries=retry_strategy
)

session.mount("http://", adapter)
session.mount("https://", adapter)

# ============================================================
# Weather Download Function
# ============================================================


# ============================================================
# Meteostat Downloader
# ============================================================


def scrape_daily_data(city):

    safe_name = (
        city["name"]
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    cache_file = os.path.join(
        CACHE_DIR,
        f"{safe_name}.csv"
    )

    # --------------------------------------------------------
    # Use cache
    # --------------------------------------------------------

    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file)
            if not df.empty and "date" in df.columns:
                print(f"[CACHE] {safe_str(city['name'])}")
                return df
        except Exception:
            pass

    try:

        print(f"[DOWNLOAD] {safe_str(city['name'])}")

        url = (
            "https://archive-api.open-meteo.com/v1/archive"
        )

        one_week_before_current_date = (
            datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        params = {

            "latitude": city["lat"],
            "longitude": city["lon"],

            "start_date": "2022-01-01",
            "end_date": one_week_before_current_date,

            "daily": (
                "temperature_2m_mean,"
                "precipitation_sum,"
                "relative_humidity_2m_mean"
            ),

            "timezone": "auto"
        }

        with request_lock:

            now = time.time()

            elapsed = now - last_request_time[0]

            MIN_DELAY = 1.2

            if elapsed < MIN_DELAY:
                time.sleep(MIN_DELAY - elapsed)

            response = session.get(
                url,
                params=params,
                timeout=60
            )

            last_request_time[0] = time.time()

        response.raise_for_status()

        data = response.json()

        daily_data = data.get("daily")

        if not daily_data:
            print(f"[EMPTY] {safe_str(city['name'])}")
            return pd.DataFrame()

        df = pd.DataFrame({
            "date": daily_data["time"],
            "temp_mean": daily_data["temperature_2m_mean"],
            "humidity_mean": daily_data["relative_humidity_2m_mean"],
            "precipitation": daily_data["precipitation_sum"]
        })

        df = (
            df
            .ffill()
            .bfill()
            .dropna()
        )

        df.to_csv(
            cache_file,
            index=False
        )

        print(
            f"[SUCCESS] "
            f"{safe_str(city['name'])} "
            f"({len(df)} rows)"
        )

        return df

    except Exception as e:
        print(
            f"[FAILED] "
            f"{safe_str(city['name'])} : {e}"
        )
        return pd.DataFrame()


# ============================================================
# 7. PyTorch Dataset
# ============================================================


class WeatherDataset(Dataset):

    def __init__(
        self,
        city_dfs,
        scaler_params,
        seq_len=30,
        pred_len=7
    ):

        self.samples = []

        print("Building training sequences...")

        for df in city_dfs:

            try:

                temp_scaled = (
                    (df["temp_mean"] -
                     scaler_params["temp_mean"])
                    / scaler_params["temp_std"]
                )

                hum_scaled = (
                    (df["humidity_mean"] -
                     scaler_params["humidity_mean"])
                    / scaler_params["humidity_std"]
                )

                prec_scaled = (
                    (df["precipitation"] -
                     scaler_params["precipitation_mean"])
                    / scaler_params["precipitation_std"]
                )

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

                scaled_data = np.stack(
                    [
                        temp_scaled,
                        hum_scaled,
                        prec_scaled,
                        year_scaled,
                        month_scaled,
                        day_scaled,
                        sin_month,
                        cos_month
                    ],
                    axis=1
                )

                for i in range(
                    len(scaled_data)
                    - seq_len
                    - pred_len
                    + 1
                ):

                    x = scaled_data[
                        i: i + seq_len
                    ]

                    # Predict only the 3 weather outcomes (temp, humidity, precipitation)
                    y = scaled_data[
                        i + seq_len:
                        i + seq_len + pred_len,
                        :3
                    ]

                    self.samples.append((
                        torch.tensor(
                            x,
                            dtype=torch.float32
                        ),

                        torch.tensor(
                            y,
                            dtype=torch.float32
                        )
                    ))

            except Exception as e:
                print(f"Dataset build error: {e}")

        print(f"Total sequences: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ============================================================
# 8. LSTM Model
# ============================================================


class WeatherLSTM(nn.Module):

    def __init__(
        self,
        input_dim=MODEL_CONFIG["input_dim"],
        hidden_dim=MODEL_CONFIG["hidden_dim"],
        num_layers=MODEL_CONFIG["num_layers"],
        output_len=MODEL_CONFIG["output_len"],
        output_dim=MODEL_CONFIG["output_dim"],
        dropout=MODEL_CONFIG["dropout"],
    ):

        super(WeatherLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(
            hidden_dim,
            output_len * output_dim
        )

        self.output_len = output_len
        self.output_dim = output_dim

    def forward(self, x):

        lstm_out, _ = self.lstm(x)

        last_out = lstm_out[:, -1, :]

        out = self.fc(last_out)

        return out.view(
            -1,
            self.output_len,
            self.output_dim
        )

# ============================================================
# 9. Main
# ============================================================


def main():

    # Re-evaluate model configuration and print selection
    global MODEL_CONFIG
    MODEL_CONFIG = get_dynamic_model_config(quiet=False)

    # --------------------------------------------------------
    # Device info
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Force line-buffered stdout so every print() appears immediately
    # in the terminal (Python buffers stdout when not attached to a TTY).
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    print(f"\n{'='*60}")
    print(f"  Training device : {device}")
    if device.type == "cuda":
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM            : {vram:.1f} GB")
    print(f"{'='*60}\n")

    # --------------------------------------------------------
    # Load Cities
    # --------------------------------------------------------

    CITIES = load_top_cities(limit=100)

    # Count cached cities
    cached_count = 0
    for city in CITIES:
        safe_name = (
            city["name"]
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )
        cache_file = os.path.join(CACHE_DIR, f"{safe_name}.csv")
        if os.path.exists(cache_file):
            try:
                df_test = pd.read_csv(cache_file)
                if not df_test.empty and "date" in df_test.columns:
                    cached_count += 1
            except Exception:
                pass

    print(
        f"Found {cached_count} cached cities out of {len(CITIES)} total cities. Processing all cities...")

    # --------------------------------------------------------
    # Download Weather Data
    # --------------------------------------------------------

    print("\nDownloading weather data...\n")

    all_dfs = []

    MAX_WORKERS = 100

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = [
            executor.submit(
                scrape_daily_data,
                city
            )
            for city in CITIES
        ]

        completed = 0

        for future in as_completed(futures):

            try:

                df = future.result()

                if not df.empty:
                    all_dfs.append(df)

                completed += 1

                if completed % 10 == 0:

                    print(
                        f"Completed "
                        f"{completed}/{len(CITIES)}"
                    )

                # Tiny delay prevents API hammering
                time.sleep(0.1)

            except Exception as e:

                print(f"Thread error: {e}")

    if len(all_dfs) == 0:
        print("No weather data collected.")
        return

    print(
        f"\nCollected data from "
        f"{len(all_dfs)} cities.\n"
    )

    # --------------------------------------------------------
    # Combine Data
    # --------------------------------------------------------

    full_df = pd.concat(
        all_dfs,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Compute Scaler Parameters
    # --------------------------------------------------------

    scaler_params = {

        "temp_mean":
            float(full_df["temp_mean"].mean()),

        "temp_std":
            float(full_df["temp_mean"].std()),

        "humidity_mean":
            float(full_df["humidity_mean"].mean()),

        "humidity_std":
            float(full_df["humidity_mean"].std()),

        "precipitation_mean":
            float(full_df["precipitation"].mean()),

        "precipitation_std":
            float(full_df["precipitation"].std())
    }

    scaler_path = os.path.join(
        PRETRAINED_DIR,
        "scaler_params.json"
    )

    with open(scaler_path, "w") as f:
        json.dump(
            scaler_params,
            f,
            indent=4
        )

    print(
        f"Scaler parameters saved:\n"
        f"{scaler_path}"
    )

    # --------------------------------------------------------
    # Save model_config.json — consumed by model.py so the
    # service always instantiates the exact same architecture.
    # --------------------------------------------------------

    config_path = os.path.join(
        PRETRAINED_DIR,
        "model_config.json"
    )

    with open(config_path, "w") as f:
        json.dump(MODEL_CONFIG, f, indent=4)

    print(f"Model config saved: {config_path}")

    # --------------------------------------------------------
    # Save training cities index (used by model.py for new-city detection)
    # --------------------------------------------------------

    cities_index_path = os.path.join(
        PRETRAINED_DIR,
        "training_cities.json"
    )

    cities_index = [
        {"name": c["name"], "lat": c["lat"], "lon": c["lon"]}
        for c in CITIES
    ]

    with open(cities_index_path, "w") as f:
        json.dump(cities_index, f)

    print(f"Training cities index saved: {cities_index_path}")

    # --------------------------------------------------------
    # Build Dataset
    # --------------------------------------------------------

    dataset = WeatherDataset(
        city_dfs=all_dfs,
        scaler_params=scaler_params,
        seq_len=MODEL_CONFIG["seq_len"],
        pred_len=MODEL_CONFIG["output_len"]
    )

    print(f"\nTraining on device: {device}")

    # --------------------------------------------------------
    # DataLoader
    # On Windows, DataLoader workers use 'spawn' and must
    # pickle the entire dataset to send to each worker.
    # With 427k+ samples this causes a silent hang/deadlock.
    # With batch_size ~16k there are only ~26 batches/epoch —
    # a single CPU thread keeps the GPU fully fed at this scale,
    # so num_workers=0 is both safe and fast enough here.
    # On Linux (container / K8s), workers are useful.
    # --------------------------------------------------------

    num_workers = 0 if platform.system() == "Windows" else 4

    # --------------------------------------------------------
    # Auto batch size — target ~60% of available VRAM.
    # Each sample: seq_len * input_dim * 4 bytes (float32)
    # With AMP the activation memory is ~half, but we budget
    # conservatively for optimizer states + activations.
    # --------------------------------------------------------
    if device.type == "cuda":

        free_vram, total_vram = torch.cuda.mem_get_info()

        print(f"Free VRAM : {free_vram / 1024**3:.2f} GB")
        print(f"Total VRAM: {total_vram / 1024**3:.2f} GB")

        hidden_dim = MODEL_CONFIG["hidden_dim"]
        seq_len = MODEL_CONFIG["seq_len"]
        num_layers = MODEL_CONFIG["num_layers"]
        input_dim = MODEL_CONFIG["input_dim"]
        output_dim = MODEL_CONFIG["output_dim"]
        output_len = MODEL_CONFIG["output_len"]

        # --------------------------------------------------------
        # More realistic LSTM memory estimate
        # --------------------------------------------------------

        # fp16 = 2 bytes, fp32 gradients/optimizer still exist
        bytes_per_float = 4

        # Input tensor
        input_mem = (
            seq_len *
            input_dim *
            bytes_per_float
        )

        # Output tensor
        output_mem = (
            output_len *
            output_dim *
            bytes_per_float
        )

        # --------------------------------------------------------
        # LSTM activations dominate memory usage
        #
        # Approximation:
        # 8 gates/states per timestep/layer
        # activations + gradients + optimizer states
        # --------------------------------------------------------

        lstm_activation_mem = (
            seq_len *
            hidden_dim *
            num_layers *
            2 *
            bytes_per_float
        )

        # Gradients + Adam optimizer states
        optimizer_overhead = 1.6

        bytes_per_sample = int(
            (
                input_mem +
                output_mem +
                lstm_activation_mem
            ) * optimizer_overhead
        )

        print(
            f"Estimated bytes/sample: "
            f"{bytes_per_sample / 1024:.1f} KB"
        )

        # Use only part of free VRAM
        target_vram = int(free_vram * 0.70)

        batch_size = target_vram // bytes_per_sample

        # Safety clamp
        batch_size = max(128, min(batch_size, 4096))

        # Round DOWN to power-of-2
        batch_size = 2 ** (
            int(batch_size).bit_length() - 1
        )

        print(
            f"Auto batch size: {batch_size:,} "
            f"(targeting "
            f"{((target_vram*0.9)/0.7)/(1024**3):.2f} GB)"
        )
    else:
        batch_size = 128
        print(f"  Batch size (CPU): {batch_size}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = WeatherLSTM().to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # LR scheduler — halve LR after 5 epochs without improvement
    # verbose=True is deprecated since PyTorch 2.2; use get_last_lr() manually.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5,
        factor=0.5,
    )

    # --------------------------------------------------------
    # Mixed Precision Training (non-deprecated API)
    # --------------------------------------------------------

    use_amp = device.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --------------------------------------------------------
    # Early stopping configuration
    # --------------------------------------------------------

    early_stop_patience = 5

    min_improvement = 0.01
    max_allowed_increase = 0.05

    epochs_without_improvement = 0

    previous_loss = None

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    epochs = 50

    print("\nStarting training...\n")

    model.train()

    best_loss = float("inf")
    model_path = os.path.join(
        PRETRAINED_DIR,
        "global_weather_model.pth"
    )

    for epoch in range(epochs):

        epoch_loss = 0.0

        start_time = time.time()

        for batch_X, batch_y in dataloader:

            batch_X = batch_X.to(device, non_blocking=True)

            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):

                outputs = model(batch_X)

                loss = criterion(
                    outputs,
                    batch_y
                )

            amp_scaler.scale(loss).backward()

            amp_scaler.step(optimizer)

            amp_scaler.update()

            epoch_loss += (
                loss.item()
                * batch_X.size(0)
            )

        epoch_loss /= len(dataset)

        elapsed = time.time() - start_time

        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch "
            f"[{epoch+1}/{epochs}] | "
            f"Loss: {epoch_loss:.6f} | "
            f"LR: {lr_now:.2e} | "
            f"Time: {elapsed:.2f}s",
            flush=True
        )

        # ----------------------------------------------------
        # Early stopping logic
        # ----------------------------------------------------

        if previous_loss is not None:

            loss_delta = previous_loss - epoch_loss

            # Good improvement
            if loss_delta >= min_improvement:

                epochs_without_improvement = 0

            # Loss increased significantly
            elif epoch_loss > previous_loss + max_allowed_increase:

                epochs_without_improvement += 1

                print(
                    f"  [WARN] Significant loss increase detected "
                    f"(+{epoch_loss - previous_loss:.6f}) "
                    f"| patience "
                    f"{epochs_without_improvement}/"
                    f"{early_stop_patience}"
                )

            # Tiny improvement / plateau
            else:

                epochs_without_improvement += 1

                print(
                    f"  * No meaningful improvement "
                    f"(|Delta| < {min_improvement}) "
                    f"| patience "
                    f"{epochs_without_improvement}/"
                    f"{early_stop_patience}"
                )

            # Trigger early stopping
            if epochs_without_improvement >= early_stop_patience:

                print("\n====================================")
                print("Early stopping triggered")
                print("====================================")

                break

        previous_loss = epoch_loss

        # Step LR scheduler — detect and log LR changes manually
        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(epoch_loss)
        lr_after = optimizer.param_groups[0]["lr"]
        if lr_after < lr_before:
            print(f"  [LR] reduced: {lr_before:.2e} -> {lr_after:.2e}")

        # Save best model weights
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), model_path)
            print(f"  [OK] Best model saved (loss={best_loss:.6f})")

        # ----------------------------------------------------
        # Save checkpoint every 5 epochs
        # ----------------------------------------------------

        if (epoch + 1) % 5 == 0:

            checkpoint_path = os.path.join(
                PRETRAINED_DIR,
                f"checkpoint_epoch_{epoch+1}.pth"
            )

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            print(
                f"Checkpoint saved:\n"
                f"{checkpoint_path}"
            )

            # VRAM utilisation report
            if device.type == "cuda":
                alloc = torch.cuda.memory_allocated(0) / 1024**3
                peak = torch.cuda.max_memory_allocated(0) / 1024**3

                print(
                    f"Allocated: {alloc:.2f} GB | "
                    f"Peak: {peak:.2f} GB"
                )
                reserv = torch.cuda.memory_reserved(0) / 1024**3
                total = torch.cuda.get_device_properties(
                    0).total_memory / 1024**3
                print(
                    f"  GPU VRAM — allocated: {alloc:.2f} GB | "
                    f"reserved: {reserv:.2f} GB | "
                    f"total: {total:.1f} GB  "
                    f"({reserv/total*100:.0f}% utilised)"
                )

    print("\n====================================")
    print("Training Complete")
    print("====================================")
    print(f"Best loss  : {best_loss:.6f}")
    print(f"Model      : {model_path}")
    print(f"Config     : {config_path}")
    print(f"Scaler     : {scaler_path}")

# ============================================================
# 10. Run
# ============================================================


if __name__ == "__main__":
    # Must be called before any other multiprocessing code on Windows
    # (required when using num_workers > 0 in DataLoader with spawn).
    import multiprocessing
    multiprocessing.freeze_support()

    # Install dependencies in the main process only — never in workers.
    ensure_dependencies()

    main()
