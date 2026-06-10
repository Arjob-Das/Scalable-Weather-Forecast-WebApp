FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install CPU-only PyTorch.
# The service runs inference + light fine-tuning on CPU.
# Full CUDA training is done LOCALLY via train_local.py before this image is built.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source files
COPY app.py model.py train_city.py worldcities.csv ./
COPY weather_cache/ ./weather_cache/

# Bake pre-trained model artefacts produced locally by train_local.py.
# These MUST exist before running `docker build`:
#   pretrained/global_weather_model.pth   — LSTM weights (128 hidden, 3 layers)
#   pretrained/model_config.json          — architecture hyperparams
#   pretrained/scaler_params.json         — global normalisation stats
#   pretrained/training_cities.json       — lat/lon index of 1000 training cities
#
# Stored in TWO locations:
#   /app/pretrained_baked/   — read-only source (always present in image layer)
#   /app/pretrained/         — writable target (K8s PVC mounted here at runtime;
#                              initContainer copies from _baked on first start)
COPY pretrained/ ./pretrained_baked/
COPY pretrained/ ./pretrained/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
