# scalable: Weather Forecasting System

This repository contains four core services:
- `frontend` — React + TypeScript + Vite application served by Nginx
- `backend` — Spring Boot API
- `ml-service` — FastAPI service
- `postgres` — PostgreSQL database

---

## 1. System Architecture & Flow

The ML pipeline has two distinct training stages with a strict execution order:

```
Step 1 (Local, CUDA GPU)          Step 2 (Docker / K8s, CPU)
─────────────────────────         ────────────────────────────────────────
python train_local.py             Container starts
  │  1000 cities × 4 years            load_pretrained_model()
  │  50 epochs, CUDA                  ↓
  │  └─ produces pretrained/        GET /api/ml/predict?lat=X&lon=Y
  │       ├── global_weather_model.pth   ↓ known city?
  │       ├── model_config.json          ├── YES → fine-tune (10 epochs, cache 7d)
  │       ├── scaler_params.json         └── NO  → train_city.py (365 days, CPU)
  │       └── training_cities.json             → reload model, then fine-tune
  │                  ↓
  │          docker build ./ml-service
  │                  ↓
  │          docker compose up  OR  kubectl apply
```

### Prediction Flow (Inside ML Container)

| Request | Action |
|---------|--------|
| Known city (within 0.15°) | Fine-tune on last 30 days → cache for 7 days |
| Cache hit (< 7 days old) | Load cached fine-tuned weights → inference directly |
| **Unknown city** | Trigger `train_city.py --lat X --lon Y` in background thread |

#### New-City Retraining (`train_city.py`)
- Downloads 365 days of historical data for the new city only.
- Continues training from existing global weights (LR=1e-4, 20 epochs).
- Saves updated `global_weather_model.pth` and updates `training_cities.json`.
- Runs entirely on **CPU** inside the container (no GPU needed).
- Current request is still served with the pre-update model.

---

## 2. File Layout

```
scalable/
├── backend/                # Spring Boot API
├── frontend/               # React + TypeScript + Vite Application
│   └── README.md           # [Removed during consolidation]
├── k8s/                    # Kubernetes Manifests
├── ml-service/             # FastAPI ML Service
│   ├── train_local.py      # Run LOCALLY with CUDA (Step 1)
│   ├── train_city.py       # Runs INSIDE container for new cities
│   ├── model.py            # Service inference + fine-tuning engine
│   ├── app.py              # FastAPI application
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── pretrained/         # Must be populated by train_local.py BEFORE docker build
│   │   ├── global_weather_model.pth
│   │   ├── model_config.json
│   │   ├── scaler_params.json
│   │   └── training_cities.json
│   ├── models/             # Runtime: fine-tuned cache + prediction data
│   │   └── finetuned/
│   │       ├── finetuned_28.61_77.21.pth
│   │       └── finetuned_28.61_77.21.meta.json
│   └── weather_cache/      # Cached historical data for train_local.py
│   └── README_TRAINING.md  # [Removed during consolidation]
├── deploy.ps1              # Deployment script for Kubernetes
└── docker-compose.yml      # Local multi-container deployment configuration
```

---

## 3. Environment Setup

### Windows Installation

#### Install Git
Download and install: [Git for Windows](https://git-scm.com/download/win)
Verify installation:
```powershell
git --version
```

#### Install Docker Desktop
Download and install: [Docker Desktop](https://www.docker.com/products/docker-desktop/)
During installation:
- Enable WSL2 integration
- Restart the system if prompted
Verify installation:
```powershell
docker --version
docker compose version
```

#### Enable Kubernetes in Docker Desktop
Steps:
1. Open Docker Desktop Settings.
2. Go to **Settings → Kubernetes**.
3. Enable Kubernetes.
4. Click **Apply & Restart**.
Verify:
```powershell
kubectl version --client
kubectl get nodes
```

#### Install Node.js 22+
Download and install: [Node.js](https://nodejs.org/en/download)
Verify installation:
```powershell
node -v
npm -v
```

#### Install Java 21
Download and install [Oracle JDK 21](https://www.oracle.com/java/technologies/downloads/#java21) or [Eclipse Temurin OpenJDK 21](https://adoptium.net/temurin/releases/).
Verify installation:
```powershell
java -version
javac -version
```

#### Install Python 3.12+
Download and install: [Python](https://www.python.org/downloads/windows/)
During installation:
- Enable **Add Python to PATH**
Verify installation:
```powershell
python --version
pip --version
```

---

### Linux Installation (Ubuntu/Debian-based)

#### Update System Packages
```bash
sudo apt update
sudo apt upgrade -y
```

#### Install Git
```bash
sudo apt install -y git
git --version
```

#### Install Docker Engine
Official documentation: [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
Quick setup:
```bash
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add current user to docker group
sudo usermod -aG docker $USER
newgrp docker
```
Verify:
```bash
docker --version
docker compose version
```

#### Install kubectl
Official documentation: [Install and Set Up kubectl on Linux](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/)
```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/
kubectl version --client
```

#### Install Minikube
Official documentation: [Minikube Start Guide](https://minikube.sigs.k8s.io/docs/start/)
```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
minikube start --driver=docker
kubectl get nodes
```

#### Install Node.js 22+
```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
node -v
npm -v
```

#### Install Java 21
```bash
sudo apt install -y openjdk-21-jdk
java -version
javac -version
```

#### Install Maven
```bash
sudo apt install -y maven
mvn -version
```

#### Install Python 3.12+
```bash
sudo apt install -y python3 python3-pip python3-venv
python3 --version
pip3 --version
```

---

## 4. Repository & Local Build Setup

### Clone Repository
```powershell
git clone https://github.com/Arjob-Das/Ai-Agent-Benchmarks.git
cd "Scalable-Weather-Forecast-WebApp"
```

### API Key Configuration
The application requires an **OpenWeather API Key** to fetch geocoding information and current weather data.

Before running the application, you must replace the `YOUR_OPENWEATHER_API_KEY` placeholder with your actual OpenWeather API key in the following locations:

1. **Docker Compose**: Set `OPENWEATHER_API_KEY` in [docker-compose.yml](file:///f:/Self_Study/Scalable%20Weather%20Forecast%20WebApp/docker-compose.yml).
2. **Kubernetes**: Update the env variable `OPENWEATHER_API_KEY` in [k8s/backend.yaml](file:///f:/Self_Study/Scalable%20Weather%20Forecast%20WebApp/k8s/backend.yaml).
3. **Local / Maven Execution**: Update the default fallback in [application.properties](file:///f:/Self_Study/Scalable%20Weather%20Forecast%20WebApp/backend/src/main/resources/application.properties) or set the environment variable `OPENWEATHER_API_KEY` on your machine:
   - **Windows (PowerShell)**: `$env:OPENWEATHER_API_KEY="your_api_key"`
   - **Linux/macOS (Bash)**: `export OPENWEATHER_API_KEY="your_api_key"`

### Local Build & Execution Commands

#### Complete Unified Build via Root Maven (Recommended)
A root `pom.xml` aggregates the backend and frontend modules. You can compile, build, and package both applications in a single command from the project root:
```bash
mvn clean package
```
This automatically:
1. Orchestrates the frontend build:
   - **Default Mode:** Runs `npm install` and `npm run build` using the system's globally installed Node/NPM for maximum speed and to avoid Windows Defender file locking/rename issues during compilation.
   - **Local Installation Mode:** If you do not have Node/NPM installed globally, or want to download a local isolated Node (`v22.11.0`) and NPM environment, you can run the build with the `download-node` profile:
     ```bash
     mvn clean package -Pdownload-node
     # or
     mvn clean package -DuseLocalNode=true
     ```
2. Compiles and packages the backend Java service (producing `backend/target/backend-0.0.1-SNAPSHOT.jar`).


---

#### Script Execution & Deployments via Maven
The root `pom.xml` defines Maven profiles that bind to the `exec-maven-plugin` to run training and deployments easily:

##### Local ML model training
```bash
mvn -Ptrain
```

##### Windows Kubernetes Deployment
* **Standard run:** `mvn -Pdeploy-windows`
* **Clean rebuild:** `mvn -Pdeploy-windows-rebuild`

##### Linux & macOS Kubernetes Deployment
* **Standard run:** `mvn -Pdeploy-linux`
* **Clean rebuild:** `mvn -Pdeploy-linux-rebuild`

##### Linux & macOS Kubernetes Deployment (Elevated Sudo/Root permissions)
* **Standard run:** `mvn -Pdeploy-linux-sudo`
* **Clean rebuild:** `mvn -Pdeploy-linux-sudo-rebuild`

---

#### Individual Local Build Commands

##### Frontend Build
```bash
cd frontend
npm install
npm run build
# Optional lint check
npm run lint
cd ..
```

##### Backend Build
```bash
cd backend
mvn clean package -DskipTests
cd ..
```

##### ML Service Setup
```bash
cd ml-service
pip install -r requirements.txt
cd ..
```

---

## 5. Machine Learning Pipeline: Training & API

### Step 1: Local Training (Must run BEFORE building Docker images)
This stage builds the global model using GPU acceleration.

#### Requirements
- NVIDIA GPU with CUDA 12.1+ drivers
- ~4 GB VRAM minimum

#### Execution
```bash
cd ml-service
python train_local.py
```
This script will:
1. Auto-install `torch` (CUDA 12.1 build) if not present.
2. Download `worldcities.csv` if not present.
3. Download 2021–2024 weather data for 1000 cities (cached in `weather_cache/`).
4. Train the global LSTM (50 epochs, AMP mixed precision on CUDA).
5. Save the global model artifacts to `pretrained/`:
   - `global_weather_model.pth` — Best model weights
   - `model_config.json` — Architecture hyperparameters
   - `scaler_params.json` — Global normalization statistics
   - `training_cities.json` — Latitude/Longitude index of the 1000 cities

> **CRITICAL**: Verify all 4 files exist in `ml-service/pretrained/` before moving to container build or deployment.

---

### Step 2: API & Prediction Services

Once local training is complete, the API service can run inside Docker or Kubernetes.

#### ML Service API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ml/predict` | GET | 7-day forecast with fine-tuned LSTM (e.g. `?latitude=28.61&longitude=77.21`) |
| `/api/ml/chart` | GET | Temperature comparison chart (PNG) |
| `/api/ml/train` | POST | No-op stub (global model is pre-trained) |
| `/api/ml/retrain-new-city` | POST | Explicitly trigger new-city retraining |
| `/api/ml/retrain-status` | GET | Check if retraining is running |

#### API Call Examples
```bash
# Get prediction forecast for New Delhi
curl "http://localhost:8000/api/ml/predict?latitude=28.61&longitude=77.21"

# Retrain/fine-tune for a new city
curl -X POST http://localhost:8000/api/ml/retrain-new-city \
  -H "Content-Type: application/json" \
  -d '{"latitude": 12.97, "longitude": 77.59}'
```

---

## 6. Docker Compose Deployment

From the repository root:

```bash
# Build and run all services
docker compose up --build

# Run in background mode
docker compose up -d --build

# Stop all services
docker compose down

# Stop all services and remove postgres volume
docker compose down -v
```

### Accessing Services (Docker Compose)
| Service    | URL                   |
| ---------- | --------------------- |
| Frontend   | http://localhost:3001 |
| Backend    | http://localhost:8080 |
| ML Service | http://localhost:8000 |
| PostgreSQL | localhost:5432        |

---

## 7. Kubernetes Deployment

### Build Docker Images
```bash
docker build -t scalable-frontend:latest .\frontend
docker build -t scalable-backend:latest .\backend
docker build -t scalable-ml-service:latest .\ml-service
```
*(On Linux/Minikube, remember to load the images):*
```bash
minikube image load scalable-frontend:latest
minikube image load scalable-backend:latest
minikube image load scalable-ml-service:latest
```

### Apply Kubernetes Resources
Apply all manifests in the `k8s/` folder:
```bash
kubectl apply -f .\k8s\
```
Or apply them individually:
```bash
kubectl apply -f .\k8s\postgres.yaml
kubectl apply -f .\k8s\ml-service.yaml
kubectl apply -f .\k8s\backend.yaml
kubectl apply -f .\k8s\frontend.yaml
```

### Verify Status & Rollout
```bash
kubectl get pods
kubectl get services

kubectl rollout status deployment/scalable-postgres
kubectl rollout status deployment/scalable-ml-service
kubectl rollout status deployment/scalable-backend
kubectl rollout status deployment/scalable-frontend
```

### Accessing Kubernetes Services
| Service     | URL                    |
| ----------- | ---------------------- |
| Frontend UI | http://localhost:30000 |
| Backend API | http://localhost:30080 |

### Automated Kubernetes Deployment

To simplify building, training, and deploying on local Kubernetes clusters, two automated scripts are provided in the repository root:
- `deploy.ps1` (for Windows environments using PowerShell and Docker Desktop Kubernetes)
- `deploy.sh` (for Linux/macOS environments using Bash and Minikube/Docker)

Both scripts automate local model pre-training, Python package installation, Docker image generation, and service configuration/rollout to the `scalable-weather-app` namespace.

---

#### 1. Windows Deployment (`deploy.ps1`)

##### Standard Execution
Runs Python dependencies installation, model training (if needed), builds Docker images, and deploys everything to Kubernetes:
```powershell
.\deploy.ps1
```

##### Clean Rebuild Execution
Forces a deep clean before rebuilding and deploying:
```powershell
.\deploy.ps1 rebuild
# or
.\deploy.ps1 -Mode rebuild
```
This will:
- Clean local pre-trained artifacts from `ml-service/pretrained/`.
- Clear the persistent Kubernetes volume data using a temporary container.
- Delete the entire Kubernetes namespace (removing all pods, deployments, PVCs, and services).
- Stop and remove local Docker Compose containers (via `docker compose down -v`).
- Remove the project's custom Docker images (`scalable-weather-app/*`).
- Re-run all builds and creation steps completely fresh.

##### Script Source (`deploy.ps1`)
Save the following as `deploy.ps1` in the project root:
```powershell
# ============================================================
#  scalable Weather App - Full Deploy Script
#  Target: Docker Desktop Kubernetes (Windows)
#
#  Usage:  .\deploy.ps1
#
#  What this script does (in order):
#    1. Install Python dependencies for training
#    2. Run train_local.py  (CUDA pre-training - skipped if artefacts already exist)
#    3. Build Docker images
#    4. Create K8s namespace
#    5. Deploy Postgres          + wait Ready
#    6. Deploy ML Service        + wait PVCs Bound + wait Ready
#    7. Deploy Backend           + wait Ready
#    8. Deploy Frontend          + wait Ready
# ============================================================
param(
    [string]$Mode = ""
)

$isRebuild = $false
if ($Mode -like "*rebuild*") {
    $isRebuild = $true
}
foreach ($arg in $args) {
    if ($arg -like "*rebuild*") {
        $isRebuild = $true
    }
}

$ErrorActionPreference = "Stop"
$NS = "scalable-weather-app"
$ROOT = $PSScriptRoot
$ML_DIR = "$ROOT\ml-service"
$TOTAL = 8

function Log-Step {
    param([int]$n, [string]$msg)
    Write-Host "`n[$(Get-Date -f 'HH:mm:ss')] >>> STEP $n/$TOTAL - $msg" -ForegroundColor Cyan 
}

function Log-OK {
    param([string]$msg)
    Write-Host "    [OK]   $msg" -ForegroundColor Green 
}

function Log-Warn {
    param([string]$msg)
    Write-Host "    [!!]   $msg" -ForegroundColor Yellow 
}

function Log-Info {
    param([string]$msg)
    Write-Host "           $msg" -ForegroundColor DarkGray 
}

function Log-Fail {
    param([string]$msg)
    Write-Host "`n    [FAIL] $msg" -ForegroundColor Red
    exit 1
}

# ============================================================
# STEP 1 - Install Python training dependencies
# ============================================================
Log-Step 1 "Installing Python training dependencies..."
Set-Location $ML_DIR

Log-Info "Installing packages from ml-service\requirements.txt..."
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Log-Fail "pip install -r requirements.txt failed" }
Log-OK "Python packages installed"

# Install CUDA-enabled PyTorch if CUDA is available and torch is not yet installed
Log-Info "Checking PyTorch / CUDA..."
$cudaAvailable = python -c "import torch; print(torch.cuda.is_available())" 2>$null
if ($cudaAvailable -eq "True") {
    Log-OK "CUDA is available - PyTorch already set up correctly"
}
elseif ($null -eq $cudaAvailable -or $cudaAvailable -eq "") {
    Log-Warn "torch not found or broken - installing CUDA 12.1 build..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -ne 0) { Log-Fail "CUDA PyTorch install failed" }
    Log-OK "CUDA PyTorch installed"
}
else {
    Log-Warn "PyTorch installed but CUDA not available - will train on CPU (slower)"
}

# ============================================================
# STEP 2 - Pre-train global model (train_local.py)
#          Skipped automatically if all 4 artefacts already exist
# ============================================================
Log-Step 2 "Pre-training global LSTM model..."

if ($isRebuild) {
    Log-Warn "Rebuild flag detected - performing deep clean before rebuilding..."
    
    # 1. Clean local pretrained folder
    if (Test-Path "$ML_DIR\pretrained") {
        Remove-Item -Path "$ML_DIR\pretrained\*" -Force -Recurse -ErrorAction SilentlyContinue
    }
    Log-OK "Pretrained folder cleaned."

    # 2. Clean persistent host volume folder on the K8s cluster before deleting the namespace
    Log-Warn "Cleaning persistent K8s volume for pretrained artefacts..."
    $existing = kubectl get namespace $NS --ignore-not-found 2>&1
    if ($existing -match $NS) {
        # Delete deployment first to release volume lock on PVC
        Log-Warn "Deleting ML deployment to release volume lock..."
        kubectl delete deployment scalable-ml-service -n $NS --ignore-not-found

        kubectl run vol-cleanup --image=alpine -n $NS --restart=Never --rm --attach -- sh -c "rm -rf /data/*" --overrides='{"spec":{"volumes":[{"name":"v","hostPath":{"path":"/data/scalable/pretrained"}}],"containers":[{"name":"c","image":"alpine","volumeMounts":[{"name":"v","mountPath":"/data"}]}]}}' 2>$null | Out-Null
        Log-OK "Persistent K8s volume cleaned."
    }

    # 3. Delete the Kubernetes namespace to remove all pods, containers, PVCs, services, etc.
    Log-Warn "Deleting Kubernetes namespace '$NS' (removes all pods, deployments, services, PVCs)..."
    kubectl delete namespace $NS --ignore-not-found
    Log-OK "Kubernetes namespace '$NS' deleted."

    # 4. Stop and remove any local Docker Compose containers
    Log-Warn "Stopping and removing local Docker Compose containers..."
    docker compose down -v --remove-orphans 2>$null | Out-Null
    Log-OK "Local Docker containers cleaned."

    # 5. Remove local project Docker images
    Log-Warn "Removing local project Docker images..."
    $images = @(
        "scalable-weather-app/ml-service:latest",
        "scalable-weather-app/backend:latest",
        "scalable-weather-app/frontend:latest",
        "scalable-weather-app/postgres:15-alpine"
    )
    foreach ($img in $images) {
        $exists = docker images -q $img 2>$null
        if ($exists) {
            docker rmi -f $img 2>$null | Out-Null
            Log-OK "Docker image $img removed"
        }
    }
}

$pretrained = @(
    "$ML_DIR\pretrained\global_weather_model.pth",
    "$ML_DIR\pretrained\model_config.json",
    "$ML_DIR\pretrained\scaler_params.json",
    "$ML_DIR\pretrained\training_cities.json"
)

$allExist = $true
foreach ($f in $pretrained) {
    if (-not (Test-Path $f)) { $allExist = $false; break }
}

if ($allExist) {
    Log-Warn "All pretrained artefacts already exist - skipping training."
    Log-Info "  Delete ml-service\pretrained\ and re-run to force a fresh train."
    foreach ($f in $pretrained) { Log-OK $f }
}
else {
    Log-Info "Starting train_local.py - this may take a long time on GPU..."
    Log-Info "  Training on: $(python -c "import torch; print('CUDA: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>$null)"
    Set-Location $ML_DIR
    python train_local.py
    if ($LASTEXITCODE -ne 0) { Log-Fail "train_local.py failed - check output above" }

    # Verify artefacts were produced
    foreach ($f in $pretrained) {
        if (Test-Path $f) { Log-OK $f }
        else { Log-Fail "train_local.py finished but missing: $f" }
    }
    Log-OK "Pre-training complete - all artefacts saved to ml-service\pretrained\"
}

# ============================================================
# STEP 3 - Build Docker images
# ============================================================
Log-Step 3 "Building Docker images..."
Set-Location $ROOT

Log-Info "Building scalable-weather-app/ml-service..."
docker build -t scalable-weather-app/ml-service:latest ./ml-service
if ($LASTEXITCODE -ne 0) { Log-Fail "scalable-weather-app/ml-service build failed" }
Log-OK "scalable-weather-app/ml-service:latest"

Log-Info "Building scalable-weather-app/backend..."
docker build -t scalable-weather-app/backend:latest ./backend
if ($LASTEXITCODE -ne 0) { Log-Fail "scalable-weather-app/backend build failed" }
Log-OK "scalable-weather-app/backend:latest"

Log-Info "Building scalable-weather-app/frontend..."
docker build -t scalable-weather-app/frontend:latest ./frontend
if ($LASTEXITCODE -ne 0) { Log-Fail "scalable-weather-app/frontend build failed" }
Log-OK "scalable-weather-app/frontend:latest"

Log-Info "Pulling and tagging postgres:15-alpine..."
docker pull postgres:15-alpine
if ($LASTEXITCODE -ne 0) { Log-Fail "docker pull postgres:15-alpine failed" }
docker tag postgres:15-alpine scalable-weather-app/postgres:15-alpine
if ($LASTEXITCODE -ne 0) { Log-Fail "docker tag postgres:15-alpine failed" }
Log-OK "scalable-weather-app/postgres:15-alpine ready"

Log-Info "Docker Desktop K8s shares the daemon - images immediately available to cluster"

# ============================================================
# STEP 4 - Kubernetes namespace
# ============================================================
Log-Step 4 "Kubernetes namespace..."

$existing = kubectl get namespace $NS --ignore-not-found 2>&1
if ($existing -match $NS) {
    Log-Warn "Namespace '$NS' already exists - skipping creation"
}
else {
    kubectl create namespace $NS
    Log-OK "Namespace '$NS' created"
}

# ============================================================
# STEP 5 - Deploy Postgres and wait until Ready
# ============================================================
Log-Step 5 "Deploying Postgres..."
Set-Location $ROOT
kubectl apply -f k8s/postgres.yaml

Log-Info "Waiting for Postgres deployment to be Ready (timeout 120s)..."
kubectl rollout status deployment/scalable-postgres -n $NS --timeout=120s
if ($LASTEXITCODE -ne 0) { Log-Fail "Postgres did not become Ready within 120s" }
Log-OK "Postgres is Ready"

# ============================================================
# STEP 6 - Deploy ML Service and wait until Ready
# ============================================================
Log-Step 6 "Deploying ML Service..."

Log-Warn "Deleting stale deployment to ensure clean volume re-seeding..."
kubectl delete deployment scalable-ml-service -n $NS --ignore-not-found

kubectl apply -f k8s/ml-service.yaml

Log-Info "Waiting for ML Service deployment to be Ready (timeout 180s)..."
kubectl rollout status deployment/scalable-ml-service -n $NS --timeout=180s
if ($LASTEXITCODE -ne 0) { Log-Fail "ML Service did not become Ready within 180s" }
Log-OK "ML Service is Ready"

# ============================================================
# STEP 7 - Deploy Backend and wait until Ready
# ============================================================
Log-Step 7 "Deploying Backend..."
kubectl apply -f k8s/backend.yaml

$postgresIp = kubectl get pods -l app=scalable-postgres -n $NS -o jsonpath='{.items[0].status.podIP}'
$mlIp = kubectl get pods -l app=scalable-ml-service -n $NS -o jsonpath='{.items[0].status.podIP}'
Log-Info "DNS Bypass: Injecting postgres pod IP ($postgresIp) and ML service pod IP ($mlIp) into backend..."
kubectl set env deployment/scalable-backend DB_HOST=$postgresIp ML_HOST=$mlIp -n $NS

Log-Info "Waiting for Backend deployment to be Ready (timeout 120s)..."
kubectl rollout status deployment/scalable-backend -n $NS --timeout=120s
if ($LASTEXITCODE -ne 0) { Log-Fail "Backend did not become Ready within 120s" }
Log-OK "Backend is Ready"

# ============================================================
# STEP 8 - Deploy Frontend and wait until Ready
# ============================================================
Log-Step 8 "Deploying Frontend..."
kubectl apply -f k8s/frontend.yaml

Log-Info "Waiting for Frontend deployment to be Ready (timeout 90s)..."
kubectl rollout status deployment/scalable-frontend -n $NS --timeout=90s
if ($LASTEXITCODE -ne 0) { Log-Fail "Frontend did not become Ready within 90s" }
Log-OK "Frontend is Ready"

# ============================================================
# SUMMARY
# ============================================================
Write-Host "`n`n============================================================" -ForegroundColor Cyan
Write-Host "  DEPLOYMENT COMPLETE  [$(Get-Date -f 'HH:mm:ss')]" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host "`n--- Pods ---" -ForegroundColor White
kubectl get pods -n $NS

Write-Host "`n--- Services ---" -ForegroundColor White
kubectl get svc -n $NS

Write-Host "`n--- PVCs ---" -ForegroundColor White
kubectl get pvc -n $NS

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  Frontend   : http://localhost:30000" -ForegroundColor Green
Write-Host "  Backend    : http://localhost:30080" -ForegroundColor Green
Write-Host "  ML Service : (cluster-internal - port-forward to test)" -ForegroundColor Yellow
Write-Host "    kubectl port-forward -n $NS svc/scalable-ml-service 8000:8000" -ForegroundColor Yellow
Write-Host "    then open: http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "============================================================`n" -ForegroundColor Cyan

Start-Process "http://localhost:30000"
```

---

#### 2. Linux & macOS Deployment (`deploy.sh`)

The `deploy.sh` script is designed for Bash-supported environments (such as Ubuntu, Debian, or macOS) and supports automated dependency resolution and Minikube configuration out-of-the-box.

##### Automated Dependency Resolution
When executed, `deploy.sh` automatically checks for and attempts to install (using `apt-get` if on Debian/Ubuntu):
- **Python 3** (and packages in `ml-service/requirements.txt`)
- **Java 21 OpenJDK** & **Maven** (required for building the backend microservice)
- **Node.js 22.x & NPM** (required for building the React frontend)
- **Docker Engine** (enabling container building)
- **kubectl** (for cluster interaction)
- **Minikube** (installed and started automatically if no active Kubernetes context is detected)

##### Minikube & Docker Daemon Integration
If running on Minikube, the script:
- Configures your shell context to use Minikube's internal Docker daemon (`eval $(minikube docker-env)`). This allows `docker build` to write directly into Minikube's image registry, making local images immediately available without loading steps.
- Automatically detects your Minikube IP and outputs correct access URLs (e.g., `http://<minikube-ip>:30000`).

##### GPU / CUDA / cuDNN Detection
- The script automatically checks if your machine has an NVIDIA GPU via `nvidia-smi`.
- If a GPU is present, it installs CUDA-enabled PyTorch (`cu121` build) into the python virtual environment. This packages the necessary CUDA/cuDNN runtimes automatically.

##### Standard Execution
```bash
chmod +x deploy.sh
./deploy.sh
```

##### Clean Rebuild Execution
```bash
./deploy.sh rebuild
# or
./deploy.sh --rebuild
```

##### Script Source (`deploy.sh`)
Save the following as `deploy.sh` in the project root:
```bash
#!/usr/bin/env bash

# ============================================================
#  scalable Weather App - Full Linux/macOS Deploy Script
#  Target: Kubernetes (Minikube / MicroK8s / Docker Desktop)
#
#  Usage:  ./deploy.sh [--rebuild]
# ============================================================

set -eo pipefail

NS="scalable-weather-app"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_DIR="$ROOT/ml-service"
TOTAL_STEPS=8

# Argument parsing
isRebuild=false
for arg in "$@"; do
    if [[ "$arg" == "--rebuild" || "$arg" == "-r" || "$arg" == "rebuild" ]]; then
        isRebuild=true
    fi
done

# Colors for output
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_step() {
    local step=$1
    local msg=$2
    echo -e "\n${CYAN}[$(date +'%H:%M:%S')] >>> STEP ${step}/${TOTAL_STEPS} - ${msg}${NC}"
}

log_ok() {
    echo -e "    ${GREEN}[OK]   $1${NC}"
}

log_warn() {
    echo -e "    ${YELLOW}[!!]   $1${NC}"
}

log_info() {
    echo -e "           $1"
}

log_fail() {
    echo -e "\n    ${RED}[FAIL] $1${NC}"
    exit 1
}

# ============================================================
# STEP 1 - Check and Install Dependencies
# ============================================================
log_step 1 "Verifying and installing dependencies..."

# Helper function to check command existence
has_cmd() {
    command -v "$1" >/dev/null 2>&1
}

# Helper function to install using apt-get if on Ubuntu/Debian
apt_install() {
    local pkg=$1
    echo "Installing $pkg via apt..."
    sudo apt-get update -qq
    sudo apt-get install -y "$pkg"
}

# 1. Check Python 3
if ! has_cmd python3; then
    if has_cmd apt-get; then
        apt_install "python3 python3-pip python3-venv"
    else
        log_fail "python3 is not installed. Please install Python 3.10+ and re-run."
    fi
fi
log_ok "Python 3 is available: \$(python3 --version | head -n1)"

# 2. Check Java 21
if ! has_cmd java || [[ "\$(java -version 2>&1 | head -n 1)" != *"21"* ]]; then
    if has_cmd apt-get; then
        echo "Installing OpenJDK 21 via apt..."
        sudo apt-get update -qq
        sudo apt-get install -y openjdk-21-jdk
    else
        log_warn "Java 21 not detected. Please ensure Java 21 is installed."
    fi
fi
log_ok "Java is available: \$(java -version 2>&1 | head -n1)"

# 3. Check Maven
if ! has_cmd mvn; then
    if has_cmd apt-get; then
        apt_install "maven"
    else
        log_fail "Maven is not installed. Please install Maven and re-run."
    fi
fi
log_ok "Maven is available: \$(mvn --version | head -n1)"

# 4. Check Node.js and NPM
if ! has_cmd npm; then
    if has_cmd apt-get; then
        echo "Installing Node.js 22.x via Nodesource..."
        curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
        sudo apt-get install -y nodejs
    else
        log_fail "npm is not installed. Please install Node.js 22+ and re-run."
    fi
fi
log_ok "NPM is available: npm v\$(npm -v)"

# 5. Check Docker
if ! has_cmd docker; then
    if has_cmd apt-get; then
        echo "Installing Docker CE..."
        curl -fsSL https://get.docker.com | sh
        sudo usermod -aG docker "\$USER"
        log_warn "Docker installed. You may need to log out and log back in to run docker without sudo."
    else
        log_fail "Docker is not installed. Please install Docker and re-run."
    fi
fi
log_ok "Docker is available: \$(docker --version)"

# 6. Check kubectl
if ! has_cmd kubectl; then
    if [[ "\$OSTYPE" == "linux-gnu"* ]]; then
        echo "Installing kubectl..."
        curl -LO "https://dl.k8s.io/release/\$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
        chmod +x kubectl
        sudo mv kubectl /usr/local/bin/
    else
        log_fail "kubectl is not installed. Please install kubectl and re-run."
    fi
fi
log_ok "kubectl is available: \$(kubectl version --client --short 2>/dev/null || kubectl version --client | head -n1)"

# 7. Check Kubernetes Cluster (Minikube fallback)
is_minikube=false
if ! kubectl cluster-info >/dev/null 2>&1; then
    log_warn "No active Kubernetes context detected. Searching for Minikube..."
    if ! has_cmd minikube; then
        if [[ "\$OSTYPE" == "linux-gnu"* ]]; then
            echo "Installing Minikube..."
            curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
            sudo install minikube-linux-amd64 /usr/local/bin/minikube
            rm minikube-linux-amd64
        else
            log_fail "Minikube is not installed. Please start your Kubernetes cluster or install minikube."
        fi
    fi
    log_ok "Minikube binary found."
    
    # Start minikube if not running
    if ! minikube status >/dev/null 2>&1; then
        echo "Starting Minikube cluster..."
        minikube start --driver=docker
    fi
    is_minikube=true
    log_ok "Minikube cluster is running."
else
    # Check if existing context is minikube
    if kubectl config current-context 2>/dev/null | grep -q "minikube"; then
        is_minikube=true
    fi
fi

# Configure docker environment to build inside Minikube daemon if minikube is used
if [ "\$is_minikube" = true ]; then
    log_info "Configuring shell to use Minikube's Docker daemon..."
    eval "\$(minikube docker-env)"
    log_ok "Docker daemon pointed to Minikube cluster context."
fi

# ============================================================
# STEP 2 - Setup Python dependencies
# ============================================================
log_step 2 "Setting up Python virtual environment & training packages..."
cd "\$ML_DIR"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    log_ok "Created python3 virtual environment 'venv'"
fi

source venv/bin/activate

log_info "Installing requirements.txt in virtual environment..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
log_ok "Base requirements installed."

# GPU / CUDA / cuDNN Detection
log_info "Checking NVIDIA GPU / CUDA support..."
gpu_info=\$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

if [ "\$gpu_info" = "True" ]; then
    log_ok "CUDA GPU is supported and configured in PyTorch."
else
    # Check if nvidia-smi is present (meaning hardware exists but drivers/torch cuDNN might not be mapped)
    if has_cmd nvidia-smi; then
        log_warn "NVIDIA GPU found but current PyTorch installation has no CUDA support. Installing CUDA 12.1 PyTorch..."
        pip install -q torch --index-url https://download.pytorch.org/whl/cu121
        log_ok "CUDA-enabled PyTorch installed."
    else
        log_info "No NVIDIA GPU detected. Training will run on CPU (slower)."
    fi
fi

# ============================================================
# STEP 3 - Clean / Rebuild Cleanup
# ============================================================
log_step 3 "Checking for Rebuild / Cleaning flag..."

if [ "\$isRebuild" = true ]; then
    log_warn "Rebuild flag detected - performing deep clean before rebuilding..."
    
    # 1. Clean local pretrained folder
    if [ -d "\$ML_DIR/pretrained" ]; then
        rm -rf "\$ML_DIR"/pretrained/*
    fi
    log_ok "Pretrained folder cleaned."

    # 2. Clean persistent host volume folder on the K8s cluster before deleting the namespace
    log_warn "Cleaning persistent K8s volume for pretrained artefacts..."
    if kubectl get namespace "\$NS" >/dev/null 2>&1; then
        # Delete deployment first to release volume lock on PVC
        log_warn "Deleting ML deployment to release volume lock..."
        kubectl delete deployment scalable-ml-service -n "\$NS" --ignore-not-found

        # Run temporary cleanup pod
        kubectl run vol-cleanup --image=alpine -n "\$NS" --restart=Never --rm --attach -- sh -c "rm -rf /data/*" --overrides='{"spec":{"volumes":[{"name":"v","hostPath":{"path":"/data/scalable/pretrained"}}],"containers":[{"name":"c","image":"alpine","volumeMounts":[{"name":"v","mountPath":"/data"}]}]}}' 2>/dev/null || true
        log_ok "Persistent K8s volume cleaned."
    fi

    # 3. Delete the Kubernetes namespace to remove all pods, containers, PVCs, services, etc.
    log_warn "Deleting Kubernetes namespace '\$NS' (removes all pods, deployments, services, PVCs)..."
    kubectl delete namespace "\$NS" --ignore-not-found
    log_ok "Kubernetes namespace '\$NS' deleted."

    # 4. Stop and remove any local Docker Compose containers
    log_warn "Stopping and removing local Docker Compose containers..."
    cd "\$ROOT"
    docker compose down -v --remove-orphans 2>/dev/null || true
    log_ok "Local Docker containers cleaned."

    # 5. Remove local project Docker images
    log_warn "Removing local project Docker images..."
    images=(
        "scalable-weather-app/ml-service:latest"
        "scalable-weather-app/backend:latest"
        "scalable-weather-app/frontend:latest"
        "scalable-weather-app/postgres:15-alpine"
        "scalable-ml-service:latest"
        "scalable-backend:latest"
        "scalable-frontend:latest"
    )
    for img in "\${images[@]}"; do
        if docker images -q "\$img" >/dev/null 2>&1; then
            docker rmi -f "\$img" 2>/dev/null || true
            log_ok "Docker image \$img removed"
        fi
    done
fi

# ============================================================
# STEP 4 - Pre-train global model (train_local.py)
# ============================================================
log_step 4 "Pre-training global LSTM model..."
cd "\$ML_DIR"

pretrained=(
    "\$ML_DIR/pretrained/global_weather_model.pth"
    "\$ML_DIR/pretrained/model_config.json"
    "\$ML_DIR/pretrained/scaler_params.json"
    "\$ML_DIR/pretrained/training_cities.json"
)

allExist=true
for f in "\${pretrained[@]}"; do
    if [ ! -f "\$f" ]; then
        allExist=false
        break
    fi
done

if [ "\$allExist" = true ]; then
    log_warn "All pretrained artefacts already exist - skipping training."
    log_info "  Delete ml-service/pretrained/ and re-run to force a fresh train."
    for f in "\${pretrained[@]}"; do log_ok "\$(basename "\$f") exists"; done
else
    log_info "Starting train_local.py - this may take a long time..."
    log_info "  Training device: \$(python3 -c "import torch; print('CUDA GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>/dev/null)"
    python3 train_local.py
    
    # Verify artefacts were produced
    for f in "\${pretrained[@]}"; do
        if [ -f "\$f" ]; then
            log_ok "\$(basename "\$f") created"
        else
            log_fail "train_local.py finished but missing: \$f"
        fi
    done
    log_ok "Pre-training complete - all artefacts saved to ml-service/pretrained/"
fi

# ============================================================
# STEP 5 - Build Docker images
# ============================================================
log_step 5 "Building Docker images..."
cd "\$ROOT"

log_info "Building scalable-weather-app/ml-service..."
docker build -t scalable-weather-app/ml-service:latest ./ml-service
log_ok "scalable-weather-app/ml-service:latest built"

log_info "Building scalable-weather-app/backend..."
docker build -t scalable-weather-app/backend:latest ./backend
log_ok "scalable-weather-app/backend:latest built"

log_info "Building scalable-weather-app/frontend..."
docker build -t scalable-weather-app/frontend:latest ./frontend
log_ok "scalable-weather-app/frontend:latest built"

log_info "Pulling and tagging postgres:15-alpine..."
docker pull postgres:15-alpine
docker tag postgres:15-alpine scalable-weather-app/postgres:15-alpine
log_ok "scalable-weather-app/postgres:15-alpine ready"

# ============================================================
# STEP 6 - Kubernetes Namespace
# ============================================================
log_step 6 "Creating Kubernetes namespace..."

if kubectl get namespace "\$NS" >/dev/null 2>&1; then
    log_warn "Namespace '\$NS' already exists - skipping creation"
else
    kubectl create namespace "\$NS"
    log_ok "Namespace '\$NS' created"
fi

# ============================================================
# STEP 7 - Deploy services to Kubernetes and wait Ready
# ============================================================
log_step 7 "Deploying Postgres and services..."
cd "\$ROOT"

# Deploy Postgres
kubectl apply -f k8s/postgres.yaml
log_info "Waiting for Postgres deployment to be Ready..."
kubectl rollout status deployment/scalable-postgres -n "\$NS" --timeout=120s
log_ok "Postgres is Ready"

# Deploy ML Service
kubectl delete deployment scalable-ml-service -n "\$NS" --ignore-not-found 2>/dev/null || true
kubectl apply -f k8s/ml-service.yaml
log_info "Waiting for ML Service deployment to be Ready..."
kubectl rollout status deployment/scalable-ml-service -n "\$NS" --timeout=180s
log_ok "ML Service is Ready"

# Deploy Backend
kubectl apply -f k8s/backend.yaml
postgresIp=\$(kubectl get pods -l app=scalable-postgres -n "\$NS" -o jsonpath='{.items[0].status.podIP}')
mlIp=\$(kubectl get pods -l app=scalable-ml-service -n "\$NS" -o jsonpath='{.items[0].status.podIP}')
log_info "DNS Bypass: Injecting postgres IP (\$postgresIp) and ML IP (\$mlIp)..."
kubectl set env deployment/scalable-backend DB_HOST="\$postgresIp" ML_HOST="\$mlIp" -n "\$NS"

log_info "Waiting for Backend deployment to be Ready..."
kubectl rollout status deployment/scalable-backend -n "\$NS" --timeout=120s
log_ok "Backend is Ready"

# Deploy Frontend
kubectl apply -f k8s/frontend.yaml
log_info "Waiting for Frontend deployment to be Ready..."
kubectl rollout status deployment/scalable-frontend -n "\$NS" --timeout=90s
log_ok "Frontend is Ready"

# ============================================================
# STEP 8 - Final summary & access links
# ============================================================
log_step 8 "Deployment summary"

# Determine IP address based on cluster
if [ "\$is_minikube" = true ]; then
    cluster_ip=\$(minikube ip)
else
    cluster_ip="localhost"
fi

echo -e "\n============================================================"
echo -e "  DEPLOYMENT COMPLETE  [\$(date +'%H:%M:%S')]"
echo -e "============================================================"

echo -e "\n--- Pods ---"
kubectl get pods -n "\$NS"

echo -e "\n--- Services ---"
kubectl get svc -n "\$NS"

echo -e "\n============================================================"
echo -e "  Frontend UI : http://\${cluster_ip}:30000"
echo -e "  Backend API : http://\${cluster_ip}:30080"
echo -e "  ML Service  : (cluster-internal - port-forward to test)"
echo -e "    kubectl port-forward -n \$NS svc/scalable-ml-service 8000:8000"
echo -e "    then open: http://localhost:8000/docs"
echo -e "============================================================\n"

# Open services in browser if on a system with open/xdg-open
if has_cmd xdg-open; then
    xdg-open "http://\${cluster_ip}:30000" &
elif has_cmd open; then
    open "http://\${cluster_ip}:30000" &
fi
```


### Kubernetes Infrastructure Notes
- Kubernetes manifests are located in the `k8s/` directory.
- Docker images use local image references.
- `imagePullPolicy: IfNotPresent` is used in Kubernetes manifests.
- PostgreSQL currently uses `emptyDir`, meaning database data is not persistent across pod restarts.
- Local Kubernetes clusters such as Docker Desktop Kubernetes, Minikube, or Kind are recommended for development/testing.

---

## 8. Frontend Development Notes (React + Vite)

This project uses a React + TypeScript template setup using Vite.

### ESLint Configuration

To enable type-aware lint rules in production, configure `eslint.config.js`:

```js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules:
      // tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules:
      // tseslint.configs.stylisticTypeChecked,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },
])
```

#### React-Specific Linting
To add rules for React and React DOM, install `eslint-plugin-react-x` and `eslint-plugin-react-dom`:

```js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      reactX.configs['recommended-typescript'],
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },
])
```

### React Compiler
The React Compiler is not enabled on this template because of its impact on dev & build performance. To add it, refer to the [React Compiler Installation Guide](https://react.dev/learn/react-compiler/installation).
