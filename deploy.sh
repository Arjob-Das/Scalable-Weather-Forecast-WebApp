#!/usr/bin/env bash

# ============================================================
#  scalable Weather App - Full Linux/macOS Deploy Script
#  Target: Kubernetes (Minikube / MicroK8s / Docker Desktop)
#
#  Usage:  ./deploy.sh [--rebuild]
#
#  Flow:
#    0. Prompt for OpenWeather API key, inject into config files
#    1-8. Build & Deploy as usual
#    9. Scrub API key from config files (always, even on failure)
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
# STEP 0 - Prompt for OpenWeather API Key and inject it
# ============================================================
echo -e "\n${CYAN}============================================================${NC}"
echo -e "${CYAN}  scalable Weather App - Deployment${NC}"
echo -e "${CYAN}============================================================${NC}\n"

if [ -t 0 ]; then
    # Running interactively — use silent read
    read -s -r -p "Enter your OpenWeather API Key: " OPENWEATHER_API_KEY
    echo
else
    # Running non-interactively (piped) — read normally
    echo -n "Enter your OpenWeather API Key: "
    read -r OPENWEATHER_API_KEY
fi

if [ -z "$OPENWEATHER_API_KEY" ]; then
    echo -e "${RED}[FAIL] No API key provided. Aborting.${NC}"
    exit 1
fi

# Scrub function — always called on exit via trap
scrub_api_key() {
    echo -e "\n  Scrubbing API key from config files..."
    if python3 "$ROOT/manage_api_keys.py" scrub "$OPENWEATHER_API_KEY" 2>/dev/null; then
        echo -e "  ${GREEN}[OK] API key scrubbed. Config files restored to placeholders.${NC}\n"
    else
        echo -e "  ${YELLOW}[!!] WARNING: API key scrub may have failed."
        echo -e "       Run manually: python3 manage_api_keys.py scrub <your-key>${NC}\n"
    fi
    unset OPENWEATHER_API_KEY
}

# Register the scrub to run on any exit (success or error)
trap scrub_api_key EXIT

echo "  Injecting API key into config files..."
python3 "$ROOT/manage_api_keys.py" inject "$OPENWEATHER_API_KEY"
echo -e "  ${GREEN}[OK] API key injected.${NC}\n"

# ============================================================
# STEP 1 - Check and Install Dependencies
# ============================================================
log_step 1 "Verifying and installing dependencies..."

# Helper function to check command existence
has_cmd() {
    command -v "$1" > /dev/null 2>&1
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
log_ok "Python 3 is available: $(python3 --version | head -n1)"

# 2. Check Java 21
if ! has_cmd java || [[ "$(java -version 2>&1 | head -n 1)" != *"21"* ]]; then
    if has_cmd apt-get; then
        echo "Installing OpenJDK 21 via apt..."
        sudo apt-get update -qq
        sudo apt-get install -y openjdk-21-jdk
    else
        log_warn "Java 21 not detected. Please ensure Java 21 is installed."
    fi
fi
log_ok "Java is available: $(java -version 2>&1 | head -n1)"

# 3. Check Maven
if ! has_cmd mvn; then
    if has_cmd apt-get; then
        apt_install "maven"
    else
        log_fail "Maven is not installed. Please install Maven and re-run."
    fi
fi
log_ok "Maven is available: $(mvn --version | head -n1)"

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
log_ok "NPM is available: npm v$(npm -v)"

# 5. Check Docker
if ! has_cmd docker; then
    if has_cmd apt-get; then
        echo "Installing Docker CE..."
        curl -fsSL https://get.docker.com | sh
        sudo usermod -aG docker "$USER"
        log_warn "Docker installed. You may need to log out and log back in to run docker without sudo."
    else
        log_fail "Docker is not installed. Please install Docker and re-run."
    fi
fi
log_ok "Docker is available: $(docker --version)"

# 6. Check kubectl
if ! has_cmd kubectl; then
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Installing kubectl..."
        curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
        chmod +x kubectl
        sudo mv kubectl /usr/local/bin/
    else
        log_fail "kubectl is not installed. Please install kubectl and re-run."
    fi
fi
log_ok "kubectl is available: $(kubectl version --client --short 2>/dev/null || kubectl version --client | head -n1)"

# 7. Check Kubernetes Cluster (Minikube fallback)
is_minikube=false
if ! kubectl cluster-info >/dev/null 2>&1; then
    log_warn "No active Kubernetes context detected. Searching for Minikube..."
    if ! has_cmd minikube; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
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
if [ "$is_minikube" = true ]; then
    log_info "Configuring shell to use Minikube's Docker daemon..."
    eval "$(minikube docker-env)"
    log_ok "Docker daemon pointed to Minikube cluster context."
fi

# ============================================================
# STEP 2 - Setup Python dependencies
# ============================================================
log_step 2 "Setting up Python virtual environment & training packages..."
cd "$ML_DIR"

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
gpu_info=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

if [ "$gpu_info" = "True" ]; then
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

if [ "$isRebuild" = true ]; then
    log_warn "Rebuild flag detected - performing deep clean before rebuilding..."
    
    # 1. Clean local pretrained folder
    if [ -d "$ML_DIR/pretrained" ]; then
        rm -rf "$ML_DIR"/pretrained/*
    fi
    log_ok "Pretrained folder cleaned."

    # 2. Clean persistent host volume folder on the K8s cluster before deleting the namespace
    log_warn "Cleaning persistent K8s volume for pretrained artefacts..."
    if kubectl get namespace "$NS" >/dev/null 2>&1; then
        # Delete deployment first to release volume lock on PVC
        log_warn "Deleting ML deployment to release volume lock..."
        kubectl delete deployment scalable-ml-service -n "$NS" --ignore-not-found

        # Run temporary cleanup pod
        kubectl run vol-cleanup --image=alpine -n "$NS" --restart=Never --rm --attach -- sh -c "rm -rf /data/*" --overrides='{"spec":{"volumes":[{"name":"v","hostPath":{"path":"/data/scalable/pretrained"}}],"containers":[{"name":"c","image":"alpine","volumeMounts":[{"name":"v","mountPath":"/data"}]}]}}' 2>/dev/null || true
        log_ok "Persistent K8s volume cleaned."
    fi

    # 3. Delete the Kubernetes namespace to remove all pods, containers, PVCs, services, etc.
    log_warn "Deleting Kubernetes namespace '$NS' (removes all pods, deployments, services, PVCs)..."
    kubectl delete namespace "$NS" --ignore-not-found
    log_ok "Kubernetes namespace '$NS' deleted."

    # 4. Stop and remove any local Docker Compose containers
    log_warn "Stopping and removing local Docker Compose containers..."
    cd "$ROOT"
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
    for img in "${images[@]}"; do
        if docker images -q "$img" >/dev/null 2>&1; then
            docker rmi -f "$img" 2>/dev/null || true
            log_ok "Docker image $img removed"
        fi
    done
fi

# ============================================================
# STEP 4 - Pre-train global model (train_local.py)
# ============================================================
log_step 4 "Pre-training global LSTM model..."
cd "$ML_DIR"

pretrained=(
    "$ML_DIR/pretrained/global_weather_model.pth"
    "$ML_DIR/pretrained/model_config.json"
    "$ML_DIR/pretrained/scaler_params.json"
    "$ML_DIR/pretrained/training_cities.json"
)

allExist=true
for f in "${pretrained[@]}"; do
    if [ ! -f "$f" ]; then
        allExist=false
        break
    fi
done

if [ "$allExist" = true ]; then
    log_warn "All pretrained artefacts already exist - skipping training."
    log_info "  Delete ml-service/pretrained/ and re-run to force a fresh train."
    for f in "${pretrained[@]}"; do log_ok "$(basename "$f") exists"; done
else
    log_info "Starting train_local.py - this may take a long time..."
    log_info "  Training device: $(python3 -c "import torch; print('CUDA GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>/dev/null)"
    python3 train_local.py
    
    # Verify artefacts were produced
    for f in "${pretrained[@]}"; do
        if [ -f "$f" ]; then
            log_ok "$(basename "$f") created"
        else
            log_fail "train_local.py finished but missing: $f"
        fi
    done
    log_ok "Pre-training complete - all artefacts saved to ml-service/pretrained/"
fi

# ============================================================
# STEP 5 - Build Docker images
# ============================================================
log_step 5 "Building Docker images..."
cd "$ROOT"

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

if kubectl get namespace "$NS" >/dev/null 2>&1; then
    log_warn "Namespace '$NS' already exists - skipping creation"
else
    kubectl create namespace "$NS"
    log_ok "Namespace '$NS' created"
fi

# ============================================================
# STEP 7 - Deploy services to Kubernetes and wait Ready
# ============================================================
log_step 7 "Deploying Postgres and services..."
cd "$ROOT"

# Deploy Postgres
kubectl apply -f k8s/postgres.yaml
log_info "Waiting for Postgres deployment to be Ready..."
kubectl rollout status deployment/scalable-postgres -n "$NS" --timeout=120s
log_ok "Postgres is Ready"

# Deploy ML Service
kubectl delete deployment scalable-ml-service -n "$NS" --ignore-not-found 2>/dev/null || true
kubectl apply -f k8s/ml-service.yaml
log_info "Waiting for ML Service deployment to be Ready..."
kubectl rollout status deployment/scalable-ml-service -n "$NS" --timeout=180s
log_ok "ML Service is Ready"

# Deploy Backend
kubectl apply -f k8s/backend.yaml
postgresIp=$(kubectl get pods -l app=scalable-postgres -n "$NS" -o jsonpath='{.items[0].status.podIP}')
mlIp=$(kubectl get pods -l app=scalable-ml-service -n "$NS" -o jsonpath='{.items[0].status.podIP}')
log_info "DNS Bypass: Injecting postgres IP ($postgresIp) and ML IP ($mlIp)..."
kubectl set env deployment/scalable-backend DB_HOST="$postgresIp" ML_HOST="$mlIp" -n "$NS"

log_info "Waiting for Backend deployment to be Ready..."
kubectl rollout status deployment/scalable-backend -n "$NS" --timeout=120s
log_ok "Backend is Ready"

# Deploy Frontend
kubectl apply -f k8s/frontend.yaml
log_info "Waiting for Frontend deployment to be Ready..."
kubectl rollout status deployment/scalable-frontend -n "$NS" --timeout=90s
log_ok "Frontend is Ready"

# ============================================================
# STEP 8 - Final summary & access links
# ============================================================
log_step 8 "Deployment summary"

# Determine IP address based on cluster
if [ "$is_minikube" = true ]; then
    cluster_ip=$(minikube ip)
else
    cluster_ip="localhost"
fi

echo -e "\n============================================================"
echo -e "  DEPLOYMENT COMPLETE  [$(date +'%H:%M:%S')]"
echo -e "============================================================"

echo -e "\n--- Pods ---"
kubectl get pods -n "$NS"

echo -e "\n--- Services ---"
kubectl get svc -n "$NS"

echo -e "\n============================================================"
echo -e "  Frontend UI : http://${cluster_ip}:30000"
echo -e "  Backend API : http://${cluster_ip}:30080"
echo -e "  ML Service  : (cluster-internal - port-forward to test)"
echo -e "    kubectl port-forward -n $NS svc/scalable-ml-service 8000:8000"
echo -e "    then open: http://localhost:8000/docs"
echo -e "============================================================\n"

# Open services in browser if on a system with open/xdg-open
if has_cmd xdg-open; then
    xdg-open "http://${cluster_ip}:30000" &
elif has_cmd open; then
    open "http://${cluster_ip}:30000" &
fi

# The trap registered in STEP 0 will handle scrubbing automatically on exit.
