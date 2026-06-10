# ============================================================
#  scalable Weather App - Full Deploy Script
#  Target: Docker Desktop Kubernetes (Windows)
#
#  Usage:  .\deploy.ps1
#
#  What this script does (in order):
#    0. Prompt for OpenWeather API key, inject into config files
#    1. Install Python dependencies for training
#    2. Run train_local.py  (CUDA pre-training - skipped if artefacts already exist)
#    3. Build Docker images
#    4. Create K8s namespace
#    5. Deploy Postgres          + wait Ready
#    6. Deploy ML Service        + wait PVCs Bound + wait Ready
#    7. Deploy Backend           + wait Ready
#    8. Deploy Frontend          + wait Ready
#    9. Scrub API key from config files (always, even on failure)
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
    throw $msg
}

# ============================================================
# STEP 0 - Prompt for OpenWeather API Key and inject it
# ============================================================
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  scalable Weather App - Deployment" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$apiKeySecure = Read-Host "Enter your OpenWeather API Key" -AsSecureString
$apiKey = [System.Net.NetworkCredential]::new("", $apiKeySecure).Password

if ([string]::IsNullOrWhiteSpace($apiKey)) {
    Write-Host "[FAIL] No API key provided. Aborting." -ForegroundColor Red
    exit 1
}

Write-Host "  Injecting API key into config files..." -ForegroundColor DarkGray
python "$ROOT\manage_api_keys.py" inject $apiKey
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] API key injection failed. Aborting." -ForegroundColor Red
    exit 1
}
Write-Host "  [OK] API key injected." -ForegroundColor Green

# Wrap the entire deployment in try/finally so the key is always scrubbed
try {

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

} finally {
    # ============================================================
    # STEP 9 - Scrub API key (always runs, even on error)
    # ============================================================
    Write-Host "`n  Scrubbing API key from config files..." -ForegroundColor DarkGray
    python "$ROOT\manage_api_keys.py" scrub $apiKey
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] API key scrubbed. Config files restored to placeholders.`n" -ForegroundColor Green
    } else {
        Write-Host "  [!!] WARNING: API key scrub may have failed. Run manually:`n       python manage_api_keys.py scrub <your-key>`n" -ForegroundColor Yellow
    }
    # Clear the key from memory
    $apiKey = $null
    $apiKeySecure = $null
}
