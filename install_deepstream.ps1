# DeepStream Installation Script for Windows PowerShell
# This script sets up DeepStream SDK via Docker on Windows

Write-Host "=== DeepStream SDK Installation for Windows ===" -ForegroundColor Green

# Check if Docker Desktop is installed
Write-Host "Checking Docker installation..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version
    Write-Host "Docker found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "Docker Desktop is not installed or not running!" -ForegroundColor Red
    Write-Host "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
    Write-Host "Make sure to enable WSL2 backend during installation." -ForegroundColor Yellow
    exit 1
}

# Check if NVIDIA Container Toolkit is available (for Docker Desktop with WSL2)
Write-Host "Checking NVIDIA GPU support..." -ForegroundColor Yellow
try {
    $gpuInfo = nvidia-smi
    Write-Host "NVIDIA GPU detected successfully" -ForegroundColor Green
} catch {
    Write-Host "Warning: nvidia-smi not found. Make sure NVIDIA drivers are installed." -ForegroundColor Yellow
}

# Create workspace directories
Write-Host "Creating workspace directories..." -ForegroundColor Yellow
$workspaceDir = "deepstream_workspace"
if (!(Test-Path $workspaceDir)) {
    New-Item -ItemType Directory -Path $workspaceDir
    Write-Host "Created $workspaceDir directory" -ForegroundColor Green
}

# Create subdirectories
$dirs = @("models", "configs", "inputs", "outputs", "logs")
foreach ($dir in $dirs) {
    $fullPath = Join-Path $workspaceDir $dir
    if (!(Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath
        Write-Host "Created $fullPath directory" -ForegroundColor Green
    }
}

# Copy configuration files to workspace
Write-Host "Copying configuration files..." -ForegroundColor Yellow
if (Test-Path "configs\deepstream") {
    Copy-Item -Path "configs\deepstream\*" -Destination "$workspaceDir\configs\" -Recurse -Force
    Write-Host "Configuration files copied successfully" -ForegroundColor Green
} else {
    Write-Host "Warning: configs/deepstream directory not found" -ForegroundColor Yellow
}

# Pull DeepStream Docker image
Write-Host "Pulling DeepStream Docker image (this may take several minutes)..." -ForegroundColor Yellow
try {
    docker pull nvcr.io/nvidia/deepstream:7.1-triton-multiarch
    Write-Host "DeepStream image pulled successfully" -ForegroundColor Green
} catch {
    Write-Host "Failed to pull DeepStream image. Check your internet connection and Docker setup." -ForegroundColor Red
    exit 1
}

# Create PowerShell runner script
$runnerScript = @"
# DeepStream Container Runner Script for Windows
# Usage: .\run_deepstream.ps1

`$workspaceDir = "deepstream_workspace"
`$currentDir = Get-Location

Write-Host "Starting DeepStream container..." -ForegroundColor Green
Write-Host "Workspace: `$currentDir\`$workspaceDir" -ForegroundColor Yellow

# Run DeepStream container with GPU support and volume mounts
docker run --gpus all -it --rm \
    -v "`$currentDir\`$workspaceDir:/workspace" \
    -v "`$currentDir\uploads:/workspace/inputs" \
    -v "`$currentDir\frames:/workspace/outputs" \
    -p 8554:8554 \
    -p 9000:9000 \
    --name deepstream-container \
    nvcr.io/nvidia/deepstream:7.1-triton-multiarch

Write-Host "DeepStream container stopped" -ForegroundColor Yellow
"@

$runnerScript | Out-File -FilePath "run_deepstream.ps1" -Encoding UTF8
Write-Host "Created run_deepstream.ps1 script" -ForegroundColor Green

# Create Python development script
$pythonScript = @"
# Python Development Environment Setup for DeepStream
# Run this inside the DeepStream container

Write-Host "Setting up Python environment for DeepStream development..." -ForegroundColor Green

# Install Python dependencies
pip install --upgrade pip
pip install fastapi uvicorn opencv-python numpy pillow requests

# Install PyTorch with CUDA support (if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "Python environment setup complete!" -ForegroundColor Green
Write-Host "You can now run your AlgoVision application with DeepStream support." -ForegroundColor Yellow
"@

$pythonScript | Out-File -FilePath "$workspaceDir\setup_python.ps1" -Encoding UTF8

# Final instructions
Write-Host "`n=== Installation Complete! ===" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Run: .\run_deepstream.ps1" -ForegroundColor White
Write-Host "2. Inside the container, run: bash /workspace/setup_python.ps1" -ForegroundColor White
Write-Host "3. Test with: python -c 'import gi; gi.require_version(\"Gst\", \"1.0\"); print(\"GStreamer OK\")'" -ForegroundColor White
Write-Host "4. Start your FastAPI application" -ForegroundColor White
Write-Host "`nWorkspace created at: $workspaceDir" -ForegroundColor Cyan
Write-Host "Runner script: run_deepstream.ps1" -ForegroundColor Cyan
Write-Host "`nFor troubleshooting, see DEEPSTREAM_SETUP.md" -ForegroundColor Yellow 