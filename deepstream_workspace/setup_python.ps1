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
