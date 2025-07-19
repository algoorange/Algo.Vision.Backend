#!/bin/bash
# AlgoVision DeepStream Container Startup Script
# Run this script inside the DeepStream container

set -e  # Exit on any error

echo "=================================="
echo "AlgoVision DeepStream Setup Script"
echo "=================================="

# Update system packages
echo "📦 Updating system packages..."
apt-get update -q

# Install system dependencies
echo "🔧 Installing system dependencies..."
apt-get install -y -q \
    python3-pip \
    python3-dev \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0 \
    build-essential \
    git \
    curl \
    wget \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    pkg-config

# Navigate to workspace
cd /workspace

# Upgrade pip
echo "⬆️ Upgrading pip..."
python3 -m pip install --upgrade pip

# Install core web framework dependencies
echo "🌐 Installing FastAPI and Uvicorn..."
pip3 install fastapi>=0.110.0 uvicorn>=0.27.1 python-multipart>=0.0.9 websockets>=11.0.0

# Install computer vision dependencies
echo "👁️ Installing computer vision packages..."
pip3 install opencv-python>=4.11.0.86 numpy>=1.24.0 pillow>=10.1.0

# Install AI/ML dependencies
echo "🤖 Installing AI/ML packages..."
pip3 install torch>=2.0.0 torchvision>=0.15.0 torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install ultralytics>=8.0.0 transformers==4.44.0 sentence-transformers>=2.2.2 huggingface_hub==0.24.0

# Install tracking dependencies
echo "🎯 Installing object tracking packages..."
pip3 install deep-sort-realtime>=1.3.2 filterpy scikit-learn scipy matplotlib lap

# Install other required packages
echo "📚 Installing additional packages..."
pip3 install python-dotenv>=1.0.0 groq>=0.8.0 pydantic>=2.0.0
pip3 install tqdm>=4.66.1

# Install vector search and NLP dependencies
echo "🔍 Installing vector search and NLP packages..."
pip3 install faiss-cpu>=1.7.4 spacy>=3.7.2
pip3 install segmentation-models-pytorch>=0.3.0 scikit-image>=0.21.0

# Install HTTP and database dependencies
echo "🌐 Installing HTTP and database packages..."
pip3 install httpx>=0.28.1 pymongo>=4.0.0

# Install DeepStream Python dependencies
echo "🎬 Installing DeepStream Python packages..."
pip3 install pygobject>=3.42.0
# Note: pyds will be installed when DeepStream SDK is available

# Set up environment variables
echo "🔧 Setting up environment..."
export PYTHONPATH=/workspace:$PYTHONPATH
echo 'export PYTHONPATH=/workspace:$PYTHONPATH' >> ~/.bashrc

# Test imports
echo "🧪 Testing imports..."
python3 -c "
import sys
sys.path.insert(0, '/workspace')

print('Testing core dependencies...')

# Test system packages
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    Gst.init(None)
    print('✅ GStreamer: OK')
except Exception as e:
    print(f'❌ GStreamer: {e}')

# Test web framework
try:
    import fastapi, uvicorn
    print('✅ FastAPI/Uvicorn: OK')
except Exception as e:
    print(f'❌ FastAPI/Uvicorn: {e}')

# Test computer vision
try:
    import cv2, numpy as np, torch
    print('✅ Computer Vision: OK')
except Exception as e:
    print(f'❌ Computer Vision: {e}')

# Test tracking
try:
    from deep_sort_realtime import DeepSort
    print('✅ Object Tracking: OK')
except Exception as e:
    print(f'❌ Object Tracking: {e}')

# Test your application
try:
    from app.services.deepstream_service import DEEPSTREAM_AVAILABLE
    print(f'✅ DeepStream Service: Available = {DEEPSTREAM_AVAILABLE}')
except Exception as e:
    print(f'❌ DeepStream Service: {e}')

try:
    from app.main import app
    print('✅ AlgoVision App: OK')
except Exception as e:
    print(f'❌ AlgoVision App: {e}')

print('Import tests complete!')
"

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Starting AlgoVision FastAPI application..."
echo "📡 Access the API at: http://localhost:8000"
echo "📖 API documentation at: http://localhost:8000/docs"
echo ""

# Start the FastAPI application
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload 