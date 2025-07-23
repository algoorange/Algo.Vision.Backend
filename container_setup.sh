#!/bin/bash
# Complete Container Setup Script for AlgoVision DeepStream

set -e
echo "🚀 Setting up AlgoVision in DeepStream Container..."

# 1. Update system and install dependencies
echo "📦 Installing system dependencies..."
apt-get update -q
apt-get install -y -q \
    python3-pip \
    python3-dev \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0 \
    build-essential

# 2. Navigate to workspace and verify structure
cd /workspace
echo "📁 Current directory: $(pwd)"
echo "📁 Directory contents:"
ls -la

# 3. Set up Python environment
echo "🐍 Setting up Python environment..."
export PYTHONPATH=/workspace:$PYTHONPATH
echo "PYTHONPATH set to: $PYTHONPATH"

# 4. Upgrade pip
python3 -m pip install --upgrade pip

# 5. Install core dependencies
echo "🌐 Installing FastAPI and core dependencies..."
pip3 install \
    fastapi>=0.110.0 \
    uvicorn>=0.27.1 \
    python-multipart>=0.0.9 \
    websockets>=11.0.0

# 6. Install computer vision dependencies
echo "👁️ Installing computer vision packages..."
pip3 install \
    opencv-python>=4.11.0.86 \
    numpy>=1.24.0 \
    Pillow>=10.1.0

# 7. Install AI/ML dependencies
echo "🤖 Installing AI/ML packages..."
pip3 install \
    torch>=2.0.0 \
    torchvision>=0.15.0 \
    ultralytics>=8.0.0 \
    sentence-transformers>=2.2.2 \
    faiss-cpu>=1.7.4

# 8. Install other dependencies
echo "📚 Installing additional dependencies..."
pip3 install \
    pymongo>=4.0.0 \
    httpx>=0.28.1 \
    groq>=0.8.0 \
    python-dotenv>=1.0.0 \
    tqdm>=4.66.1

# 9. Verify Python can find the app module
echo "🔍 Verifying Python module structure..."
python3 -c "import sys; print('Python path:', sys.path)"
python3 -c "import os; print('Current dir contents:', os.listdir('.'))"

# Check if app directory exists
if [ ! -d "app" ]; then
    echo "❌ ERROR: 'app' directory not found in /workspace"
    echo "Directory contents:"
    ls -la
    exit 1
fi

echo "✅ App directory found"

# 10. Test import
echo "🧪 Testing app import..."
python3 -c "
import sys
sys.path.insert(0, '/workspace')
try:
    from app.main import app
    print('✅ Successfully imported app.main')
except ImportError as e:
    print(f'❌ Import error: {e}')
    import os
    print(f'Current working directory: {os.getcwd()}')
    print(f'Contents: {os.listdir()}')
    if os.path.exists('app'):
        print(f'App directory contents: {os.listdir(\"app\")}')
    exit(1)
"

# 11. Create necessary directories
echo "📁 Creating required directories..."
mkdir -p uploads frames outputs inputs

# 12. Start the application
echo "🚀 Starting AlgoVision FastAPI server..."
echo "🌐 Server will be available at: http://localhost:8000"
echo "📖 API Documentation: http://localhost:8000/docs"
echo ""

# Run the application with proper Python path
cd /workspace
PYTHONPATH=/workspace python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload 