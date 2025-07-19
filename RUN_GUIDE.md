# 🚀 AlgoVision DeepStream Setup & Run Guide

## Step-by-Step Instructions

### 1. 🐳 Start the DeepStream Container

From your Windows PowerShell in the AlgoVision directory:

```powershell
# Remove any existing container
docker rm -f deepstream-container

# Start the new container
.\run_deepstream.ps1
```

### 2. 📦 Install Dependencies & Run Project

Once inside the container, run the startup script:

```bash
# Make the script executable and run it
chmod +x /workspace/startup_container.sh
/workspace/startup_container.sh
```

**OR** run the commands manually:

```bash
# Quick manual setup
cd /workspace

# Install system dependencies
apt-get update && apt-get install -y python3-pip python3-gi python3-gi-cairo gir1.2-gstreamer-1.0

# Install Python dependencies
pip3 install fastapi uvicorn python-multipart websockets opencv-python numpy torch ultralytics deep-sort-realtime

# Set Python path
export PYTHONPATH=/workspace:$PYTHONPATH

# Start the application
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. 🌐 Access Your Application

- **API Base URL**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Home Endpoint**: http://localhost:8000/

### 4. 🧪 Test the API

Test with curl or visit the documentation:

```bash
# Test the home endpoint
curl http://localhost:8000/

# Test upload functionality (if you have test files)
curl -X POST "http://localhost:8000/upload/video" -F "file=@test_video.mp4"
```

### 5. 📁 Available Endpoints

Your AlgoVision API includes:

- **Upload**: `/upload/*` - File upload endpoints
- **Query**: `/query/*` - Video analysis queries  
- **Videos**: `/videos/*` - Video management
- **Detection**: `/detection/*` - Object detection
- **DeepStream**: `/deepstream/*` - GPU-accelerated processing

### 6. 🔧 Troubleshooting

**If you get import errors:**
```bash
# Reinstall missing packages
pip3 install [package-name]

# Check Python path
echo $PYTHONPATH
```

**If DeepStream isn't available:**
```bash
# Check DeepStream status
python3 -c "from app.services.deepstream_service import DEEPSTREAM_AVAILABLE; print(f'DeepStream: {DEEPSTREAM_AVAILABLE}')"
```

**If container exits:**
```bash
# Check Docker logs
docker logs deepstream-container
```

### 7. 📋 What the Startup Script Does

1. ✅ Updates system packages
2. ✅ Installs GStreamer and system dependencies  
3. ✅ Installs Python dependencies (FastAPI, OpenCV, PyTorch, etc.)
4. ✅ Sets up environment variables
5. ✅ Tests all imports
6. ✅ Starts the FastAPI application

### 8. 🎯 Expected Output

You should see:
```
✅ GStreamer: OK
✅ FastAPI/Uvicorn: OK
✅ Computer Vision: OK
✅ Object Tracking: OK
✅ DeepStream Service: Available = True
✅ AlgoVision App: OK

🚀 Starting AlgoVision FastAPI application...
📡 Access the API at: http://localhost:8000
```

### 9. 🛑 Stopping the Application

To stop the application:
- Press `Ctrl+C` in the container
- Type `exit` to leave the container
- The container will automatically be removed due to `--rm` flag

---

**Ready to run? Execute the commands above and your AlgoVision project with DeepStream GPU acceleration will be running! 🎉** 