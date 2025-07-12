# NVIDIA DeepStream Setup Guide

This guide explains how to set up NVIDIA DeepStream for high-performance object detection in your AlgoVision project.

## 🚀 **What is DeepStream?**

NVIDIA DeepStream is a streaming analytics toolkit that provides:
- **10x faster inference** compared to CPU-only solutions
- **Hardware-accelerated video processing** using NVIDIA GPUs
- **Multi-stream processing** capabilities
- **Optimized memory management** for real-time applications

## 📋 **Prerequisites**

### Hardware Requirements
- **NVIDIA GPU** with compute capability 6.0+ (GTX 1060 or better)
- **8GB+ GPU memory** recommended
- **Ubuntu 20.04/22.04** or **CentOS 7/8** (Linux only)

### Software Requirements
- **NVIDIA Driver** 470.57.02 or later
- **CUDA Toolkit** 11.4 or later
- **TensorRT** 8.2 or later
- **Docker** (optional but recommended)

## 🛠️ **Installation Steps**

### Step 1: Install NVIDIA DeepStream SDK

#### Option A: Using Debian Package (Recommended)
```bash
# Download DeepStream 6.3 SDK
wget https://developer.download.nvidia.com/assets/deepstream/secure/6.3/deepstream-6.3_6.3.0-1_amd64.deb

# Install DeepStream
sudo apt-get install ./deepstream-6.3_6.3.0-1_amd64.deb

# Verify installation
deepstream-app --version
```

#### Option B: Using Docker
```bash
# Pull DeepStream Docker image
docker pull nvcr.io/nvidia/deepstream:6.3-devel

# Run container with GPU support
docker run --gpus all -it --rm -v $(pwd):/workspace nvcr.io/nvidia/deepstream:6.3-devel
```

### Step 2: Install Python Dependencies

```bash
# Install GStreamer Python bindings
sudo apt-get update
sudo apt-get install python3-gi python3-gi-cairo gir1.2-gstreamer-1.0

# Install additional dependencies
sudo apt-get install gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
                     gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
                     gstreamer1.0-libav gstreamer1.0-tools

# Install Python packages
pip install pygobject pycairo
```

### Step 3: Download Pre-trained Models

```bash
# Create models directory
mkdir -p models/Primary_Detector

# Download ResNet-10 model (lightweight)
wget -O models/Primary_Detector/resnet10.caffemodel \
  https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/pruned_quantized_v2.3.2/files/resnet10.caffemodel

# Download config files
wget -O models/Primary_Detector/config_infer_primary.txt \
  https://raw.githubusercontent.com/NVIDIA-AI-IOT/deepstream_python_apps/master/apps/deepstream-test1/dstest1_pgie_config.txt

# Download labels
wget -O models/Primary_Detector/labels.txt \
  https://raw.githubusercontent.com/NVIDIA-AI-IOT/deepstream_python_apps/master/apps/deepstream-test1/labels.txt
```

### Step 4: Configure Your Project

Update your Python environment:
```bash
# Add DeepStream to Python path
export PYTHONPATH="${PYTHONPATH}:/opt/nvidia/deepstream/deepstream/lib"

# Set GST plugin path
export GST_PLUGIN_PATH="/opt/nvidia/deepstream/deepstream/lib/gst-plugins/"

# Add to your ~/.bashrc for permanent setup
echo 'export PYTHONPATH="${PYTHONPATH}:/opt/nvidia/deepstream/deepstream/lib"' >> ~/.bashrc
echo 'export GST_PLUGIN_PATH="/opt/nvidia/deepstream/deepstream/lib/gst-plugins/"' >> ~/.bashrc
```

## 🎯 **Usage**

### Basic Object Detection
```python
from app.services.object_detector import detect_objects_with_fast_rcnn

# Process a frame
detections, annotated_frame = detect_objects_with_fast_rcnn(frame)

# The function will automatically use DeepStream if available,
# otherwise fall back to PyTorch Faster R-CNN
```

### Performance Monitoring
```python
# Check if DeepStream is being used
from app.services.object_detector import DEEPSTREAM_AVAILABLE, deepstream_detector

if DEEPSTREAM_AVAILABLE and deepstream_detector.initialized:
    print("✅ Using NVIDIA DeepStream for object detection")
else:
    print("⚠️  Using PyTorch Faster R-CNN fallback")
```

## 🔧 **Configuration**

### Model Selection
You can use different pre-trained models:

1. **PeopleNet** - Optimized for person detection
2. **TrafficCamNet** - Optimized for vehicles
3. **DashCamNet** - Optimized for automotive scenes
4. **FaceDetectIR** - Optimized for face detection

### Performance Tuning
```python
# Adjust batch size for better GPU utilization
streammux.set_property("batch-size", 4)  # Process 4 frames at once

# Adjust inference interval
primary_detector.set_property("interval", 5)  # Process every 5th frame

# Use INT8 precision for faster inference
primary_detector.set_property("model-engine-file", "model_int8.engine")
```

## 📊 **Expected Performance Improvements**

| Method | FPS (1080p) | GPU Memory | Accuracy |
|--------|-------------|------------|----------|
| PyTorch Faster R-CNN | 5-10 | 4-6GB | High |
| DeepStream + ResNet-10 | 60-120 | 2-3GB | High |
| DeepStream + YOLOv5 | 80-150 | 3-4GB | Very High |

## 🐛 **Troubleshooting**

### Common Issues

1. **"Failed to create DeepStream elements"**
   ```bash
   # Check if DeepStream plugins are installed
   gst-inspect-1.0 nvstreammux
   ```

2. **"No module named 'gi'"**
   ```bash
   # Install GObject introspection
   sudo apt-get install python3-gi
   ```

3. **"CUDA out of memory"**
   ```python
   # Reduce batch size
   streammux.set_property("batch-size", 1)
   ```

### Verification Commands
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Check DeepStream installation
deepstream-app --version

# Test GStreamer plugins
gst-inspect-1.0 nvstreammux
```

## 🎯 **Next Steps**

1. **Custom Models**: Train your own models using NVIDIA TAO Toolkit
2. **Multi-stream**: Process multiple video streams simultaneously
3. **Edge Deployment**: Deploy on NVIDIA Jetson devices
4. **Cloud Integration**: Use with NVIDIA Triton Inference Server

## 📚 **Resources**

- [NVIDIA DeepStream Documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/)
- [DeepStream Python Apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)
- [TAO Toolkit](https://developer.nvidia.com/tao-toolkit)
- [TensorRT Optimization](https://developer.nvidia.com/tensorrt)

---

**Note**: DeepStream is currently Linux-only. Windows users will automatically fall back to PyTorch Faster R-CNN implementation. 