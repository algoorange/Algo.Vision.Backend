# Faster R-CNN Implementation for Vision Recognition System

## 🎯 Overview

This implementation adds **Faster R-CNN** object detection capabilities to the existing Vision Recognition system, complementing the existing YOLOv8 and RTDETR models.

## 🚀 Features

### Multi-Model Detection System
- **YOLOv8**: Ultra-fast real-time detection (10-20ms)
- **RTDETR**: Balanced speed and accuracy (15-30ms)
- **Faster R-CNN**: Highest accuracy two-stage detector (50-100ms)

### Flexible Configuration
- Enable/disable individual models
- Adjust confidence thresholds per model
- Prioritize models for better results
- Real-time configuration updates

### Enhanced Visualization
- Color-coded detection results by model:
  - 🔵 **Blue**: Faster R-CNN detections
  - 🟡 **Yellow**: RTDETR detections  
  - 🟢 **Green**: YOLOv8 detections
- Source model displayed in labels

## 📁 Files Modified/Added

### Core Detection System
- `app/services/object_detector.py` - Enhanced with Faster R-CNN support
- `app/routers/detection.py` - New API endpoints for model management
- `app/main.py` - Added detection router

### Testing & Documentation
- `test_faster_rcnn.py` - Comprehensive test suite
- `FASTER_RCNN_README.md` - This documentation

## 🛠️ Installation & Setup

### 1. Dependencies
The required dependencies are already in `requirements.txt`:
```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
```

### 2. Model Download
Faster R-CNN model will be automatically downloaded on first use:
- Model: `fasterrcnn_resnet50_fpn`
- Pre-trained on COCO dataset
- 80 object classes supported

### 3. Test Installation
Run the test script to verify everything works:
```bash
python test_faster_rcnn.py
```

## 🔧 API Endpoints

### Detection Configuration

#### GET `/detection/config`
Get current detection model configuration
```json
{
  "success": true,
  "current_config": {
    "use_yolo": true,
    "use_rtdetr": true,
    "use_faster_rcnn": true,
    "yolo_threshold": 0.7,
    "rtdetr_threshold": 0.6,
    "faster_rcnn_threshold": 0.7,
    "prioritize_faster_rcnn": true
  },
  "available_models": ["YOLOv8", "RTDETR", "Faster R-CNN"],
  "device": "cuda"
}
```

#### POST `/detection/config`
Update detection configuration
```json
{
  "use_faster_rcnn": true,
  "faster_rcnn_threshold": 0.8,
  "prioritize_faster_rcnn": true
}
```

#### GET `/detection/models`
Get detailed model information
```json
{
  "success": true,
  "models": ["YOLOv8", "RTDETR", "Faster R-CNN"],
  "model_status": {
    "YOLOv8": {"enabled": true, "threshold": 0.7},
    "RTDETR": {"enabled": true, "threshold": 0.6},
    "Faster R-CNN": {"enabled": true, "threshold": 0.7}
  },
  "device": "cuda"
}
```

#### GET `/detection/performance`
Get performance information and recommendations
```json
{
  "success": true,
  "device_info": {
    "device": "cuda",
    "cuda_available": true,
    "cuda_device_name": "GeForce RTX 3080"
  },
  "model_info": {
    "YOLOv8": {
      "description": "Ultra-fast object detection",
      "typical_inference_time": "10-20ms",
      "accuracy": "High"
    },
    "Faster R-CNN": {
      "description": "Two-stage detector, highest accuracy",
      "typical_inference_time": "50-100ms", 
      "accuracy": "Highest"
    }
  },
  "recommendations": {
    "real_time": ["YOLOv8"],
    "highest_accuracy": ["Faster R-CNN"],
    "multiple_models": ["Faster R-CNN", "RTDETR"]
  }
}
```

## 💻 Usage Examples

### Using the API

#### 1. Enable Only Faster R-CNN
```bash
curl -X POST "http://localhost:8000/detection/config" \
     -H "Content-Type: application/json" \
     -d '{
       "use_yolo": false,
       "use_rtdetr": false,
       "use_faster_rcnn": true
     }'
```

#### 2. Use All Models with Faster R-CNN Priority
```bash
curl -X POST "http://localhost:8000/detection/config" \
     -H "Content-Type: application/json" \
     -d '{
       "use_yolo": true,
       "use_rtdetr": true,
       "use_faster_rcnn": true,
       "prioritize_faster_rcnn": true
     }'
```

#### 3. Adjust Confidence Thresholds
```bash
curl -X POST "http://localhost:8000/detection/config" \
     -H "Content-Type: application/json" \
     -d '{
       "faster_rcnn_threshold": 0.8,
       "yolo_threshold": 0.6
     }'
```

### Using in Python Code

```python
from app.services import object_detector
import cv2

# Load an image
image = cv2.imread("test_image.jpg")

# Configure to use only Faster R-CNN
object_detector.set_detection_config(
    use_yolo=False,
    use_rtdetr=False,
    use_faster_rcnn=True,
    faster_rcnn_threshold=0.7
)

# Run detection
detections, annotated_image = object_detector.detect_objects(image)

# Print results
for detection in detections:
    print(f"Found {detection['label']} with {detection['confidence']:.3f} confidence")
    print(f"Detection source: {detection['source']}")
    print(f"Bounding box: {detection['bbox']}")

# Save annotated image
cv2.imwrite("result.jpg", annotated_image)
```

## 🎮 Model Comparison

| Model | Speed | Accuracy | Best Use Case | Memory Usage |
|-------|-------|----------|---------------|--------------|
| **YOLOv8** | ⚡ Fastest | 🟢 Good | Real-time video | 💚 Low |
| **RTDETR** | ⚡ Fast | 🟡 Very Good | Balanced performance | 🟡 Medium |
| **Faster R-CNN** | 🐌 Slower | 🔴 Excellent | High-accuracy analysis | 🔴 High |

### Recommended Configurations

#### Real-time Video Processing
```json
{
  "use_yolo": true,
  "use_rtdetr": false,
  "use_faster_rcnn": false,
  "yolo_threshold": 0.6
}
```

#### Balanced Performance
```json
{
  "use_yolo": false,
  "use_rtdetr": true,
  "use_faster_rcnn": false,
  "rtdetr_threshold": 0.6
}
```

#### Maximum Accuracy
```json
{
  "use_yolo": false,
  "use_rtdetr": false,
  "use_faster_rcnn": true,
  "faster_rcnn_threshold": 0.8
}
```

#### Multi-Model Ensemble
```json
{
  "use_yolo": true,
  "use_rtdetr": true,
  "use_faster_rcnn": true,
  "prioritize_faster_rcnn": true,
  "faster_rcnn_threshold": 0.7,
  "rtdetr_threshold": 0.6,
  "yolo_threshold": 0.7
}
```

## 🔍 Detection Results

The system now returns enhanced detection information:

```json
{
  "label": "car",
  "confidence": 0.95,
  "bbox": [100, 150, 200, 100],
  "source": "Faster R-CNN"
}
```

### COCO Classes Supported
Faster R-CNN supports 80 COCO object classes including:
- **Vehicles**: car, truck, bus, motorcycle, bicycle
- **People**: person
- **Animals**: cat, dog, horse, cow, sheep
- **Objects**: chair, table, laptop, phone, bottle
- And many more...

## 🚀 Getting Started

1. **Start the server:**
   ```bash
   uvicorn app.main:app --reload
   ```

2. **Visit the API documentation:**
   ```
   http://localhost:8000/docs
   ```

3. **Test the detection configuration:**
   ```bash
   python test_faster_rcnn.py
   ```

4. **Upload a video and see Faster R-CNN in action!**

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory
If you get CUDA out of memory errors:
```json
{
  "use_faster_rcnn": false,
  "use_yolo": true
}
```

#### Slow Performance
For faster processing, disable Faster R-CNN:
```json
{
  "use_faster_rcnn": false,
  "use_rtdetr": true
}
```

#### No Detections
Try lowering the confidence threshold:
```json
{
  "faster_rcnn_threshold": 0.5
}
```

## 🔮 Future Enhancements

- [ ] Support for custom Faster R-CNN models
- [ ] Batch processing optimization
- [ ] Model benchmarking tools
- [ ] Advanced ensemble methods
- [ ] Custom training pipeline

## 📞 Support

For issues or questions about the Faster R-CNN implementation:
1. Check the test script output: `python test_faster_rcnn.py`
2. Review API logs at `/detection/performance`
3. Verify GPU availability and memory

---

**🎉 Enjoy using Faster R-CNN for high-accuracy object detection!** 