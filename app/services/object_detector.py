	
from ultralytics import YOLO, RTDETR
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import detection
import torchvision
# from app.services.crack_detector import predict_crack

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# Load Faster R-CNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
faster_rcnn_model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn_model.to(device)
faster_rcnn_model.eval()

# COCO class names for Faster R-CNN
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Transform for Faster R-CNN
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Detection configuration
DETECTION_CONFIG = {
    "use_yolo": False,
    "use_rtdetr": False,
    "use_faster_rcnn": True,
    "yolo_threshold": 0.60,
    "rtdetr_threshold": 0.60,
    "faster_rcnn_threshold": 0.60,
    "prioritize_faster_rcnn": True  # If True, Faster R-CNN results come first
}

# --- Helper function for IOU ---
def compute_iou(box1, box2):
    """Compute Intersection over Union (IOU) for two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    box1_x2, box1_y2 = x1 + w1, y1 + h1
    box2_x2, box2_y2 = x2 + w2, y2 + h2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(box1_x2, box2_x2), min(box1_y2, box2_y2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def merge_detections(detections, iou_threshold=0.7):
    """Merge duplicate detections based on IOU and label."""
    merged = []
    for det in detections:
        duplicate = False
        for m in merged:
            iou = compute_iou(det["bbox"], m["bbox"])
            if iou > iou_threshold and det["label"] == m["label"]:
                m["confidence"] = max(m["confidence"], det["confidence"])  # Take higher confidence
                duplicate = True
                break
        if not duplicate:
            merged.append(det)
    return merged


def detect_with_yolo(frame):
    """YOLOv8 detection"""
    detections = []
    yolo_results = yolo_model(frame)[0]
    for box in yolo_results.boxes:
        cls_id = int(box.cls)
        label = yolo_model.names[cls_id]
        conf = float(box.conf)
        if conf >= DETECTION_CONFIG["yolo_threshold"]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "source": "YOLOv8"
            })
    return detections

def detect_with_rtdetr(frame):
    """RTDETR detection"""
    detections = []
    rtdetr_model = RTDETR('rtdetr-l.pt')
    rtdetr_results = rtdetr_model(frame)[0]
    for box in rtdetr_results.boxes:
        cls_id = int(box.cls)
        label = rtdetr_model.names[cls_id]
        conf = float(box.conf)
        if conf >= DETECTION_CONFIG["rtdetr_threshold"]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "source": "RTDETR"
            })
    return detections

def detect_with_faster_rcnn(frame):
    """Faster R-CNN detection"""
    detections = []
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = transform(frame_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = faster_rcnn_model(frame_tensor)
    
    # Process Faster R-CNN predictions
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    
    print(f"Faster R-CNN detections boxes: {len(boxes)}")
    print(f"Faster R-CNN detections labels : {labels}")
    print(f"Faster R-CNN detections scores : {scores}")

    for box, label_id, score in zip(boxes, labels, scores):
        print(f" -- Faster R-CNN detections box: {box}")
        print(f" -- Faster R-CNN detections label_id: {label_id}")
        print(f" -- Faster R-CNN detections score: {score}")
        if score >= DETECTION_CONFIG["faster_rcnn_threshold"]:
            x1, y1, x2, y2 = map(int, box)
            try:
                label = COCO_CLASSES[label_id]
                detections.append({
                    "label": label,
                    "confidence": float(score),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "source": "Faster R-CNN"
                })
            except:
                print(f" -->>>>>>>>>> Faster R-CNN detections label_id: {label_id}")
                print(f" -->>>>>>>>>> Faster R-CNN detections score: {score}")
                print(f" -->>>>>>>>>> Faster R-CNN detections box: {box}") 
    return detections

def detect_objects(frame):
    """
    Run detection with selected models (YOLO, RTDETR, and/or Faster R-CNN)
    Returns combined detections and annotated frame
    """
    all_detections = []

    # Run detections based on configuration
    if DETECTION_CONFIG["use_yolo"]:
        yolo_detections = detect_with_yolo(frame)
        all_detections.extend(yolo_detections)

    if DETECTION_CONFIG["use_rtdetr"]:
        rtdetr_detections = detect_with_rtdetr(frame)
        all_detections.extend(rtdetr_detections)

    if DETECTION_CONFIG["use_faster_rcnn"]:
        faster_rcnn_detections = detect_with_faster_rcnn(frame)
        all_detections.extend(faster_rcnn_detections)

    # Sort detections by priority if configured
    if DETECTION_CONFIG["prioritize_faster_rcnn"]:
        all_detections.sort(key=lambda x: {
            "Faster R-CNN": 0, 
            "RTDETR": 1, 
            "YOLOv8": 2
        }.get(x["source"], 3))

    # Draw boxes and labels with different colors for each model
    annotated_frame = frame.copy()
    for det in all_detections:
        x, y, w, h = det["bbox"]
        label = det["label"]
        conf = det["confidence"]
        source = det["source"]
        
        # Different colors for different models
        if source == "Faster R-CNN":
            color = (255, 0, 0)  # Blue
        elif source == "RTDETR":
            color = (0, 255, 255)  # Yellow
        else:  # YOLOv8
            color = (0, 255, 0)  # Green
        # Special color for trucks
        if label == "truck":
            color = (0, 255, 255)  # Yellow for trucks
            
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated_frame, f"{source}: {label} {conf:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    merged_detections = merge_detections(all_detections)                
    return merged_detections, annotated_frame

def set_detection_config(**kwargs):
    """Update detection configuration"""
    for key, value in kwargs.items():
        if key in DETECTION_CONFIG:
            DETECTION_CONFIG[key] = value

def get_detection_config():
    """Get current detection configuration"""
    return DETECTION_CONFIG.copy()

def get_available_models():
    """Get list of available detection models"""
    return ["YOLOv8", "RTDETR", "Faster R-CNN"]
