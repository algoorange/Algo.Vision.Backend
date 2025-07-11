from ultralytics import YOLO
import torch
import cv2
import numpy as np
from app.services.crack_detector import predict_crack
# from rt_detr_module import RTDETR


# Load YOLOv8 model
yolo_model = YOLO("yolov8x.pt")

def detect_objects(frame):
    """
    Run YOLO and crack detection on the same frame
    """
    detections = []

    # ---------- YOLOv8 Detection ----------
    yolo_results = yolo_model(frame)[0]
    for box in yolo_results.boxes:
        cls_id = int(box.cls)
        label = yolo_model.names[cls_id]
        conf = float(box.conf)
        if conf >= 70:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "source": "YOLOv8"
            })

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



# Run Crack Detector
    crack_mask = predict_crack(frame)
    crack_confidence = crack_mask.mean() / 255.0  # crude confidence

    if crack_confidence > 0.2:  # Lower threshold for testing
        detections.append({
            "label": "crack",
            "confidence": crack_confidence,
            "bbox": None,  # No bbox for cracks
            "source": "CrackDetector"
        })
        # Overlay mask in red
        frame[crack_mask > 0] = [0, 0, 255]

    return detections, frame