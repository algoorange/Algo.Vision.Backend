from ultralytics import YOLO, RTDETR
import torch
import cv2
import numpy as np
# from app.services.crack_detector import predict_crack

# Load YOLOv8 model
yolo_model = YOLO("yolov8x.pt")

def detect_objects(frame):
    """
    Run YOLO detection only, return detections and annotated frame
    """
    detections = []
    detections2 = []

    # Run YOLOv8 Detection
    yolo_results = yolo_model(frame)[0]
    for box in yolo_results.boxes:
        cls_id = int(box.cls)
        label = yolo_model.names[cls_id]
        conf = float(box.conf)
        if conf >= 0.7:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "source": "YOLOv8"
            })


    #use rtdetr model
    rtdetr_model = RTDETR('rtdetr-l.pt')
    rtdetr_results = rtdetr_model(frame)[0]
    for box in rtdetr_results.boxes:
        cls_id = int(box.cls)
        label = rtdetr_model.names[cls_id]
        conf = float(box.conf)
        if conf >= 0.7:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections2.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "source": "RTDETR"
            })


   


    # Draw boxes and labels (truck in yellow, others in green)
    for det in detections2:
        x, y, w, h = det["bbox"]
        label = det["label"]
        conf = det["confidence"]
        color = (0, 255, 255) if label == "truck" else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


    return detections2, frame
