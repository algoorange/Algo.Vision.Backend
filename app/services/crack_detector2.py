import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np

# COCO class names for Faster R-CNN (0 is background, 1-90 are classes)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Faster R-CNN model (ResNet-50 FPN backbone)
# This implements the architecture described in the arXiv paper
faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
faster_rcnn_model.to(device)
faster_rcnn_model.eval()

# Preprocessing transform for Faster R-CNN
faster_rcnn_transform = transforms.Compose([
    transforms.ToTensor(),
])

def detect_objects_faster_rcnn(frame, confidence_threshold=0.7):
    """
    Object detection using Faster R-CNN model from arXiv:1506.01497
    
    Args:
        frame: Input image frame (BGR format from OpenCV)
        confidence_threshold: Minimum confidence score for detections
        
    Returns:
        detections: List of detected objects with bounding boxes and labels
        annotated_frame: Frame with detection boxes drawn
    """
    detections = []
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Preprocess frame
    input_tensor = faster_rcnn_transform(rgb_frame)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # Add batch dimension
    
    # Run inference
    with torch.no_grad():
        outputs = faster_rcnn_model(input_tensor)[0]
    
    # Process detections
    boxes = outputs['boxes'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()
    labels = outputs['labels'].cpu().numpy()
    
    for idx in range(len(boxes)):
        confidence = float(scores[idx])
        
        if confidence >= confidence_threshold:
            x1, y1, x2, y2 = map(int, boxes[idx])
            label_idx = int(labels[idx])
            
            # Get class name
            class_name = COCO_INSTANCE_CATEGORY_NAMES[label_idx] if label_idx < len(COCO_INSTANCE_CATEGORY_NAMES) else str(label_idx)
            
            # Store detection
            detections.append({
                "label": class_name,
                "confidence": confidence,
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # Convert to [x, y, width, height]
                "source": "Faster R-CNN"
            })
            
            # Draw bounding box on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return detections, frame

def detect_objects_with_nms(frame, confidence_threshold=0.7, nms_threshold=0.5):
    """
    Object detection with Non-Maximum Suppression (NMS) post-processing
    
    Args:
        frame: Input image frame
        confidence_threshold: Minimum confidence score
        nms_threshold: NMS IoU threshold
        
    Returns:
        detections: List of filtered detections
        annotated_frame: Frame with detection boxes
    """
    detections = []
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Preprocess frame
    input_tensor = faster_rcnn_transform(rgb_frame)
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = faster_rcnn_model(input_tensor)[0]
    
    # Apply NMS
    keep_indices = torchvision.ops.nms(
        outputs['boxes'], 
        outputs['scores'], 
        nms_threshold
    )
    
    # Filter results
    boxes = outputs['boxes'][keep_indices].cpu().numpy()
    scores = outputs['scores'][keep_indices].cpu().numpy()
    labels = outputs['labels'][keep_indices].cpu().numpy()
    
    for idx in range(len(boxes)):
        confidence = float(scores[idx])
        
        if confidence >= confidence_threshold:
            x1, y1, x2, y2 = map(int, boxes[idx])
            label_idx = int(labels[idx])
            
            class_name = COCO_INSTANCE_CATEGORY_NAMES[label_idx] if label_idx < len(COCO_INSTANCE_CATEGORY_NAMES) else str(label_idx)
            
            detections.append({
                "label": class_name,
                "confidence": confidence,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "source": "Faster R-CNN (NMS)"
            })
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    return detections, frame

def get_model_info():
    """
    Get information about the Faster R-CNN model
    """
    return {
        "model_name": "Faster R-CNN with ResNet-50 FPN",
        "paper": "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks",
        "arxiv_url": "https://arxiv.org/abs/1506.01497",
        "authors": ["Shaoqing Ren", "Kaiming He", "Ross Girshick", "Jian Sun"],
        "backbone": "ResNet-50 with Feature Pyramid Network (FPN)",
        "dataset": "COCO",
        "num_classes": len(COCO_INSTANCE_CATEGORY_NAMES) - 1,  # Exclude background
        "device": str(device)
    }

# Example usage and testing
if __name__ == "__main__":
    # Test with a sample image
    import os
    
    print("Faster R-CNN Object Detector")
    print("=" * 40)
    
    # Print model info
    info = get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print("\nModel ready for inference!")
    print("Use detect_objects_faster_rcnn(frame) for object detection")
    print("Use detect_objects_with_nms(frame) for detection with NMS post-processing")
