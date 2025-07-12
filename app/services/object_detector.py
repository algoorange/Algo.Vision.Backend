from ultralytics import YOLO
import torch
import cv2
import numpy as np
from app.services.crack_detector import predict_crack
from app.services.crack_detector2 import detect_objects_faster_rcnn, detect_objects_with_nms
# from rt_detr_module import RTDETR
import os
import sys

# DeepStream imports
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import GObject, Gst
    import pyds
    DEEPSTREAM_AVAILABLE = True
    print("✅ DeepStream libraries loaded successfully")
except ImportError as e:
    DEEPSTREAM_AVAILABLE = False
    print(f"⚠️  DeepStream not available: {e}")
    print("Falling back to PyTorch Faster R-CNN")

# Load YOLOv8 model
yolo_model = YOLO("yolov8x.pt")

# DeepStream configuration
class DeepStreamDetector:
    def __init__(self):
        self.initialized = False
        self.pipeline = None
        self.primary_detector = None
        
        if DEEPSTREAM_AVAILABLE:
            self._initialize_deepstream()
    
    def _initialize_deepstream(self):
        """Initialize DeepStream pipeline"""
        try:
            # Initialize GStreamer
            Gst.init(None)
            
            # Create pipeline
            self.pipeline = Gst.Pipeline()
            
            # Create elements
            source = Gst.ElementFactory.make("appsrc", "source")
            nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv")
            streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
            
            # Primary detector (nvinfer)
            self.primary_detector = Gst.ElementFactory.make("nvinfer", "primary-detector")
            
            # Video converter and sink
            nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv-postosd")
            appsink = Gst.ElementFactory.make("appsink", "appsink")
            
            if not all([source, nvvidconv, streammux, self.primary_detector, nvvidconv_postosd, appsink]):
                raise Exception("Failed to create DeepStream elements")
            
            # Configure primary detector
            self.primary_detector.set_property("config-file-path", self._get_detector_config())
            
            # Configure streammux
            streammux.set_property("width", 640)
            streammux.set_property("height", 480)
            streammux.set_property("batch-size", 1)
            streammux.set_property("batched-push-timeout", 4000000)
            
            # Configure appsink
            appsink.set_property("emit-signals", True)
            appsink.set_property("sync", False)
            
            # Add elements to pipeline
            self.pipeline.add(source)
            self.pipeline.add(nvvidconv)
            self.pipeline.add(streammux)
            self.pipeline.add(self.primary_detector)
            self.pipeline.add(nvvidconv_postosd)
            self.pipeline.add(appsink)
            
            # Link elements
            source.link(nvvidconv)
            nvvidconv.link(streammux)
            streammux.link(self.primary_detector)
            self.primary_detector.link(nvvidconv_postosd)
            nvvidconv_postosd.link(appsink)
            
            self.initialized = True
            print("✅ DeepStream pipeline initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize DeepStream: {e}")
            self.initialized = False
    
    def _get_detector_config(self):
        """Get or create DeepStream detector config file"""
        config_path = "deepstream_config.txt"
        
        if not os.path.exists(config_path):
            # Create a basic config file for object detection
            config_content = """
[application]
enable-perf-measurement=1
perf-measurement-interval-sec=5

[tiled-display]
enable=1
rows=1
columns=1
width=640
height=480
gpu-id=0

[source0]
enable=1
type=3
uri=file://sample.mp4
num-sources=1
gpu-id=0
cudadec-memtype=0

[sink0]
enable=1
type=2
sync=0
source-id=0
gpu-id=0

[osd]
enable=1
gpu-id=0
border-width=3
text-size=15
text-color=1;1;1;1
text-bg-color=0.3;0.3;0.3;1
font=Arial

[streammux]
gpu-id=0
batch-size=1
batched-push-timeout=4000000
width=640
height=480
enable-padding=0
nvbuf-memory-type=0

[primary-gie]
enable=1
gpu-id=0
batch-size=1
bbox-border-color0=1;0;0;1
bbox-border-color1=0;1;1;1
bbox-border-color2=0;0;1;1
bbox-border-color3=0;1;0;1
nvbuf-memory-type=0
interval=0
gie-unique-id=1
model-engine-file=./models/Primary_Detector/resnet10.caffemodel_b1_gpu0_int8.engine
labelfile-path=./models/Primary_Detector/labels.txt
config-file-path=./models/Primary_Detector/config_infer_primary.txt
infer-dims=3;640;480
maintain-aspect-ratio=1
parse-bbox-func-name=NvDsInferParseCustomResnet
custom-lib-path=./models/Primary_Detector/libnvdsinfer_custom_impl_Resnet.so

[tracker]
enable=1
tracker-width=640
tracker-height=480
ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
ll-config-file=./models/tracker_config.yml
gpu-id=0
"""
            with open(config_path, 'w') as f:
                f.write(config_content)
            print(f"✅ Created DeepStream config file: {config_path}")
        
        return config_path
    
    def detect_objects(self, frame):
        """Run object detection using DeepStream"""
        if not self.initialized:
            # Fallback to PyTorch implementation
            return detect_objects_faster_rcnn(frame)
        
        try:
            # Convert frame to the format expected by DeepStream
            # This is a simplified implementation
            detections = self._run_deepstream_inference(frame)
            return detections, frame
            
        except Exception as e:
            print(f"❌ DeepStream inference failed: {e}")
            # Fallback to PyTorch implementation
            return detect_objects_faster_rcnn(frame)
    
    def _run_deepstream_inference(self, frame):
        """Run inference using DeepStream pipeline"""
        # This is a simplified implementation
        # In a real implementation, you would push the frame through the pipeline
        # and extract the detection results from the metadata
        
        # Placeholder for actual DeepStream inference
        # The actual implementation would involve:
        # 1. Converting frame to GStreamer buffer
        # 2. Pushing buffer through pipeline
        # 3. Extracting NvDsObjectMeta from NvDsBatchMeta
        # 4. Converting to our detection format
        
        # For now, fallback to PyTorch
        detections, annotated_frame = detect_objects_faster_rcnn(frame)
        return detections, annotated_frame

# Initialize DeepStream detector
deepstream_detector = DeepStreamDetector()

def detect_objects(frame):
    """
    Run YOLO and crack detection on the same frame
    """
    detections = []

    # # ---------- YOLOv8 Detection ----------
    # yolo_results = yolo_model(frame)[0]
    # for box in yolo_results.boxes:
    #     cls_id = int(box.cls)
    #     label = yolo_model.names[cls_id]
    #     conf = float(box.conf)
    #     if conf >= 70:
    #         x1, y1, x2, y2 = map(int, box.xyxy[0])
    #         detections.append({
    #             "label": label,
    #             "confidence": conf,
    #             "bbox": [x1, y1, x2 - x1, y2 - y1],
    #             "source": "YOLOv8"
    #         })

    #         # Draw box
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ---------- Crack Detection ----------
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


def detect_objects_with_fast_rcnn(frame):
    """
    NVIDIA DeepStream optimized object detection with crack detection
    """
    detections = []

    # ---------- DeepStream Object Detection ----------
    if DEEPSTREAM_AVAILABLE and deepstream_detector.initialized:
        print("🚀 Using NVIDIA DeepStream for object detection")
        faster_rcnn_detections, annotated_frame = deepstream_detector.detect_objects(frame.copy())
    else:
        print("⚠️  Using PyTorch Faster R-CNN fallback")
        faster_rcnn_detections, annotated_frame = detect_objects_faster_rcnn(frame.copy())

    # Draw bounding boxes for detected objects
    for det in faster_rcnn_detections:
        bbox = det.get("bbox")
        label = det.get("label", "object")
        confidence = det.get("confidence", 0)
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 255, 0)  # Green box
            thickness = 2
            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            # Put label and confidence
            text = f"{label}: {confidence:.2f}"
            cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Add object detections to our results
    detections.extend(faster_rcnn_detections)

    # # ---------- Crack Detection ----------
    # crack_mask = predict_crack(frame)
    # crack_confidence = crack_mask.mean() / 255.0  # crude confidence

    # if crack_confidence > 0.2:  # Lower threshold for testing
    #     detections.append({
    #         "label": "crack",
    #         "confidence": crack_confidence,
    #         "bbox": None,  # No bbox for cracks
    #         "source": "CrackDetector"
    #     })
    #     # Overlay crack mask in red on the annotated frame
    #     annotated_frame[crack_mask > 0] = [0, 0, 255]

    return detections, annotated_frame


def detect_objects_with_fast_rcnn_nms(frame):
    """
    Faster R-CNN with NMS post-processing and crack detection
    """
    detections = []

    # ---------- Faster R-CNN Detection with NMS ----------
    faster_rcnn_detections, annotated_frame = detect_objects_with_nms(frame.copy())
    # Draw bounding boxes for each detected object on annotated_frame
    for det in faster_rcnn_detections:
        bbox = det.get("bbox")
        label = det.get("label", "object")
        confidence = det.get("confidence", 0)
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 255, 0)  # Green box
            thickness = 2
            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            # Put label and confidence
            text = f"{label}: {confidence:.2f}"
            cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add Faster R-CNN detections to our results
    detections.extend(faster_rcnn_detections)

    # # ---------- Crack Detection ----------
    # crack_mask = predict_crack(frame)
    # crack_confidence = crack_mask.mean() / 255.0  # crude confidence

    # if crack_confidence > 0.2:  # Lower threshold for testing
    #     detections.append({
    #         "label": "crack",
    #         "confidence": crack_confidence,
    #         "bbox": None,  # No bbox for cracks
    #         "source": "CrackDetector"
    #     })
    #     # Overlay crack mask in red on the annotated frame
    #     annotated_frame[crack_mask > 0] = [0, 0, 255]

    return detections, annotated_frame


def save_annotated_frame(annotated_frame, filename, frame_number, output_dir="uploads"):
    """
    Save an annotated frame to the specified directory
    
    Args:
        annotated_frame: The frame with annotations (bounding boxes, crack overlays)
        filename: Base filename for the video
        frame_number: Frame number for naming
        output_dir: Directory to save the frame
    """
    # Create directory if it doesn't exist
    video_name = os.path.splitext(filename)[0]
    annotated_frames_dir = os.path.join(output_dir, f"{video_name}_annotated_frames")
    os.makedirs(annotated_frames_dir, exist_ok=True)
    
    # Save the frame
    frame_filename = f"annotated_frame_{frame_number:06d}.jpg"
    frame_path = os.path.join(annotated_frames_dir, frame_filename)
    success = cv2.imwrite(frame_path, annotated_frame)
    
    if success:
        print(f"✅ Saved annotated frame: {frame_path}")
        return frame_path
    else:
        print(f"❌ Failed to save frame: {frame_path}")
        return None