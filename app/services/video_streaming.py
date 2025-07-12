import cv2
import os
from fastapi.responses import StreamingResponse
from app.services import object_detector, object_tracker

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

RESIZE_WIDTH = 640
DETECTION_INTERVAL = 5  # Detect every 5 frames

def stream_video(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        raise RuntimeError(f"Video file does not exist: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {path}")
    frame_count = 0
    
    # Create annotated frames directory
    video_name = os.path.splitext(filename)[0]
    annotated_frames_dir = os.path.join(UPLOAD_DIR, f"{video_name}_annotated_frames")
    os.makedirs(annotated_frames_dir, exist_ok=True)
    print(f"✅ Created annotated frames directory: {annotated_frames_dir}")

    def generate_frames():
        tracks = []

        while True:
            success, frame = cap.read()
            if not success:
                break

            # Resize frame for faster processing
            scale_factor = RESIZE_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, (RESIZE_WIDTH, int(frame.shape[0] * scale_factor)))

            nonlocal frame_count
            frame_count += 1

            # Set the detection interval to process one frame per second
            if frame_count % DETECTION_INTERVAL == 0:
                detections, annotated_frame = object_detector.detect_objects_with_fast_rcnn(frame)
                tracks = object_tracker.track_objects(annotated_frame, detections)
                
                # Save annotated frame
                frame_filename = f"frame_{frame_count:06d}.jpg"
                frame_path = os.path.join(annotated_frames_dir, frame_filename)
                cv2.imwrite(frame_path, annotated_frame)
                print(f"💾 Saved annotated frame: {frame_path}")
                
                # Use annotated frame for streaming
                frame = annotated_frame

            # Draw ALL YOLO detections (even stationary ones)
                for det in detections:
                    if det["source"] == "YOLO" and det["bbox"] is not None:
                        x, y, w, h = det["bbox"]
                        label = det["label"]
                        conf = det["confidence"]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        cap.release()
        print(f"🎉 Video streaming complete. Annotated frames saved in: {annotated_frames_dir}")

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
