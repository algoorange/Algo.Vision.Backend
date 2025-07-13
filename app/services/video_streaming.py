
# import cv2
# import os

# from app.services import object_detector, object_tracker

# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# RESIZE_WIDTH = 640
# DETECTION_INTERVAL = 5  # Detect every 5 frames

# def extract_and_save_frames(filename: str):
#     path = os.path.join(UPLOAD_DIR, filename)
#     cap = cv2.VideoCapture(path)
#     if not cap.isOpened():
#         raise RuntimeError(f"Failed to open video file: {path}")

#     frame_count = 0
#     FRAMES_DIR = "frames"
#     os.makedirs(FRAMES_DIR, exist_ok=True)

#     while True:
#         success, frame = cap.read()
#         if not success:
#             break

#         # Resize frame for consistent processing
#         scale_factor = RESIZE_WIDTH / frame.shape[1]
#         frame = cv2.resize(frame, (RESIZE_WIDTH, int(frame.shape[0] * scale_factor)))

#         frame_count += 1
#         if frame_count % DETECTION_INTERVAL == 0:
#             detections, annotated_frame = object_detector.detect_objects(frame.copy())
#             # Save annotated frame to frames folder
#             save_path = os.path.join(FRAMES_DIR, f"frame_{frame_count:06d}.jpg")
#             cv2.imwrite(save_path, annotated_frame)

#     cap.release()
#     return frame_count