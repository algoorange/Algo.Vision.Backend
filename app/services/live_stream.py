from fastapi import WebSocket, WebSocketDisconnect
import cv2
import asyncio
from collections import deque
from app.services import object_detector, object_tracker

frame_queue = deque(maxlen=10)

async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            if frame_queue:
                frame_bytes = frame_queue.popleft()
                await websocket.send_bytes(frame_bytes)
            else:
                await asyncio.sleep(0.033)  # Wait for new frames
    except WebSocketDisconnect:
        print("Client disconnected")

def process_frame(frame):
    detections = object_detector.detect_objects(frame)
    tracks = object_tracker.track_objects(frame, detections)
    for t in tracks:
        if not t.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, t.to_ltrb())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

async def process_video_for_streaming(file_path):
    cap = cv2.VideoCapture(file_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        frame_queue.append(frame_bytes)
        await asyncio.sleep(0.033)  # Simulate real-time processing delay
    cap.release()




#############################################################

                #     FOR TESTING    #

        #  THIS CODE IS FOR THE STREAMING LIVE CAMERA # 
                          
#############################################################