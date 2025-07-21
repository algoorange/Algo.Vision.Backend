import os
import cv2
import numpy as np
from app.services import object_detector, object_tracker
from app.utils.helpers import format_result, build_summary_prompt
from app.utils.embeddings import embedder, embedding_index, embedding_metadata
from app.services.summary_generate_by_llm import generate_summary
from fastapi import UploadFile
import uuid
from app.utils.helpers import is_point_in_polygon, scale_coordinates, is_bbox_in_polygon

# --- MongoDB Setup ---
from pymongo import MongoClient
# Connect to MongoDB (make sure MongoDB is running on localhost:27017)
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["algo_compliance_db_2"]
video_details_collection = db["video_details"]
# --- End MongoDB Setup ---

UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'uploads'))
os.makedirs(UPLOAD_DIR, exist_ok=True)
FRAMES_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'frames'))
os.makedirs(FRAMES_BASE_DIR, exist_ok=True)


async def save_video(file: UploadFile, video_id: str) -> str:
    """
    Save uploaded video to disk using UUID and return the saved file path
    """
    filename = f"{video_id}_{file.filename}"
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(await file.read())
    print(f"✅ Video saved at: {path}")
    return path


async def process_video(file: UploadFile, video_id: str, coords=None, preview_width=None, preview_height=None):
    video_filename = f"{video_id}_{file.filename}"
    video_path = await save_video(file, video_id)

    cap = cv2.VideoCapture(video_path)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if coords:
        # Use provided preview size for scaling, fallback to 640x360
        pw = int(preview_width) if preview_width else 640
        ph = int(preview_height) if preview_height else 360
        coords = scale_coordinates(coords, pw, ph, video_width, video_height)

    interval = int(fps * 1)
    frame_number = 0
    frames_saved = 0
    frameid_map = {}
    last_saved_frames_id = None
    frames_dir = os.path.join(FRAMES_BASE_DIR, video_id)
    os.makedirs(frames_dir, exist_ok=True)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_number += 1
        if frame_number % interval == 0:
            scale_ratio = 640 / frame.shape[1]
            resized_frame = cv2.resize(frame, (640, int(frame.shape[0] * scale_ratio)))
            detected_objects, annotated_frame = object_detector.detect_objects(resized_frame.copy())

            filtered_objects = []
            # Debug: Print zone and bbox info for first frame with coords
            if coords and frame_number == interval:
                print("[DEBUG] Zone polygon (coords):", coords)
                for obj in detected_objects:
                    x, y, w, h = obj["bbox"]
                    scale = frame.shape[1] / 640
                    orig_bbox = [int(x * scale), int(y * scale), int(w * scale), int(h * scale)]
                    print(f"[DEBUG] Detected bbox (original): {orig_bbox}")
            for obj in detected_objects:
                # Ensure bbox is in original video scale
                # If detection is on resized frame, rescale bbox to original
                x, y, w, h = obj["bbox"]
                scale = frame.shape[1] / 640  # original/resized
                orig_bbox = [int(x * scale), int(y * scale), int(w * scale), int(h * scale)]
                if not coords or is_bbox_in_polygon(orig_bbox, coords):
                    filtered_objects.append(obj)

            if filtered_objects:
                if coords and preview_width and preview_height:
                    scale_x = annotated_frame.shape[1] / video_width
                    scale_y = annotated_frame.shape[0] / video_height

                    scaled_pts = np.array([
                        [int(pt['x'] * scale_x), int(pt['y'] * scale_y)]
                        for pt in coords
                    ], np.int32).reshape((-1, 1, 2))

                    cv2.polylines(annotated_frame, [scaled_pts], isClosed=True, color=(0, 0, 255), thickness=2)
                frames_id = f"{uuid.uuid4()}.jpg"
                frame_path = os.path.join(frames_dir, frames_id)
                cv2.imwrite(frame_path, annotated_frame)
                frames_saved += 1
                frameid_map[frame_number] = frames_id
                last_saved_frames_id = frames_id

    cap.release()

    cap = cv2.VideoCapture(video_path)
    tracks = build_tracking_data(
        cap, object_detector.detect_objects, object_tracker.track_objects, fps, interval, video_id, video_filename, frameid_map, coords
    )
    cap.release()

    result = format_result(tracks, frames_saved, fps, frames_saved / fps)
    summary_prompt = build_summary_prompt(tracks)
    summary = await generate_summary(summary_prompt) if summary_prompt.strip() else "Unable to generate summary."

    embedding = embedder.encode(summary)
    embedding_index.add(embedding[None, :])
    embedding_metadata.append({
        "video": video_filename,
        "summary": summary,
        "tracks": result["tracks"],
        "duration": result["summary"]["duration_seconds"],
        "video_id": video_id,
        "coords": coords,
        "frames_id": frameid_map,
    })
    print("🎉 Video processing complete")

    return {
        "video_id": video_id,
        "summary": result["summary"],
        "natural_language_summary": summary,
        "tracks": result["tracks"],
        "frames_file_name": os.path.splitext(last_saved_frames_id)[0] if last_saved_frames_id else "",
        "file_path": video_filename,
        "frames_dir": video_id,
    }

def build_tracking_data(cap, detect_fn, track_fn, fps, interval, video_id, video_name, frameid_map, zone_coords=None):
    frame_number = 0
    track_db = {}
    bulk_docs = []

    while True:
        ret = cap.grab()
        if not ret:
            break

        if frame_number % interval == 0:
            ret, frame = cap.retrieve()
            if not ret:
                break

            detections, _ = detect_fn(frame)
            if zone_coords:
                detections = [det for det in detections if is_point_in_polygon(det["bbox"][0], det["bbox"][1], zone_coords)]

            tracks = track_fn(frame, detections)
            for obj in tracks:
                cur_frames_id = frameid_map.get(frame_number, "")
                doc = {
                    "video_id": video_id,
                    "video_name": os.path.splitext(video_name.split("_", 1)[1])[0],
                    "frames_id": os.path.splitext(cur_frames_id)[0],
                    "track_id": obj["track_id"],
                    "frame_time": round(frame_number / fps, 2),
                    "model_name": "RTDETR",
                    "frame": frame_number,
                    "detected_object": obj["object_type"],
                    "confidence": obj["confidence"],
                    "position": obj["position"]
                }
                bulk_docs.append(doc)

                if obj["track_id"] not in track_db:
                    track_db[obj["track_id"]] = {
                        "track_id": obj["track_id"],
                        "label": obj["object_type"],
                        "trajectory": [],
                        "timestamps": [],
                        "frames": []
                    }

                track_db[obj["track_id"]]["trajectory"].append(
                    ((obj["position"]["x"] + obj["position"]["x1"]) / 2, (obj["position"]["y"] + obj["position"]["y1"]) / 2)
                )
                track_db[obj["track_id"]]["timestamps"].append(round(frame_number / fps, 2))
                track_db[obj["track_id"]]["frames"].append(frame_number)

        frame_number += 1

    if bulk_docs:
        video_details_collection.insert_many(bulk_docs)

    return list(track_db.values())