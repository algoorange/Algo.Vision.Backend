import os
import cv2
from app.services import object_detector, object_tracker
from app.utils.helpers import format_result, build_summary_prompt
from app.utils.embeddings import embedder, embedding_index, embedding_metadata
from app.services.summary_generate_by_llm import generate_summary
from fastapi import UploadFile
import uuid

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


async def process_video(file, video_id):
    """
    Processes a video file:
    - Runs object detection and tracking
    - Generates a summary using LLM
    - Stores embeddings for later querying
    - Extracts frames to frames/{video_id}/
    Args:
        file (UploadFile): The uploaded video file.
    Returns:
        dict: Summary and tracking details, including video_id.
    """
    # Create the frames directory for this video if it doesn't exist
    frames_dir = os.path.join(FRAMES_BASE_DIR, video_id)
    os.makedirs(frames_dir, exist_ok=True)

    # Save the uploaded video file with the video_id in its name
    video_filename = f"{video_id}_{file.filename}"

    video_path = os.path.join(UPLOAD_DIR, video_filename)

    with open(video_path, "wb") as video_file:
        video_file.write(await file.read())
    print(f"✅ Video saved at: {video_path}")

    # Open the saved video using OpenCV
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 1  # Prevent division by zero
    interval = int(fps * 5)  # Extract one frame every 5 seconds

    frame_number = 0  # Tracks the current frame number
    frames_saved = 0  # Counts how many frames we saved

    while True:
        success, frame = cap.read()
        if not success:
            break  # No more frames to read
        frame_number += 1

        # Only save a frame every 'interval' frames
        if frame_number % interval == 0:
            # Resize the frame to width 640px, keeping aspect ratio
            scale = 640 / frame.shape[1]
            new_height = int(frame.shape[0] * scale)
            frame_resized = cv2.resize(frame, (640, new_height))

            # Run object detection and get the annotated frame
            _, annotated_frame = object_detector.detect_objects(frame_resized.copy())

            # Save the annotated frame as a JPEG
            frames_id = f"{uuid.uuid4()}.jpg"
            frame_path = os.path.join(frames_dir, frames_id)
            cv2.imwrite(frame_path, annotated_frame)
            frames_saved += 1

    cap.release()  # Release the video file
    print(f"✅ Extracted {frames_saved} frames to {frames_dir}")

    # Tracking and summary (on full video, not just extracted frames)
    cap = cv2.VideoCapture(video_path)
    tracks = build_tracking_data(
        cap,
        object_detector.detect_objects,
        object_tracker.track_objects,
        fps,
        interval,
        video_id=video_id,
        frames_id=frames_id,
        video_name=video_filename,
        model_name="RTDETR"  # Change if you use a different model
    )
    result = format_result(tracks, frames_saved, fps, frames_saved / fps)
    cap.release()

    summary_prompt = build_summary_prompt(tracks)
    if "error" in summary_prompt.lower() or not summary_prompt.strip():
        summary = "Unable to generate summary for this video."
    else:
        summary = await generate_summary(summary_prompt)

    if summary and "Unable to generate" not in summary:
        embedding = embedder.encode(summary)
        embedding_index.add(embedding[None, :])
        embedding_metadata.append({
            "video": video_filename,
            "summary": summary,
            "tracks": result["tracks"],
            "duration": result["summary"]["duration_seconds"],
            "video_id": video_id,
            "frames_id": frames_id,
        })
    print("🎉 Video processing complete")

    return {
        "video_id": video_id,
        "summary": result["summary"],
        "natural_language_summary": summary,
        "tracks": result["tracks"],
        "file_path": video_filename,
        "frames_dir": video_id,
    }


def build_tracking_data(cap, detect_fn, track_fn, fps, interval, video_id=None, video_name=None, frames_id=None, model_name="RTDETR"):
    """
    Processes video frames, runs detection and tracking, and saves results to MongoDB.
    Args:
        cap: OpenCV video capture object
        detect_fn: function to detect objects
        track_fn: function to track objects
        fps: frames per second of the video
        interval: frame interval for processing
        video_id: unique id for the video
        frames_id: unique id for the frames
        video_name: name of the video file
        model_name: name of the detection model
    """
    print(f"[DEBUG] build_tracking_data called with video_id={video_id}, video_name={video_name}, model_name={model_name}")
    frame_id = 0
    track_db = {}
    while True:
        ret = cap.grab()
        if not ret:
            break
        if frame_id % interval == 0:
            ret, frame = cap.retrieve()
            if not ret:
                break
            detections, _ = detect_fn(frame)
            tracks = track_fn(frame, detections)

            print(f"[DEBUG] Tracks at frame {frame_id}: {tracks}")
            for t in tracks:
                print(f"[DEBUG] Track: {t}")
                if not t.is_confirmed():
                    print("[DEBUG] Track not confirmed, skipping.")
                    continue
                bbox = t.to_ltrb()
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                tid = t.track_id
                # --- MongoDB Insert: Save detection info for each confirmed object ---
                # Build a document for MongoDB
                doc = {
                    "video_id": video_id,
                    "video_name": os.path.splitext(video_name.split("_", 1)[1])[0],
                    "frames_id": os.path.splitext(frames_id)[0],
                    "track_id": tid,
                    "frame_time": round(frame_id / fps, 2),
                    "model_name": model_name,
                    "detected_object": t.det_class,
                    "position": {
                        "x": int(bbox[0]),
                        "y": int(bbox[1]),
                        "x1": int(bbox[2]),
                        "y1": int(bbox[3]),
                    }
                }
                # Insert the document into MongoDB with error handling and debug print
                try:
                    video_details_collection.insert_one(doc)
                    print(f"Inserted to MongoDB: {doc}")
                except Exception as e:
                    print(f"MongoDB insert error: {e}\nDoc: {doc}")
                # --- End MongoDB Insert ---
                if tid not in track_db:
                    track_db[tid] = {
                        "track_id": tid,
                        "label": t.det_class,
                        "trajectory": [],
                        "timestamps": [],
                        "frames": []
                    }
                track_db[tid]["trajectory"].append(center)
                track_db[tid]["timestamps"].append(round(frame_id / fps, 2))
                track_db[tid]["frames"].append(frame_id)
        frame_id += 1
    print(f"✅ Processed {frame_id} frames, {len(track_db)} tracks")
    return list(track_db.values())
