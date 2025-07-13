import os
import cv2
from app.services import object_detector, object_tracker
from app.utils.helpers import format_result, build_summary_prompt
from app.utils.embeddings import embedder, embedding_index, embedding_metadata
from app.services.summary_generate_by_llm import generate_summary
from fastapi import UploadFile

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
    # Generate unique video ID
    frames_dir = os.path.join(FRAMES_BASE_DIR, video_id)
    os.makedirs(frames_dir, exist_ok=True)

    # Save video with UUID
    filename = f"{video_id}_{file.filename}"
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(await file.read())
    print(f"✅ Video saved at: {path}")

    # Extract frames to frames/{video_id}/
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0
    interval = int(fps * 10) if fps else 1  # Process 1 frame every 10 seconds

    frame_id = 0
    saved_frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % interval == 0:
            scale_factor = 640 / frame.shape[1]
            frame_resized = cv2.resize(frame, (640, int(frame.shape[0] * scale_factor)))
            detections, annotated_frame = object_detector.detect_objects(frame_resized.copy())
            save_path = os.path.join(frames_dir, f"frame_{frame_id:06d}.jpg")
            cv2.imwrite(save_path, annotated_frame)
            saved_frame_count += 1
    cap.release()
    print(f"✅ Extracted {saved_frame_count} frames to {frames_dir}")

    # Tracking and summary (on full video, not just extracted frames)
    cap = cv2.VideoCapture(path)
    tracks = build_tracking_data(
        cap, object_detector.detect_objects, object_tracker.track_objects, fps, interval
    )
    result = format_result(tracks, total_frames, fps, duration)
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
            "video": filename,
            "summary": summary,
            "tracks": result["tracks"],
            "duration": result["summary"]["duration_seconds"],
            "video_id": video_id,
        })
    print("🎉 Video processing complete")

    return {
        "video_id": video_id,
        "summary": result["summary"],
        "natural_language_summary": summary,
        "tracks": result["tracks"],
        "file_path": filename,
        "frames_dir": video_id,
    }


def build_tracking_data(cap, detect_fn, track_fn, fps, interval):
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
            for t in tracks:
                if not t.is_confirmed():
                    continue
                bbox = t.to_ltrb()
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                tid = t.track_id
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
