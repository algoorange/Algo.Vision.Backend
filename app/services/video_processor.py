import os
import cv2
from app.services import object_detector, object_tracker
from app.utils.helpers import format_result, build_summary_prompt
from app.utils.embeddings import embedder, embedding_index, embedding_metadata
from app.services.summary_generate_by_llm import generate_summary
from fastapi import UploadFile

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


async def save_video(file: UploadFile) -> str:
    """
    Save uploaded video to disk and return the saved file path
    """
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())
    print(f"✅ Video saved at: {path}")
    return path


async def process_video(file):
    """
    Processes a video file:
    - Runs object detection and tracking
    - Generates a summary using LLM
    - Stores embeddings for later querying

    Args:
        file (UploadFile): The uploaded video file.

    Returns:
        dict: Summary and tracking details.
    """
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())

    print(f"✅ Video saved at: {path}")    

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    interval = int(fps)  # Process 1 frame every 1 second

    tracks = build_tracking_data(
        cap, object_detector.detect_objects, object_tracker.track_objects, fps, interval
    )
    result = format_result(tracks, total_frames, fps, duration)

    cap.release()
    # shutil.rmtree(UPLOAD_DIR)

    summary_prompt = build_summary_prompt(tracks)

    # Fallback if summary fails
    if "error" in summary_prompt.lower() or not summary_prompt.strip():
        summary = "Unable to generate summary for this video."

    else:
        # Generate the actual summary when there's no error
        summary = await generate_summary(summary_prompt)

    # Add embedding only if summary succeeded
    if summary and "Unable to generate" not in summary:
        embedding = embedder.encode(summary)
        embedding_index.add(embedding[None, :])
        embedding_metadata.append({
            "video": file.filename,
            "summary": summary,
            "tracks": result["tracks"],
            "duration": result["summary"]["duration_seconds"],
            
        })

    print("🎉 Video processing complete")    

    return {
        "summary": result["summary"],
        "natural_language_summary": summary,
        "tracks": result["tracks"],
        "file_path": file.filename,
        
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

 # Run object detection
            detections, _ = detect_fn(frame)
            
            
            
            # Track objects (excluding cracks)
            tracks = track_fn(frame, detections)

            # Draw tracked objects with blue boxes
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
