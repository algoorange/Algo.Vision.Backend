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
    if not file.filename:
        raise ValueError("File must have a filename")
    
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
        cap, object_detector.detect_objects_with_fast_rcnn, object_tracker.track_objects, fps, interval, file.filename
    )
    result = format_result(tracks, total_frames, fps, duration)

    cap.release()
    # shutil.rmtree(UPLOAD_DIR)

    summary_prompt = build_summary_prompt(tracks, max_objects=100)

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
            "crack_count": result["crack_count"]
        })

    print("🎉 Video processing complete")    

    return {
        "summary": result["summary"],
        "natural_language_summary": summary,
        "tracks": result["tracks"],
        "file_path": file.filename,
        "crack_count": result["crack_count"]
    }


def build_tracking_data(cap, detect_fn, track_fn, fps, interval, filename=None):
    """
    Builds tracking data for detected objects.

    Args:
        cap (cv2.VideoCapture): The video capture object.
        detect_fn (function): Object detection function.
        track_fn (function): Object tracking function.
        fps (float): Frames per second.
        interval (int): Frame interval for processing.
        filename (str): Optional filename for saving annotated frames.

    Returns:
        list: List of tracked objects.
    """
    frame_id = 0
    track_db = {}
    crack_count = 0
    
    # Create annotated frames directory if filename is provided
    annotated_frames_dir = None
    if filename:
        video_name = os.path.splitext(filename)[0]
        annotated_frames_dir = os.path.join(UPLOAD_DIR, f"{video_name}_processed_frames")
        os.makedirs(annotated_frames_dir, exist_ok=True)
        print(f"✅ Created processed frames directory: {annotated_frames_dir}")

    while True:
        ret = cap.grab()
        if not ret:
            break
            
        if frame_id % interval == 0:
            ret, frame = cap.retrieve()
            if not ret:
                break
                
            # Run object detection
            detections, annotated_frame = detect_fn(frame)
            
            # Save annotated frame if directory is set
            if annotated_frames_dir:
                frame_filename = f"processed_frame_{frame_id:06d}.jpg"
                frame_path = os.path.join(annotated_frames_dir, frame_filename)
                cv2.imwrite(frame_path, annotated_frame)
                print(f"💾 Saved processed frame: {frame_path}")
            
            # Count cracks
            for det in detections:
                if det.get("label") == "crack":
                    crack_count += 1
            
            # Track objects (excluding cracks)
            tracks = track_fn(annotated_frame, [d for d in detections if d.get("label") != "crack"])
            
            # Update track database
            for t in tracks:
                if not t.is_confirmed():
                    continue
                    
                tid = t.track_id
                bbox = t.to_ltrb()
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                
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
    
    if annotated_frames_dir:
        print(f"🎉 Processing complete. Annotated frames saved in: {annotated_frames_dir}")
    
    return list(track_db.values())