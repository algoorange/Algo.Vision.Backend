import os
import cv2
import numpy as np
from app.services import object_detector, object_tracker
from app.utils.helpers import format_result, build_summary_prompt, scale_coordinates, is_bbox_in_polygon
from app.utils.embeddings import embedder, embedding_index, embedding_metadata
from app.services.summary_generate_by_llm import generate_summary
from app.services.llava_groq_service import analyze_video_frames_with_llava, generate_video_summary_with_llava
from fastapi import UploadFile
import uuid
import numpy as np
import tempfile
from app.utils.helpers import is_point_in_polygon
import datetime
from app.services.chromadb_service import chromadb_service

# --- MongoDB Setup ---
from pymongo import MongoClient
# Connect to MongoDB (make sure MongoDB is running on localhost:27017)
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["algo_compliance_db_2"]
video_details_collection = db["video_details"]
video_details_collection_llava = db["video_details_llava"]

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
        # pw = int(preview_width) if preview_width else 640
        # ph = int(preview_height) if preview_height else 360
        # coords = scale_coordinates(coords, pw, ph, video_width, video_height)

        coords = scale_coordinates(coords, video_width, video_height)

    # interval = int(fps * 0.5)
    interval = max(1, int(fps * 0.25))

    frame_number = 0
    frames_saved = 0
    frameid_map = {}
    last_saved_frames_id = None
    frames_dir = os.path.join(FRAMES_BASE_DIR, video_id)
    os.makedirs(frames_dir, exist_ok=True)

    tracks_by_frame = {}  
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
                # --- TRACKING AND ANNOTATION WITH TRACK ID ---
                # Run tracking on the resized frame and filtered detections
                tracks = object_tracker.track_objects(resized_frame, filtered_objects)
                if tracks:  # Only proceed if there are actual tracked objects
                    tracks_by_frame[frame_number] = tracks  # <-- Store tracks for this frame

                    # Draw bounding box and track_id for each tracked object
                    for track in tracks:
                        pos = track.get("position", {})
                        track_id = track.get("track_id")
                        conf = track.get("confidence", 0.0)
                        obj_type = track.get("object_type", "object")
                        if pos and track_id is not None:
                            x1, y1, x2, y2 = pos.get("x", 0), pos.get("y", 0), pos.get("x1", 0), pos.get("y1", 0)
                            color = (0, 255, 0)
                            if conf > 0.7:
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            label_text = f"ID:{track_id} {obj_type} {conf:.2f}"
                            # Draw the track_id just above the bounding box (or inside if space)
                            text_x, text_y = x1, max(y1 - 10, 0)
                            cv2.putText(annotated_frame, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Draw zone polygon if needed
                    if coords and preview_width and preview_height:
                        scale_x = annotated_frame.shape[1] / video_width
                        scale_y = annotated_frame.shape[0] / video_height
                        scaled_pts = np.array([
                            [int(pt['x'] * scale_x), int(pt['y'] * scale_y)]
                            for pt in coords
                        ], np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [scaled_pts], isClosed=True, color=(0, 0, 255), thickness=2)
                    frames_id = f"frame_{frame_number:05d}.jpg"
                    frame_save_path = os.path.join(frames_dir, frames_id)
                    cv2.imwrite(frame_save_path, annotated_frame)
                    frames_saved += 1
                    frameid_map[frame_number] = frames_id
                    last_saved_frames_id = frames_id
            # If no filtered_objects or no tracks, do NOT save frame or add to frameid_map or tracks_by_frame!

    # --- Collect tracked objects and trajectories in the main loop ---
    frames_data = []
    track_db = {}
    frame_number = 0  # Ensure frame_number is correct if not already
    for fn in sorted(frameid_map):
        frames_id = frameid_map[fn]
        tracks = tracks_by_frame.get(fn, [])  # <-- Restore this line
        frame_objects = []
        for obj in tracks:
            if obj.get("confidence", 0) == 0:
                continue
            track_id = obj.get("track_id")
            if track_id is not None:
                object_type = obj.get("object_type", "unknown")
                position = obj.get("position", {})
                if position and "x" in position and "x1" in position and "y" in position and "y1" in position:
                    frame_objects.append({
                        "track_id": track_id,
                        "object_type": object_type,
                        "confidence": obj.get("confidence", 0.0),
                        "position": position,
                        "bbox": [position["x"], position["y"], position["x1"] - position["x"], position["y1"] - position["y"]],
                        "model_name": "RTDETR",
                        "frame_number": fn,
                        "frame_time": round(fn / fps, 2)
                    })

                if track_id not in track_db:
                    track_db[track_id] = {
                        "track_id": track_id,
                        "label": object_type,
                        "trajectory": [],
                        "timestamps": [],
                        "frames": []
                    }
                if position and "x" in position and "x1" in position and "y" in position and "y1" in position:
                    track_db[track_id]["trajectory"].append(
                        ((position["x"] + position["x1"]) / 2, (position["y"] + position["y1"]) / 2)
                    )
                    track_db[track_id]["timestamps"].append(round(fn / fps, 2))
                    track_db[track_id]["frames"].append(fn)
        if frame_objects:
            frame_data = {
                "frame_id": os.path.splitext(frames_id)[0],
                "frame_number": fn,
                "frame_time": round(fn / fps, 2),
                "total_tracked_objects": len(frame_objects),
                "objects": frame_objects,
            }
            frames_data.append(frame_data)

    # Compose the full video document
    video_doc = {
        "video_id": video_id,
        "video_name": os.path.splitext(video_filename.split("_", 1)[1])[0] if "_" in video_filename else video_filename,
        "fps": fps,
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps if fps > 0 else 0,
        "frames": frames_data,
        "created_at": datetime.datetime.now(datetime.timezone.utc),
        "updated_at": datetime.datetime.now(datetime.timezone.utc),
        "file_path": video_filename,
        "frames_dir": video_id,
        "ordered_frame_ids": [frameid_map[k] for k in sorted(frameid_map)]
    }
    video_details_collection.insert_one(video_doc)
    chromadb_service.store_frame_objects(video_id, frames_data)

    # Optionally, you can build summary/tracks from just the tracks (not frames_data)
    result = format_result(list(track_db.values()), frames_saved, fps, frames_saved / fps)
    summary_prompt = build_summary_prompt(list(track_db.values()))
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

    # Return frames in original order
    ordered_frame_ids = [frameid_map[k] for k in sorted(frameid_map)]
    # Store in ChromaDB
    video_data = {
        "video_id": video_id,
        "summary": result["summary"],
        "natural_language_summary": summary,
        "tracks": result["tracks"],
        "frames_file_name": os.path.splitext(last_saved_frames_id)[0] if last_saved_frames_id else "",
        "file_path": video_filename,
        "frames_dir": video_id,
        "ordered_frame_ids": ordered_frame_ids,
        "coords": coords,
    }
    
    # Store video analysis in ChromaDB
    chromadb_service.store_video_analysis(video_data)
    
    # Store frame objects in ChromaDB (if frames data is available)
    if hasattr(result, 'frames') and result.get('frames'):
        chromadb_service.store_frame_objects(video_id, result['frames'])

    return video_data

# --- REMOVE build_tracking_data function, as it is no longer needed ---


# async def process_video_with_laava(file: UploadFile, video_id: str, coords=None, preview_width=None, preview_height=None):
    """
    Process video using LLaVA (Large Language and Vision Assistant) via Groq API.
    This function analyzes video frames using vision-language model for comprehensive understanding.
    """
    video_filename = f"{video_id}_{file.filename}"
    video_path = await save_video(file, video_id)

    cap = cv2.VideoCapture(video_path)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"🎬 Processing video with LLaVA: {video_filename}")
    print(f"📊 Video specs: {video_width}x{video_height}, {fps:.2f} FPS, {duration:.2f}s duration")

    if coords:
        # Use provided preview size for scaling, fallback to 640x360
        pw = int(preview_width) if preview_width else 640
        ph = int(preview_height) if preview_height else 360
        coords = scale_coordinates(coords, pw, ph, video_width, video_height)
        print(f"🎯 Zone coordinates scaled for analysis")

    # Extract frames for LLaVA analysis (every 2 seconds or max 10 frames to avoid API limits)
    interval = max(int(fps * 2), 1)  # Every 2 seconds
    max_frames = 10  # Limit to avoid API rate limits
    frame_number = 0
    frames_for_analysis = []
    frame_metadata = []
    frames_dir = os.path.join(FRAMES_BASE_DIR, video_id)
    os.makedirs(frames_dir, exist_ok=True)

    print(f"🔍 Extracting frames every {interval} frames (every ~{interval/fps:.1f}s)")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_number += 1
        if frame_number % interval == 0 and len(frames_for_analysis) < max_frames:
            # Store original frame for LLaVA analysis
            timestamp = frame_number / fps
            
            # Apply zone filtering if specified
            frame_to_analyze = frame.copy()
            if coords:
                # Create a mask for the zone
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                zone_points = np.array([[int(pt['x']), int(pt['y'])] for pt in coords], np.int32)
                cv2.fillPoly(mask, [zone_points], 255)
                
                # Apply mask to frame (optional - you might want full frame context)
                # frame_to_analyze = cv2.bitwise_and(frame, frame, mask=mask)
                
                # Draw zone boundary on frame for visualization
                cv2.polylines(frame_to_analyze, [zone_points], isClosed=True, color=(0, 0, 255), thickness=3)

            frames_for_analysis.append(frame_to_analyze)
            frame_metadata.append({
                "frame_number": frame_number,
                "timestamp": timestamp,
                "has_zone": coords is not None
            })

            # Save annotated frame
            frames_id = f"frame_{frame_number:05d}.jpg"
            frame_path = os.path.join(frames_dir, frames_id)
            cv2.imwrite(frame_path, frame_to_analyze)
            print(f"📸 Saved frame {frame_number} at {timestamp:.2f}s for LLaVA analysis")

    cap.release()

    if not frames_for_analysis:
        print("❌ No frames extracted for analysis")
        return {
            "video_id": video_id,
            "summary": "No frames could be extracted from the video",
            "natural_language_summary": "Unable to analyze video - no frames extracted",
            "llava_analysis": [],
            "frames_file_name": "",
            "file_path": video_filename,
            "frames_dir": video_id,
        }

    print(f"🧠 Analyzing {len(frames_for_analysis)} frames with LLaVA...")

    # Create prompts for each frame
    base_prompt = "Analyze this video frame and describe:"
    if coords:
        base_prompt += " Pay special attention to the red outlined zone area."
    
    prompts = []
    for i, metadata in enumerate(frame_metadata):
        timestamp = metadata["timestamp"]
        zone_note = " (focus on red zone)" if metadata["has_zone"] else ""
        prompt = f"{base_prompt}\n1. What objects and people are visible?\n2. What activities or movements are happening?\n3. Describe the scene composition and context.\nFrame at {timestamp:.1f}s{zone_note}"
        prompts.append(prompt)

    try:
        # Analyze frames with LLaVA
        llava_analyses = await analyze_video_frames_with_llava(frames_for_analysis, prompts)
        
        print(f"✅ LLaVA analysis complete for {len(llava_analyses)} frames")

        # Generate comprehensive summary
        video_metadata = {
            "duration": duration,
            "fps": fps,
            "total_frames": total_frames,
            "analyzed_frames": len(frames_for_analysis),
            "has_zone_restriction": coords is not None
        }

        summary = await generate_video_summary_with_llava(llava_analyses, video_metadata)
        
        # Create embedding for the summary
        embedding = embedder.encode(summary)
        embedding_index.add(embedding[None, :])
        embedding_metadata.append({
            "video": video_filename,
            "summary": summary,
            "llava_analysis": llava_analyses,
            "duration": duration,
            "video_id": video_id,
            "coords": coords,
            "analysis_method": "llava_groq",
            "frames_analyzed": len(frames_for_analysis)
        })

        print("🎉 LLaVA video processing complete!")

        return {
            "video_id": video_id,
            "summary": {
                "total_frames_analyzed": len(frames_for_analysis),
                "duration_seconds": duration,
                "fps": fps,
                "analysis_method": "LLaVA via Groq",
                "zone_restricted": coords is not None
            },
            "natural_language_summary": summary,
            "llava_analysis": [
                {
                    "frame_number": frame_metadata[i]["frame_number"],
                    "timestamp": frame_metadata[i]["timestamp"],
                    "analysis": analysis
                }
                for i, analysis in enumerate(llava_analyses)
            ],
            "frames_file_name": video_id,  # Directory name where frames are stored
            "file_path": video_filename,
            "frames_dir": video_id,
        }

    except Exception as e:
        print(f"❌ Error during LLaVA analysis: {str(e)}")
        return {
            "video_id": video_id,
            "summary": f"Error during analysis: {str(e)}",
            "natural_language_summary": f"LLaVA analysis failed: {str(e)}",
            "llava_analysis": [],
            "frames_file_name": video_id,
            "file_path": video_filename,
            "frames_dir": video_id,
        }
