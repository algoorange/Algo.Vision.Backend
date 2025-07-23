try:
    from deep_sort_realtime import DeepSort
    # Enhanced configuration for CCTV tracking
    tracker = DeepSort(
        max_age=100,           # Keep tracks longer for CCTV (was 50)
        n_init=3,              # Confirm tracks quickly
        max_iou_distance=0.8,  # Higher threshold for better association
        max_cosine_distance=0.3,  # Feature-based association
        nn_budget=100,         # Memory for appearance features
        embedder="torchreid",  # Better re-identification
        half=True,             # Use FP16 for speed
        bgr=True,              # OpenCV uses BGR
        embedder_gpu=True,     # Use GPU if available
        embedder_model_name="osnet_x0_25",  # Lightweight ReID model
        polygon=False,         # Use bounding boxes
        today=None
    )
    TRACKER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DeepSort tracker not available: {e}")
    print("Falling back to simple tracking...")
    tracker = None
    TRACKER_AVAILABLE = False

import math
import numpy as np
from collections import defaultdict, deque

# Track trajectory storage for movement analysis
track_trajectories = defaultdict(lambda: deque(maxlen=30))  # Store last 30 positions
track_metadata = defaultdict(dict)  # Store additional track info

def track_objects(frame, detections):
    """
    Enhanced object tracking for CCTV footage with trajectory analysis.
    
    Tracks all detected objects with bounding boxes and returns a list of active tracks 
    with their type, position, track_id, confidence, and movement information.
    
    Each output is a dict:
    {
        'object_type': <class label>,
        'position': {'x': x1, 'y': y1, 'x1': x2, 'y1': y2},
        'track_id': <track_id>,
        'confidence': <confidence>,
        'trajectory': [list of recent positions],
        'velocity': {'speed': float, 'direction': float},
        'is_stationary': bool,
        'time_visible': int  # frames since first detection
    }
    """
    if not TRACKER_AVAILABLE or tracker is None:
        # Enhanced fallback with basic trajectory tracking
        return _simple_tracking_fallback(detections)

    # Format detections for DeepSort
    formatted_detections = []
    detection_map = {}
    
    for det in detections:
        x, y, w, h = det["bbox"]
        x1, y1, x2, y2 = x, y, x + w, y + h
        
        # Ensure valid bounding box
        x1, y1, x2, y2 = max(0, x1), max(0, y1), max(x1+1, x2), max(y1+1, y2)
        
        formatted_detections.append((
            [x1, y1, x2, y2],  # bbox
            det["confidence"],  # confidence
            det["label"]       # class
        ))
        
        detection_map[(x1, y1, x2, y2)] = {
            "label": det["label"],
            "confidence": det["confidence"]
        }

    # Update tracker with new detections
    tracks = tracker.update_tracks(formatted_detections, frame=frame)

    # Process tracking results
    output_tracks = []
    current_frame_tracks = set()
    
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        current_frame_tracks.add(track_id)
        bbox = track.to_ltrb()  # [x1, y1, x2, y2]
        
        # Get detection information
        det_label = getattr(track, 'det_class', None)
        det_conf = getattr(track, 'det_conf', None)
        
        # Fallback to detection map if tracker doesn't have class info
        if not det_label or not det_conf:
            bbox_key = tuple(map(int, bbox))
            det = detection_map.get(bbox_key, {})
            det_label = det.get("label", "unknown")
            det_conf = det.get("confidence", 0.0)
        
        # Calculate center point for trajectory
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        current_position = (center_x, center_y)
        
        # Update trajectory
        track_trajectories[track_id].append(current_position)
        
        # Update metadata
        if track_id not in track_metadata:
            track_metadata[track_id] = {
                'first_seen': 0,
                'object_type': det_label
            }
        track_metadata[track_id]['time_visible'] = track_metadata[track_id].get('time_visible', 0) + 1
        
        # Calculate movement metrics
        trajectory = list(track_trajectories[track_id])
        velocity = calculate_velocity(trajectory)
        is_stationary = is_object_stationary(trajectory)
        
        output_tracks.append({
            "object_type": det_label,
            "position": {
                "x": int(bbox[0]),
                "y": int(bbox[1]),
                "x1": int(bbox[2]),
                "y1": int(bbox[3])
            },
            "track_id": track_id,
            "confidence": float(det_conf) if det_conf is not None else 0.0,
            "trajectory": trajectory,
            "velocity": velocity,
            "is_stationary": is_stationary,
            "time_visible": track_metadata[track_id].get('time_visible', 1),
            "center": current_position
        })
    
    # Clean up old trajectories for tracks that are no longer active
    _cleanup_old_tracks(current_frame_tracks)
    
    return output_tracks

def _simple_tracking_fallback(detections):
    """Enhanced fallback tracking with basic trajectory support"""
    static_track_id = getattr(_simple_tracking_fallback, 'next_id', 0)
    output_tracks = []
    
    for i, det in enumerate(detections):
        x, y, w, h = det["bbox"]
        x1, y1, x2, y2 = x, y, x + w, y + h
        track_id = static_track_id + i
        
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        current_position = (center_x, center_y)
        
        output_tracks.append({
            "object_type": det["label"],
            "position": {
                "x": int(x1),
                "y": int(y1),
                "x1": int(x2),
                "y1": int(y2)
            },
            "track_id": track_id,
            "confidence": float(det["confidence"]),
            "trajectory": [current_position],
            "velocity": {"speed": 0.0, "direction": 0.0},
            "is_stationary": True,
            "time_visible": 1,
            "center": current_position
        })
    
    _simple_tracking_fallback.next_id = static_track_id + len(detections)
    return output_tracks

def calculate_velocity(trajectory):
    """Calculate velocity (speed and direction) from trajectory"""
    if len(trajectory) < 2:
        return {"speed": 0.0, "direction": 0.0}
    
    # Use last few points for velocity calculation
    recent_points = trajectory[-min(5, len(trajectory)):]
    
    if len(recent_points) < 2:
        return {"speed": 0.0, "direction": 0.0}
    
    # Calculate average velocity over recent points
    total_distance = 0.0
    total_angle = 0.0
    
    for i in range(1, len(recent_points)):
        p1, p2 = recent_points[i-1], recent_points[i]
        
        # Distance (speed)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        distance = math.sqrt(dx*dx + dy*dy)
        total_distance += distance
        
        # Direction
        if distance > 0:  # Avoid division by zero
            angle = math.degrees(math.atan2(dy, dx))
            total_angle += angle
    
    avg_speed = total_distance / (len(recent_points) - 1)
    avg_direction = total_angle / (len(recent_points) - 1) if len(recent_points) > 1 else 0.0
    
    return {
        "speed": round(avg_speed, 2),
        "direction": round(avg_direction, 2)
    }

def is_object_stationary(trajectory, threshold=10.0):
    """Determine if object is stationary based on trajectory"""
    if len(trajectory) < 3:
        return True
    
    # Calculate total movement over trajectory
    total_movement = 0.0
    for i in range(1, len(trajectory)):
        p1, p2 = trajectory[i-1], trajectory[i]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        total_movement += math.sqrt(dx*dx + dy*dy)
    
    # Object is stationary if average movement per frame is below threshold
    avg_movement = total_movement / (len(trajectory) - 1)
    return avg_movement < threshold

def get_movement_direction(dx, dy):
    """Get cardinal direction of movement"""
    if abs(dx) < 1 and abs(dy) < 1:
        return "stationary"
    
    angle = math.degrees(math.atan2(dy, dx))
    
    # Normalize angle to 0-360
    if angle < 0:
        angle += 360
    
    # Convert to cardinal directions
    if 337.5 <= angle or angle < 22.5:
        return "east"
    elif 22.5 <= angle < 67.5:
        return "northeast"
    elif 67.5 <= angle < 112.5:
        return "north"
    elif 112.5 <= angle < 157.5:
        return "northwest"
    elif 157.5 <= angle < 202.5:
        return "west"
    elif 202.5 <= angle < 247.5:
        return "southwest"
    elif 247.5 <= angle < 292.5:
        return "south"
    else:  # 292.5 <= angle < 337.5
        return "southeast"

def _cleanup_old_tracks(current_tracks):
    """Remove trajectories for tracks that are no longer active"""
    # Remove tracks not seen for a while
    tracks_to_remove = []
    for track_id in track_trajectories:
        if track_id not in current_tracks:
            tracks_to_remove.append(track_id)
    
    # Only remove if we have too many old tracks (memory management)
    if len(tracks_to_remove) > 100:  # Keep some history
        for track_id in tracks_to_remove[:50]:  # Remove oldest 50
            del track_trajectories[track_id]
            if track_id in track_metadata:
                del track_metadata[track_id]

def calculate_angle(p1, p2):
    """Calculate angle between two points"""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))

def get_track_statistics():
    """Get overall tracking statistics"""
    return {
        "active_tracks": len(track_trajectories),
        "total_trajectories": sum(len(traj) for traj in track_trajectories.values()),
        "track_metadata": dict(track_metadata)
    }
