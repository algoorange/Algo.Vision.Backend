try:
    from deep_sort_realtime import DeepSort
    tracker = DeepSort(max_age=50, n_init=3, max_iou_distance=0.7)
    TRACKER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DeepSort tracker not available: {e}")
    print("Falling back to simple tracking...")
    tracker = None
    TRACKER_AVAILABLE = False

import math

def track_objects(frame, detections):
    """
    Tracks all detected objects with bounding boxes and returns a list of active tracks with their type, position, track_id, and confidence.
    Each output is a dict:
    {
        'object_type': <class label>,
        'position': {'x': x1, 'y': y1, 'x1': x2, 'y1': y2},
        'track_id': <track_id>,
        'confidence': <confidence>
    }
    """
    if not TRACKER_AVAILABLE or tracker is None:
        # Simple fallback: assign sequential IDs without actual tracking
        output_tracks = []
        for i, det in enumerate(detections):
            x, y, w, h = det["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            output_tracks.append({
                "object_type": det["label"],
                "position": {
                    "x": int(x1),
                    "y": int(y1),
                    "x1": int(x2),
                    "y1": int(y2)
                },
                "track_id": i,  # Simple sequential ID
                "confidence": float(det["confidence"])
            })
        return output_tracks

    # Normal tracking with DeepSort
    formatted = []
    detection_map = {}
    for det in detections:
        x, y, w, h = det["bbox"]
        x1, y1, x2, y2 = x, y, x + w, y + h
        formatted.append((
            [x1, y1, x2, y2],  # [x1, y1, x2, y2]
            det["confidence"],
            det["label"]
        ))
        detection_map[(x1, y1, x2, y2)] = {
            "label": det["label"],
            "confidence": det["confidence"]
        }

    # Update tracker
    tracks = tracker.update_tracks(formatted, frame=frame)

    # Prepare output: for each active track, report its info
    output_tracks = []
    for tracked_data in tracks:
        if not tracked_data.is_confirmed():
            continue
        bbox = tracked_data.to_ltrb()  # [x1, y1, x2, y2]
        # Attempt to get label/confidence from detection_map by bbox
        det_label = getattr(tracked_data, 'det_class', None)
        det_conf = getattr(tracked_data, 'det_conf', None)
        # If tracker supports custom fields, update them
        if hasattr(tracked_data, 'det_class') and hasattr(tracked_data, 'det_conf'):
            pass  # already set by DeepSort
        else:
            # Try to match with detection map (by bbox)
            det = detection_map.get(tuple(map(int, bbox)), None)
            if det:
                det_label = det["label"]
                det_conf = det["confidence"]
            else:
                det_label = det_label or "unknown"
                det_conf = det_conf or 0.0
        output_tracks.append({
            "object_type": det_label,
            "position": {
                "x": int(bbox[0]),
                "y": int(bbox[1]),
                "x1": int(bbox[2]),
                "y1": int(bbox[3])
            },
            "track_id": tracked_data.track_id,
            "confidence": float(det_conf) if det_conf is not None else 0.0
        })
    return output_tracks

def calculate_angle(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))
