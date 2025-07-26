from deep_sort_realtime.deepsort_tracker import DeepSort
import math

# Initialize trackers
# deepsort_tracker = DeepSort(
#     max_age=30,
#     n_init=3,
#     embedder="mobilenet",  # 👈 use mobilenet for appearance features
#     half=True,             # use FP16 if GPU supports
#     max_iou_distance=0.7
# )

deepsort_tracker = DeepSort(
    max_age=50,              # Increase from 30 - keep tracks longer
    n_init=2,                # Decrease from 3 - confirm tracks faster  
    embedder="mobilenet",    
    half=True,
    max_iou_distance=0.7,   
    max_cosine_distance=0.3, # Add this - stricter appearance matching
    nn_budget=100,           # Add this - limit feature budget
    nms_max_overlap=0.9      # Add this - reduce NMS aggressiveness
)

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
        print('detection_map', detection_map)

    # Update tracker
    tracks = deepsort_tracker.update_tracks(formatted, frame=frame)

    # Prepare output: for each active track, report its info
    output_tracks = []
    for track_object in tracks:
        print(f"Track ID: {track_object.track_id}, Confirmed: {track_object.is_confirmed()}, State: {getattr(track_object, 'state', None)}")  # Debug info
        if not track_object.is_confirmed():
            continue
        bbox = track_object.to_ltrb()  # [x1, y1, x2, y2]
        # Attempt to get label/confidence from detection_map by bbox
        det_label = getattr(track_object, 'det_class', None)
        det_conf = getattr(track_object, 'det_conf', None)
        # If tracker supports custom fields, update them
        if hasattr(track_object, 'det_class') and hasattr(track_object, 'det_conf'):
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
        # Ensure det_conf is a float and not None
        if det_conf is None:
            det_conf = 0.0
        if det_conf > 0:
            print("det_label", det_label)
            print("det_conf", det_conf)
            output_tracks.append({
                "object_type": det_label,
                "position": {
                    "x": int(bbox[0]),
                    "y": int(bbox[1]),
                    "x1": int(bbox[2]),
                    "y1": int(bbox[3])
                },
                "track_id": track_object.track_id,
                "confidence": float(det_conf)
            })
    return output_tracks



def calculate_angle(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))









