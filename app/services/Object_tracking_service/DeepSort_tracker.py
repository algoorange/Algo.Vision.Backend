from deep_sort_realtime.deepsort_tracker import DeepSort
import math

deepsort_tracker = DeepSort(
    max_age=50,
    n_init=2,
    embedder="mobilenet",             # Valid embedder option
    half=False,                       # Use full precision for CPU
    max_iou_distance=0.7,
    max_cosine_distance=0.2,          # Stricter matching
    nn_budget=300,                    # Keep more history for matching
    nms_max_overlap=0.9
)


# Color detection function moved to centralized utility: app.utils.color_detection

def track_objects(frame, detections):
    """
    Tracks all detected objects with bounding boxes and returns a list of active tracks with their type, position, track_id, and confidence.
    Each output is a dict:
    {
        'object_type': <class label>,
        'position': {'x': x1, 'y': y1, 'x1': x2, 'y1': y2},
        'track_id': <track_id>,
        'confidence': <confidence>,
        'color': <detected_color>
    }
    """
    formatted = []
    detection_colors = []  # Store colors by detection index
    detection_labels = []  # Store labels by detection index
    
    for i, det in enumerate(detections):
        x, y, w, h = det["bbox"]
        x1, y1, x2, y2 = x, y, x + w, y + h
        formatted.append((
            [x1, y1, x2, y2],  # [x1, y1, x2, y2]
            det["confidence"],
            det["label"]
        ))
        # Store color and label by detection index
        color = det.get("color", "unknown")
        detection_colors.append(color)
        detection_labels.append(det["label"])
        print(f"[DEBUG] Detection {i}: Label={det['label']}, Color={color}, BBox={det['bbox']}")

    # Update tracker
    tracks = deepsort_tracker.update_tracks(formatted, frame=frame)

    # Prepare output: for each active track, report its info
    output_tracks = []
    for track_object in tracks:
        if not track_object.is_confirmed():
            continue
            
        bbox = track_object.to_ltrb()  # [x1, y1, x2, y2]
        
        # Get label/confidence from tracker attributes
        det_label = getattr(track_object, 'det_class', None)
        det_conf = getattr(track_object, 'det_conf', None)
        
        # Find the best matching detection for color propagation
        def compute_iou(box1, box2):
            """Compute IoU between two boxes [x1,y1,x2,y2]"""
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
                
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        # Find best matching detection for color using multiple strategies
        print(f"[DEBUG] Track {track_object.track_id} BBox: {bbox}")
        
        def get_bbox_center(bbox_coords):
            """Get center point of bounding box"""
            if len(bbox_coords) == 4:  # [x1, y1, x2, y2]
                return ((bbox_coords[0] + bbox_coords[2]) / 2, (bbox_coords[1] + bbox_coords[3]) / 2)
            return (0, 0)
        
        def compute_distance(center1, center2):
            """Compute Euclidean distance between two centers"""
            return ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
        
        track_center = get_bbox_center(bbox)
        best_iou = 0.0
        best_distance = float('inf')
        best_color = "unknown"
        best_detection_idx = -1
        
        # Strategy 1: Try IoU matching first
        for i, det in enumerate(detections):
            det_x, det_y, det_w, det_h = det["bbox"]
            det_bbox = [det_x, det_y, det_x + det_w, det_y + det_h]
            
            iou = compute_iou(bbox, det_bbox)
            print(f"[DEBUG] Track {track_object.track_id} vs Detection {i}: IoU={iou:.3f}, Det_Color={detection_colors[i]}")
            
            if iou > best_iou and iou > 0.05:  # Very low threshold
                best_iou = iou
                best_color = detection_colors[i]
                best_detection_idx = i
        
        # Strategy 2: If IoU fails, use center distance + label matching
        if best_color == "unknown" and det_label != "unknown":
            print(f"[DEBUG] IoU matching failed, trying distance-based matching for label: {det_label}")
            
            for i, det in enumerate(detections):
                if det["label"] == det_label:  # Same object type
                    det_x, det_y, det_w, det_h = det["bbox"]
                    det_center = get_bbox_center([det_x, det_y, det_x + det_w, det_y + det_h])
                    distance = compute_distance(track_center, det_center)
                    
                    print(f"[DEBUG] Track {track_object.track_id} center {track_center} vs Detection {i} center {det_center}: Distance={distance:.1f}")
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_color = detection_colors[i]
                        best_detection_idx = i
        
        # Strategy 3: If still no match, use the first detection with same label
        if best_color == "unknown" and det_label != "unknown":
            print(f"[DEBUG] Distance matching failed, using first detection with same label: {det_label}")
            for i, det in enumerate(detections):
                if det["label"] == det_label:
                    best_color = detection_colors[i]
                    best_detection_idx = i
                    break
        
        print(f"[DEBUG] Track {track_object.track_id}: Final - IoU={best_iou:.3f}, Distance={best_distance:.1f}, Color={best_color}, Detection Index={best_detection_idx}")
        det_color = best_color
        
        # Fallback for label/confidence if not available from tracker
        if not det_label:
            det_label = "unknown"
        if not det_conf:
            det_conf = 0.0
        
        # Ensure det_conf is a float and not None
        if det_conf is None:
            det_conf = 0.0
            
        if det_conf > 0:
            print(f"Track ID: {track_object.track_id}, Label: {det_label}, Confidence: {det_conf:.2f}, Color: {det_color}")
            
            output_tracks.append({
                "object_type": det_label,
                "position": {
                    "x": int(bbox[0]),
                    "y": int(bbox[1]),
                    "x1": int(bbox[2]),
                    "y1": int(bbox[3])
                },
                "track_id": track_object.track_id,
                "confidence": float(det_conf),
                "color": det_color
            })
    return output_tracks



def calculate_angle(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))









