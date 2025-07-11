from deep_sort_realtime.deepsort_tracker import DeepSort

import math

tracker = DeepSort(max_age=30)

def track_objects(frame, detections):
    """
    Tracks only YOLO-detected objects with bounding boxes.
    Skips crack detections (no bounding boxes).
    """
    formatted = []
    for det in detections:
        if det["bbox"] is None:
            # Skip cracks (U-Net does not produce bounding boxes)
            continue

        x, y, w, h = det["bbox"]
        formatted.append((
            [x, y, x + w, y + h],  # Convert bbox to [x1, y1, x2, y2]
            det["confidence"],
            det["label"]
        ))

    # Update tracker
    return tracker.update_tracks(formatted, frame=frame)
    
def calculate_angle(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))
