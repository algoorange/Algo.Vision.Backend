from typing import List, Dict, Any

def segment_tracking_data(tracking_data: List[Dict[str, Any]], segment_duration: float) -> List[List[Dict[str, Any]]]:
    """
    Splits tracking data into segments of given duration (in seconds).
    tracking_data: list of dicts with keys including 'frame_time' or 'timestamp'
    Returns: list of segments, each a list of tracking records
    """
    if not tracking_data:
        return []
    segments = []
    current_segment = []
    segment_start = tracking_data[0].get('frame_time', tracking_data[0].get('timestamp', 0))
    for record in tracking_data:
        t = record.get('frame_time', record.get('timestamp', 0))
        if t - segment_start < segment_duration:
            current_segment.append(record)
        else:
            segments.append(current_segment)
            current_segment = [record]
            segment_start = t
    if current_segment:
        segments.append(current_segment)
    return segments

def summarize_segment(segment: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarizes a segment of tracking data (frames_data style).
    Returns a dict summary including all object details.
    Includes both total detection counts and unique object counts by track_id for each object type.
    """
    objects = []
    unique_objects_by_type = {}  # unique track_id per type
    detection_counts = {}  # total detections per type

#calculation of unique objects by track_id

# car_1 present in frames 0–8 (so in both segment 1 and segment 2)
# car_2 present in frames 0–4 (only segment 1)
# car_3 present in frames 5–9 (only segment 2)
# Segment 1 (0–5s): car_1, car_2
# Segment 2 (5–10s): car_1, car_3

# Segment 1 unique cars: 2 (car_1, car_2)
# Segment 2 unique cars: 2 (car_1, car_3)
# 10s segment unique cars: 3 (car_1, car_2, car_3)
# Sum of 5s segments: 2+2 = 4
# Actual 10s segment: 3



    for frame in segment:
        frame_time = frame.get("frame_time")
        for obj in frame.get('objects', []):
            obj_type = obj.get('object_type', 'unknown')
            track_id = obj.get('track_id')
            # Count total detections
            detection_counts[obj_type] = detection_counts.get(obj_type, 0) + 1
            # Count unique objects by track_id
            if obj_type not in unique_objects_by_type:
                unique_objects_by_type[obj_type] = set()
            if track_id is not None:
                unique_objects_by_type[obj_type].add(track_id)
            # Collect object details if needed
            objects.append(obj)

    object_counts = {k: len(v) for k, v in unique_objects_by_type.items()}

    summary = {
        "start_time": segment[0]["frame_time"],
        "end_time": segment[-1]["frame_time"],
        "object_counts": object_counts,        # unique objects by track_id
        "detection_counts": detection_counts,  # total detections per type
        "frame_count": len(segment),
        "objects": objects,  # List of all detected objects with details
    }
    return summary
