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
    """
    object_types = {}
    total_objects = 0
    objects = []

    for frame in segment:
        frame_time = frame.get("frame_time")
        for obj in frame.get('objects', []):
            obj_type = obj.get('object_type', 'unknown')
            object_types[obj_type] = object_types.get(obj_type, 0) + 1
            total_objects += 1
            objects.append({
                "frame_time": frame_time,
                "object_type": obj_type,
                "confidence": obj.get("confidence"),
                "position": obj.get("position"),
                "bbox": obj.get("bbox"),
                "track_id": obj.get("track_id"),
            })

    summary = {
        "start_time": segment[0]["frame_time"],
        "end_time": segment[-1]["frame_time"],
        "object_counts": object_types,
        "total_objects": total_objects,
        "frame_count": len(segment),
        "objects": objects,  # List of all detected objects with details
    }
    return summary

