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

def summarize_segment(segment: List[Dict[str, Any]], video_id: str) -> Dict[str, Any]:
    """
    Summarizes a segment of tracking data (frames_data style).
    Returns a dict summary restricted to required fields only.
    Includes unique object counts by track_id for each object type and a filtered list of objects.
    """
    # Aggregate per unique track_id within the segment
    # We will assign a single dominant (majority) object_type per track_id to ensure
    # that the sum(object_counts.values()) == len(objects)
    track_agg: Dict[str, Dict[str, Any]] = {}

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



    # Only keep these fields for each object in the segment
    allowed_object_keys = {
        "track_id",
        "object_type",
        "confidence",
        "frame_time",
        "color",
        "start_time",
        "end_time",
        "start_frame",
        "end_frame",
        "start_position",
        "end_position",
    }

    for frame in segment:
        for obj in frame.get('objects', []):
            obj_type = obj.get('object_type', 'unknown')
            raw_tid = obj.get('track_id')
            if raw_tid is None:
                # Skip untracked objects to avoid inflating counts
                continue
            track_id = str(raw_tid)  # normalize type to string for stable keys

            # Initialize aggregation entry
            if track_id not in track_agg:
                filtered_obj = {k: obj[k] for k in allowed_object_keys if k in obj}
                track_agg[track_id] = {
                    "first_obj": filtered_obj,
                    "type_counts": {},  # object_type -> count within this segment
                }
            # Increment type count for this track within the segment
            tc = track_agg[track_id]["type_counts"]
            tc[obj_type] = tc.get(obj_type, 0) + 1

    # Decide dominant type per track and build final counts and objects
    object_counts: Dict[str, int] = {}
    objects: List[Dict[str, Any]] = []
    for tid, data in track_agg.items():
        type_counts = data["type_counts"]
        if not type_counts:
            # Should not happen, but guard anyway
            continue
        # Pick majority class; tie-breaker: lexical order to keep deterministic
        dominant_type = sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        object_counts[dominant_type] = object_counts.get(dominant_type, 0) + 1
        obj = dict(data["first_obj"])  # copy
        obj["object_type"] = dominant_type
        objects.append(obj)

    summary = {
        "start_time": segment[0]["frame_time"],
        "end_time": segment[-1]["frame_time"],
        "object_counts": object_counts,        # unique objects by track_id
        "frame_count": len(segment),
        "objects": objects,  # Filtered list of detected objects with required details only
    }
    return summary
