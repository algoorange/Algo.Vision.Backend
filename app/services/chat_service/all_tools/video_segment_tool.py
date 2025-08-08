from pymongo import MongoClient
import os
import json
import base64
from bson.binary import Binary

client = MongoClient(os.getenv("MONGO_URI"))
db = client["algo_compliance_db_2"]
collection = db["video_details"]
video_details_segment = db["video_details_segment"]

class VideoSegmentToolService:
    def __init__(self, request):
        self.request = request
        self.db = request.app.db if request and hasattr(request, 'app') and hasattr(request.app, 'db') else db

 ###########segmentation tools###########    

    async def get_video_segment_details(self, args):
        """
        Analyze segmented video data and return object tracking insights.

        Supports:
        - Unique or total object counts per segment
        - Count of objects in the first N seconds/minutes
        - Track-level frame, time, and position range
        """

        import json

        # Load arguments if passed as string
        if isinstance(args, str):
            args = json.loads(args)

        # Extract params with type safety
        def to_bool(val):
            return str(val).lower() == "true" if isinstance(val, str) else bool(val)

        video_id = args.get("video_id")
        segment_duration = float(args.get("segment_duration", 5.0))
        get_segment_object_counts = to_bool(args.get("get_segment_object_counts", False))
        get_total_object_count = to_bool(args.get("get_total_object_count", False))
        get_track_frame_range = to_bool(args.get("get_track_frame_range", False))
        get_track_time_range = to_bool(args.get("get_track_time_range", False))
        get_track_position_range = to_bool(args.get("get_track_position_range", False))
        get_count_by_type = to_bool(args.get("get_count_by_type", False))
        get_count_by_color_in_segment = to_bool(args.get("get_count_by_color_in_segment", False))
        get_number_of_segments = to_bool(args.get("get_number_of_segments", False))
       
        count_within_seconds = float(args.get("count_within_seconds", 0))
        time_range_start = args.get("time_range_start")
        time_range_end = args.get("time_range_end")
        last_n_seconds = args.get("last_n_seconds")
        track_id = args.get("track_id")

        if not video_id:
            return {"error": "video_id is required"}

        # Get all matching segments
        segments = list(video_details_segment.find({
            "video_id": video_id,
            "segment_duration": segment_duration
        }).sort("segment_index", 1))

        if not segments:
            return {"error": "No segments found for this video_id and duration"}

        result = {}
        all_track_ids = set()

        # === 1. Segment-wise object count ===
        if get_segment_object_counts:
            segment_counts = {}
            for segment in segments:
                index = segment.get("segment_index")
                objects = segment.get("summary", {}).get("objects", [])
                track_ids = {
                    obj["track_id"]
                    for obj in objects
                    if isinstance(obj, dict) and "track_id" in obj
                }
                segment_counts[f"segment_{index}"] = {
                    "unique_track_ids": list(track_ids),
                    "count": len(track_ids)
                }
                all_track_ids.update(track_ids)
            result["segment_object_counts"] = segment_counts

        # === 2. Total unique object count ===
        if get_total_object_count:
            for segment in segments:
                objects = segment.get("summary", {}).get("objects", [])
                track_ids = {
                    obj["track_id"]
                    for obj in objects
                    if isinstance(obj, dict) and "track_id" in obj
                }
                all_track_ids.update(track_ids)
            result["total_unique_objects"] = len(all_track_ids)
        #=== 3. Number of segments ===

        if get_number_of_segments:
            # Number of segments is the count of unique segment indices found
            segment_indices = set()
            for segment in segments:
                index = segment.get("segment_index")
                if index is not None:
                    segment_indices.add(index)
            result["number_of_segments"] = len(segment_indices)
            
        if get_count_by_color_in_segment:

            def _as_list(value):
                """Normalize a value to a lowercase list or return None if missing."""
                if value is None:
                    return None
                if isinstance(value, (list, tuple, set)):
                    return [str(x).lower() for x in value]
                return [str(value).lower()]

            colors = _as_list(args.get("color"))
            types = _as_list(args.get("object_type"))
            # Support both min_confidence and confidence (fallback)
            mc = args.get("min_confidence", args.get("confidence"))
            min_conf = float(mc) if mc is not None else None

            # Resolve video end once
            last_seg = segments[-1] if segments else None
            video_end = None
            if last_seg is not None:
                video_end = last_seg.get("summary", {}).get("end_time")
                if video_end is None:
                    video_end = (last_seg.get("segment_index", len(segments)-1) + 1) * segment_duration

            # Time window: first N, explicit [start,end], or last N
            video_starting_time, video_ending_time = None, None
            if args.get("count_within_seconds") is not None:
                video_starting_time = 0.0
                video_ending_time = float(args.get("count_within_seconds"))
            elif args.get("time_range_start") is not None or args.get("time_range_end") is not None:
                video_starting_time = float(args.get("time_range_start")) if args.get("time_range_start") is not None else 0.0
                video_ending_time = float(args.get("time_range_end")) if args.get("time_range_end") is not None else video_end
            elif args.get("last_n_seconds") is not None and video_end is not None:
                last_n = float(args.get("last_n_seconds"))
                video_starting_time = max(0.0, video_end - last_n)
                video_ending_time = video_end

            def _overlap(s, e):
                """
                Check if a segment's time range [s, e] overlaps with a given time window [ws, we].
                If the window is not specified (ws and we are None), always return True.
                """
                if video_starting_time is None and video_ending_time is None:
                    return True 
                # Ensure start and end times are valid
                s = 0.0 if s is None else float(s)
                e = s if e is None else float(e)
                # Check if the segment ends before the window starts or starts after the window ends
                if video_starting_time is not None and e < float(video_starting_time):
                    return False
                if video_ending_time is not None and s > float(video_ending_time):
                    return False
                # If no overlap is found, return True
                return True

            # Optional segment filtering
            seg_idx_arg = args.get("segment_index")
            if isinstance(seg_idx_arg, (list, tuple, set)):
                segment_index_set = {int(i) for i in seg_idx_arg}
            elif seg_idx_arg is not None:
                segment_index_set = {int(seg_idx_arg)}
            else:
                segment_index_set = None

            # Aggregate unique track_ids per color
            track_ids_by_color = {}
            for seg in segments:
                if segment_index_set is not None and seg.get("segment_index") not in segment_index_set:
                    continue
                for obj in seg.get("objects", []):  # use only per-segment objects as per DB structure
                    trk = obj.get("track_id")
                    if trk is None:
                        continue
                    col = str(obj.get("color", "")).lower()
                    if colors is not None and col not in colors:
                        continue
                    otype = str(obj.get("object_type", "unknown")).lower()
                    if types is not None and otype not in types:
                        continue
                    if min_conf is not None and float(obj.get("confidence", 0.0)) < min_conf:
                        continue
                    if not _overlap(obj.get("start_time"), obj.get("end_time")):
                        continue
                    track_ids_by_color.setdefault(col, set()).add(str(trk))

            result["count_by_color"] = {
                "window_start": video_starting_time,
                "window_end":   video_ending_time,
                "colors": colors,
                "object_types": types,
                "unique_track_ids_by_color": {c: sorted(list(s)) for c, s in track_ids_by_color.items()},
                "unique_count_by_color": {c: len(s) for c, s in track_ids_by_color.items()},
                "total_unique_objects": sum(len(s) for s in track_ids_by_color.values()),
            }
        

        # === 4. Count objects within N seconds (legacy) ===
        if count_within_seconds > 0:
            unique_track_ids_by_type = {}
            total_detections = 0
            for segment in segments:
                start_time = segment.get("summary", {}).get("start_time", 0)
                if start_time < count_within_seconds:
                    objects = segment.get("summary", {}).get("objects", [])
                    for obj in objects:
                        if isinstance(obj, dict) and "track_id" in obj:
                            obj_type = obj.get("object_type", "unknown")
                            track_id = obj["track_id"]
                            if obj_type not in unique_track_ids_by_type:
                                unique_track_ids_by_type[obj_type] = set()
                            unique_track_ids_by_type[obj_type].add(track_id)
                            total_detections += 1

            unique_count_by_type = {k: len(v) for k, v in unique_track_ids_by_type.items()}
            total_unique_objects = sum(unique_count_by_type.values())

            result["objects_in_first_n_seconds"] = {
                "within_seconds": count_within_seconds,
                "unique_track_ids_by_type": {k: list(v) for k, v in unique_track_ids_by_type.items()},
                "unique_count_by_type": unique_count_by_type,
                "total_unique_objects": total_unique_objects,
                "total_detections": total_detections
            }

        # === 5. Flexible time-based object queries ===
        # Handles: 'last N seconds', 'between A and B seconds', 'after X seconds'
        time_query_result = None
        # Compute video end time if needed
        video_end_time = None
        if segments:
            last_segment = segments[-1]
            video_end_time = last_segment.get("summary", {}).get("end_time")
            if video_end_time is None:
                # fallback: try segment duration * number of segments
                video_end_time = (last_segment.get("segment_index", len(segments)-1) + 1) * segment_duration
        # Determine time window
        video_starting_time = None
        video_ending_time = None
        if last_n_seconds is not None:
            try:
                last_n_seconds = float(last_n_seconds)
                if video_end_time is not None:
                    video_starting_time = max(0, video_end_time - last_n_seconds)
                    video_ending_time = video_end_time
            except Exception:
                pass
        elif time_range_start is not None or time_range_end is not None:
            try:
                video_starting_time = float(time_range_start) if time_range_start is not None else 0
                video_ending_time = float(time_range_end) if time_range_end is not None else video_end_time
            except Exception:
                pass
        # Only run if a valid window is set
        if video_starting_time is not None and video_ending_time is not None and video_ending_time > video_starting_time:
            unique_track_ids_by_type = {}
            total_detections = 0
            object_presence = {}
            for segment in segments:
                objects = segment.get("summary", {}).get("objects", [])
                for obj in objects:
                    if isinstance(obj, dict) and "track_id" in obj:
                        obj_type = obj.get("object_type", "unknown")
                        track_id = obj["track_id"]
                        # Use object's start_time and end_time for presence
                        o_start = obj.get("start_time", 0)
                        o_end = obj.get("end_time", 0)
                        # Check if object was present at any point in the window
                        if o_end >= video_starting_time and o_start <= video_ending_time:
                            if obj_type not in unique_track_ids_by_type:
                                unique_track_ids_by_type[obj_type] = set()
                            unique_track_ids_by_type[obj_type].add(track_id)
                            total_detections += 1
                            # Track presence window for each object
                            if track_id not in object_presence:
                                object_presence[track_id] = {
                                    "object_type": obj_type,
                                    "start_time": o_start,
                                    "end_time": o_end
                                }
            unique_count_by_type = {k: len(v) for k, v in unique_track_ids_by_type.items()}
            total_unique_objects = sum(unique_count_by_type.values())
            time_query_result = {
                "window_start": video_starting_time,
                "window_end": video_ending_time,
                "unique_track_ids_by_type": {k: list(v) for k, v in unique_track_ids_by_type.items()},
                "unique_count_by_type": unique_count_by_type,
                "total_unique_objects": total_unique_objects,
                "object_presence": object_presence,
                "total_detections": total_detections
            }
            result["objects_in_time_window"] = time_query_result

        # === 4. Track ID - Frame range ===
        if get_track_frame_range and track_id:
            min_frame, max_frame = float("inf"), float("-inf")
            found = False
            for segment in segments:
                for obj in segment.get("summary", {}).get("objects", []):
                    if isinstance(obj, dict) and obj.get("track_id") == str(track_id):
                        found = True
                        min_frame = min(min_frame, obj.get("start_frame", float("inf")))
                        max_frame = max(max_frame, obj.get("end_frame", float("-inf")))
            if found:
                result["track_id_frame_range"] = {
                    "track_id": track_id,
                    "start_frame": min_frame,
                    "end_frame": max_frame
                }
            else:
                result["track_id_frame_range"] = {"error": "Track ID not found", "track_id": track_id}

        # === 5. Track ID - Time range ===
        if get_track_time_range and track_id:
            min_time, max_time = float("inf"), float("-inf")
            found = False
            for segment in segments:
                for obj in segment.get("summary", {}).get("objects", []):
                    if isinstance(obj, dict) and obj.get("track_id") == str(track_id):
                        found = True
                        min_time = min(min_time, obj.get("start_time", float("inf")))
                        max_time = max(max_time, obj.get("end_time", float("-inf")))
            if found:
                result["track_id_time_range"] = {
                    "track_id": track_id,
                    "start_time": min_time,
                    "end_time": max_time
                }
            else:
                result["track_id_time_range"] = {"error": "Track ID not found", "track_id": track_id}      

        # === 6. Track ID - Position range ===
        if get_track_position_range and track_id:
            start_pos, end_pos = None, None
            found = False
            for segment in segments:
                for obj in segment.get("summary", {}).get("objects", []):
                    if isinstance(obj, dict) and obj.get("track_id") == str(track_id):
                        found = True
                        start_pos = obj.get("start_position", start_pos)
                        end_pos = obj.get("end_position", end_pos)
            if found:
                result["track_id_position_range"] = {
                    "track_id": track_id,
                    "start_position": start_pos,
                    "end_position": end_pos
                }
            else:
                result["track_id_position_range"] = {"error": "Track ID not found", "track_id": track_id}

        # === Default fallback ===
        if not result:
            result["message"] = (
                "No valid flags provided. Use flags like "
                "`get_segment_object_counts`, `get_total_object_count`, `count_within_seconds`, or track-level flags, "
            )

        return {"result": result, "reformat": True}
