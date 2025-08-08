from pymongo import MongoClient
import os
import json
import base64
from bson.binary import Binary

client = MongoClient(os.getenv("MONGO_URI"))
db = client["algo_compliance_db_2"]
collection = db["video_details"]
video_details_segment = db["video_details_segment"]

class VideoToolService:
    def __init__(self, request):
        self.request = request
        self.db = request.app.db if request and hasattr(request, 'app') and hasattr(request.app, 'db') else db


    async def get_all_object_details(self, args):
        """
        LLM Tool: Returns object details for a video with flags to control what kind of data is returned.

        Parameters:
        - video_id (str): ID of the video.
        - get_count (bool): If True, return count of unique objects.
        - get_unique_objects (bool): If True, return list of unique objects by track_id.
        - get_confidences (bool): If True, return confidence values.
        - get_object_types (bool): If True, return object types.
        - frame_number (int, optional): If provided, filter results to this frame only.
        - track_id (int, optional): Used if user asks about specific object.
        - get_frame_object_count (bool): If True, return object count in a specific frame.
        - get_frame_track_ids (bool): If True, return track IDs only for a specific frame.
        
        Returns:
        Dict based on requested flags.
        """

        import json

        if isinstance(args, str):
            args = json.loads(args)

        video_id = args.get("video_id")
        get_count = args.get("get_count", False)
        get_unique = args.get("get_unique_objects", False)
        get_confidences = args.get("get_confidences", False)
        get_object_types = args.get("get_object_types", False)
        get_frame_object_count = args.get("get_frame_object_count", False)
        get_position_by_frame_and_track = args.get("get_position_by_frame_and_track", False)
        get_object_color = args.get("get_object_color", False)
        get_frame_track_ids = args.get("get_frame_track_ids", False)  # ✅ NEW FLAG
        frame_number = args.get("frame_number")
        track_id = args.get("track_id")

        if not video_id:
            return {"error": "Missing video_id"}

        # Fetch video data
        video_doc = collection.find_one({"video_id": video_id})
        if not video_doc or "frames" not in video_doc:
            return {"result": []}

        # Flatten all objects
        all_objects = []
        for frame in video_doc["frames"]:
            for obj in frame.get("objects", []):
                obj["frame_number"] = frame.get("frame_number")  # Attach frame number for reference
                all_objects.append(obj)

        results = {}

        # ✅ Track IDs in a specific frame
        if get_frame_track_ids and frame_number is not None:
            frame_track_ids = {
                obj["track_id"]
                for obj in all_objects
                if obj.get("frame_number") == frame_number and "track_id" in obj
            }
            results["frame_number"] = frame_number
            results["track_ids"] = list(frame_track_ids)

        # ✅ General count of unique track IDs
        if get_count:
            unique_ids = {obj.get("track_id") for obj in all_objects if "track_id" in obj}
            results["count"] = len(unique_ids)

        # ✅ Unique object list by track ID
        if get_unique:
            seen = {}
            for obj in all_objects:
                tid = obj.get("track_id")
                if tid not in seen:
                    seen[tid] = {
                        "object_type": obj.get("object_type"),
                        "track_id": tid,
                        "confidence": obj.get("confidence"),
                        "frame_number": obj.get("frame_number")
                    }
            results["unique_objects"] = list(seen.values())

        # ✅ Confidence(s)
        if get_confidences:
            if track_id is not None:
                objects = [obj for obj in all_objects if obj.get("track_id") == track_id]
                confidences = list({obj.get("confidence") for obj in objects if "confidence" in obj})
                results["track_id"] = track_id
                results["confidences"] = confidences
            else:
                confidences = list({obj.get("confidence") for obj in all_objects if "confidence" in obj})
                results["confidences"] = confidences

        # ✅ Object count in specific frame
        if get_frame_object_count and frame_number is not None:
            unique_ids = set()
            for obj in all_objects:
                if obj.get("frame_number") == frame_number:
                    tid = obj.get("track_id")
                    if tid is not None:
                        unique_ids.add(tid)
            results["frame_number"] = frame_number
            results["object_count"] = len(unique_ids)

        if get_position_by_frame_and_track and track_id is not None and frame_number is not None:       
            matching_objects = [
                obj for obj in all_objects
                if str(obj.get("track_id")) == str(track_id) and int(obj.get("frame_number")) == int(frame_number)
            ]

            positions = []
            for obj in matching_objects:
                pos_info = {
                    "frame_number": obj.get("frame_number"),
                    "track_id": obj.get("track_id"),
                }
                if "position" in obj:
                    pos_info["position"] = obj["position"]
                if "bbox" in obj:
                    pos_info["bbox"] = obj["bbox"]
                if "start_position" in obj:
                    pos_info["start_position"] = obj["start_position"]
                if "end_position" in obj:
                    pos_info["end_position"] = obj["end_position"]
                positions.append(pos_info)

            results["positions"] = positions

        if get_object_color:
            if track_id is not None and frame_number is not None:
                # Both track_id and frame_number provided: filter by both
                objects = [
                    obj for obj in all_objects
                    if str(obj.get("track_id")) == str(track_id) and int(obj.get("frame_number")) == int(frame_number)
                ]
                colors = list({obj.get("color") for obj in objects if "color" in obj})
                results["track_id"] = track_id
                results["frame_number"] = frame_number
                results["color"] = colors
            elif track_id is not None:
                # Only track_id provided
                objects = [obj for obj in all_objects if str(obj.get("track_id")) == str(track_id)]
                colors = list({obj.get("color") for obj in objects if "color" in obj})
                results["track_id"] = track_id
                results["color"] = colors
            elif frame_number is not None:
                # Only frame_number provided
                objects = [obj for obj in all_objects if int(obj.get("frame_number")) == int(frame_number)]
                # Return all unique (object_type, track_id, color) tuples
                details = [
                    {
                        "object_type": obj.get("object_type"),
                        "track_id": obj.get("track_id"),
                        "color": obj.get("color")
                    }
                    for obj in objects if "color" in obj and "object_type" in obj and "track_id" in obj
                ]
                results["frame_number"] = frame_number
                results["object_details"] = details
            elif args.get("object_type") and args.get("color"):
                # Query for count of objects of a specific color and type (e.g., red cars)
                object_type_query = str(args.get("object_type")).strip().lower()
                color_query = str(args.get("color")).strip().lower()
                matching_track_ids = {
                    str(obj["track_id"])
                    for obj in all_objects
                    if str(obj.get("object_type", "")).strip().lower() == object_type_query
                        and str(obj.get("color", "")).strip().lower() == color_query
                        and "track_id" in obj
                }
                results[f"{color_query}_{object_type_query}_count"] = len(matching_track_ids)
                results[f"{color_query}_{object_type_query}_track_ids"] = list(matching_track_ids)
            elif args.get("color") and not any([args.get("object_type"), args.get("track_id"), args.get("frame_number")]):
                # Only color provided: return breakdown by object type
                color_query = str(args.get("color")).strip().lower()
                type_to_tracks = {}
                for obj in all_objects:
                    if str(obj.get("color", "")).strip().lower() == color_query and "object_type" in obj and "track_id" in obj:
                        obj_type = str(obj.get("object_type")).strip().lower()
                        if obj_type not in type_to_tracks:
                            type_to_tracks[obj_type] = set()
                        type_to_tracks[obj_type].add(str(obj["track_id"]))
                # Format result: e.g., { 'car': 10, 'truck': 2 }
                results[f"{color_query}_count"] = {k: len(v) for k, v in type_to_tracks.items()}
                results[f"{color_query}_track_ids"] = {k: list(v) for k, v in type_to_tracks.items()}
            else:
                # Neither provided: all unique colors in the video
                colors = list({obj.get("color") for obj in all_objects if "color" in obj})
                results["colors"] = colors

        # ✅ Object types
        if get_object_types:
            if track_id is not None and frame_number is not None:
                # Both track_id and frame_number provided: filter by both
                objects = [
                    obj for obj in all_objects
                    if str(obj.get("track_id")) == str(track_id) and int(obj.get("frame_number")) == int(frame_number)
                ]
                types = list({obj.get("object_type") for obj in objects if "object_type" in obj})
                results["track_id"] = track_id
                results["frame_number"] = frame_number
                results["object_type"] = types
            elif track_id is not None:
                # Only track_id provided
                objects = [obj for obj in all_objects if str(obj.get("track_id")) == str(track_id)]
                types = list({obj.get("object_type") for obj in objects if "object_type" in obj})
                results["track_id"] = track_id
                results["object_type"] = types
            elif frame_number is not None:
                # Only frame_number provided
                objects = [obj for obj in all_objects if int(obj.get("frame_number")) == int(frame_number)]
                types = list({obj.get("object_type") for obj in objects if "object_type" in obj})
                results["frame_number"] = frame_number
                results["object_types"] = types   
            else:
                # Neither provided: all unique types in the video
                types = list({obj.get("object_type") for obj in all_objects if "object_type" in obj})
                results["object_types"] = types

        # ✅ Default fallback if no flags
        if not any([get_count, get_unique, get_confidences, get_object_types, get_frame_object_count, get_frame_track_ids, get_position_by_frame_and_track, get_object_color]):
            results["result"] = self.filter_objects(
                all_objects, fields=["object_type", "track_id", "confidence"]
            )

        print('results', results)
        return {"result": results, "reformat": True}



    
    
    
    def filter_objects(self, objects, fields=None, query=None):
        """
        Filters a list of object dicts to only include specified fields or answers user query.
        Args:
            objects (list): List of dicts representing objects.
            fields (list, optional): List of fields to retain in each dict.
            query (str, optional): User query to interpret for a relevant answer.
        Returns:
            list or dict: Filtered list of dicts or relevant answer.
        """
        if fields is not None:
            return [{k: obj[k] for k in fields if k in obj} for obj in objects]

        # If a query is provided, try to answer it
        if query:
            q = query.lower()
            if "color" in q:
                colors = list({obj.get("color") for obj in objects if "color" in obj})
                return {"colors": colors}
            if "count" in q or "how many" in q:
                return {"count": len(objects)}
            if "type" in q or "object" in q:
                types = list({obj.get("object_type") for obj in objects if "object_type" in obj})
                return {"object_types": types}
            if "confidence" in q:
                confidences = [obj.get("confidence") for obj in objects if "confidence" in obj]
                return {"confidences": confidences}
            if "position" in q or "location" in q:
                positions = [obj.get("position") for obj in objects if "position" in obj]
                return {"positions": positions}
            # Add more patterns as needed

        # Default: return all objects
        return {"result": objects, "reformat": True}


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

        # === 3. Count objects within N seconds (legacy) ===
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

        # === 3b. Flexible time-based object queries ===
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
        window_start = None
        window_end = None
        if last_n_seconds is not None:
            try:
                last_n_seconds = float(last_n_seconds)
                if video_end_time is not None:
                    window_start = max(0, video_end_time - last_n_seconds)
                    window_end = video_end_time
            except Exception:
                pass
        elif time_range_start is not None or time_range_end is not None:
            try:
                window_start = float(time_range_start) if time_range_start is not None else 0
                window_end = float(time_range_end) if time_range_end is not None else video_end_time
            except Exception:
                pass
        # Only run if a valid window is set
        if window_start is not None and window_end is not None and window_end > window_start:
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
                        if o_end >= window_start and o_start <= window_end:
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
                "window_start": window_start,
                "window_end": window_end,
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



    async def show_evidence(self, args):
        """
        Retrieves evidence images and captions for detected objects in a video.
        
        Supports filtering by object type, track ID, frame number, or general question text.
        Can be used to answer questions like:
        - "Show me frames with red cars"
        - "Show evidence of person near entrance in frame 120"
        - "Display frames where object 25 appeared"
        """
        if args is None:
            args = {}
        elif isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                return {"result": "Invalid arguments: could not parse JSON."}

        # Extract params
        video_id = args.get("video_id")
        object_type = args.get("object_type")
        track_id = args.get("track_id")
        frame_number = args.get("frame_number")
        color_query = args.get("color", "")
        start_time = args.get("start_time")
        end_time = args.get("end_time")
        get_evidence = args.get("get_evidence", True)
        question = args.get("question", "")

        if not get_evidence:
            return {"result": "Evidence retrieval not requested."}
        if not video_id:
            return {"error": "video_id is required."}

        # Get video document
        video_doc = self.db["video_details"].find_one({"video_id": video_id})
        if not video_doc:
            return {"error": "Video not found."}

        evidence_images = []
        for frame in video_doc.get("frames", []):
            if frame_number is not None and int(frame["frame_number"]) != int(frame_number):
                continue
            if color_query and frame.get("color", "").lower() != color_query.lower():
                continue
            if start_time is not None and end_time is not None:
                frame_time = float(frame.get("frame_time", 0))
                if not (start_time <= frame_time <= end_time):
                    continue
            for obj in frame.get("objects", []):
                if object_type and obj.get("object_type") != object_type:
                    continue
                if track_id and str(obj.get("track_id")) != str(track_id):
                    continue
                frame_id = frame["frame_id"]
                # Try to get image from MongoDB (BSON Binary)
                image_doc = self.db["frame_images"].find_one({"video_id": video_id, "frame_id": frame_id})
                image_b64 = None
                if image_doc and image_doc.get("image_data"):
                    # image_data is BSON Binary
                    image_b64 = base64.b64encode(image_doc["image_data"]).decode("utf-8")
                else:
                    # fallback: load from disk if not in db
                    image_path = os.path.join("frames", video_id, f"{frame_id}.jpg")
                    if os.path.exists(image_path):
                        with open(image_path, "rb") as img_file:
                            image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
                evidence_images.append({
                    "image_data": image_b64,
                    "caption": f"{obj.get('object_type', 'Object').capitalize()} detected in frame {frame['frame_number']}",
                    "frame_number": frame["frame_number"],
                    "video_id": video_id,
                    "object_type": obj.get('object_type', ''),
                    "track_id": obj.get('track_id', None)
                })
        if not evidence_images:
            return {"result": "No evidence frames found for the query."}
        response_text = "These are the evidence frames showing the detected " + (object_type or "objects") + "."
        return {
            "type": "evidence",
            "images": evidence_images,
            "text": response_text
        }
