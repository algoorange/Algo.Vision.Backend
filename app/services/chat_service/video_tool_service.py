from pymongo import MongoClient
import os
import json
import re

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
                        "confidence": obj.get("confidence")
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

        # ✅ Object types
        if get_object_types:
            if track_id is not None:
                objects = [obj for obj in all_objects if obj.get("track_id") == track_id]
                types = list({obj.get("object_type") for obj in objects if "object_type" in obj})
                results["track_id"] = track_id
                results["object_type"] = types
            else:
                types = list({obj.get("object_type") for obj in all_objects if "object_type" in obj})
                results["object_types"] = types

        # ✅ Default fallback if no flags
        if not any([get_count, get_unique, get_confidences, get_object_types, get_frame_object_count, get_frame_track_ids]):
            results["result"] = filter_objects(
                all_objects, fields=["object_type", "track_id", "confidence"]
            )

        print('results', results)
        return results


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

        # === 3. Count objects within N seconds ===
        if count_within_seconds > 0:
            track_ids = set()
            total_count = 0
            object_type_counts = {}
            for segment in segments:
                start_time = segment.get("summary", {}).get("start_time", 0)
                end_time = segment.get("summary", {}).get("end_time", 0)
                if start_time < count_within_seconds:
                    objects = segment.get("summary", {}).get("objects", [])
                    for obj in objects:
                        if isinstance(obj, dict) and "track_id" in obj:
                            track_ids.add(obj["track_id"])
                            obj_type = obj.get("object_type", "unknown")
                            object_type_counts[obj_type] = object_type_counts.get(obj_type, 0) + 1
                            total_count += 1
            result["objects_in_first_n_seconds"] = {
                "within_seconds": count_within_seconds,
                "unique_track_ids": list(track_ids),
                "unique_count": len(track_ids),
                "total_detections": total_count,
                "by_type": object_type_counts
            }

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
                "`get_segment_object_counts`, `get_total_object_count`, `count_within_seconds`, or track-level flags."
            )

        return result



    async def get_specific_object_type(self, args):
        """
        Returns all objects of a specific type or a specific track_id from all frames for the given video_id.
        Args:
            video_id (str): The video ID to search for.
            object_type (str): (optional) The type of object to filter (e.g., 'car', 'truck').
            question (str): (optional) The user's question, which may contain track_id or object_type.
        Returns:
            dict: List of dicts with selected fields for each object of the specified type or track_id.
        """
        if args is None:
            args = {}
        elif isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                return {"result": "Invalid arguments: could not parse JSON."}

        video_id = args.get("video_id")
        object_type = args.get("object_type")
        question = args.get("question", "")

        video_doc = collection.find_one({"video_id": video_id})
        if not video_doc or "frames" not in video_doc:
            return {"result": "No data found for this video."}

        # Try to extract track_id from question
        track_id = args.get("track_id")
        if not track_id and question:
            match = re.search(r"track[_ ]?id[:= ]+(\d+)", question, re.IGNORECASE)
            if match:
                track_id = match.group(1)

        # Try to extract object_type from question if not provided
        if not object_type and question:
            known_types = ["car", "truck", "person", "bus", "bike", "airplane"]
            for t in known_types:
                if t in question.lower():
                    object_type = t
                    break

        # Try to extract frame_number from args or question
        frame_number = args.get("frame_number")
        if not frame_number and question:
            match = re.search(r"frame[_ ]?number[:= ]+(\d+)", question, re.IGNORECASE)
            if match:
                frame_number = int(match.group(1))
            else:
                # Try a more general pattern: "frame 123"
                match = re.search(r"frame[\s#]*(\d+)", question, re.IGNORECASE)
                if match:
                    frame_number = int(match.group(1))

        # Priority: frame_number + track_id > frame_number > track_id > object_type
        if frame_number is not None:
            frame_number = int(frame_number)
            for frame in video_doc["frames"]:
                if frame.get("frame_number") == frame_number:
                    # If track_id is also specified, filter within this frame
                    if track_id:
                        filtered = [obj for obj in frame.get("objects", []) if isinstance(obj, dict) and str(obj.get("track_id")) == str(track_id)]
                        if not filtered:
                            return {"result": f"No object with track_id {track_id} found in frame {frame_number}."}
                        return {"result": filtered}
                    # Otherwise, return all objects in this frame (dicts only)
                    all_objs = [obj for obj in frame.get("objects", []) if isinstance(obj, dict)]
                    return {"result": all_objs}
            return {"result": f"No frame found with frame_number {frame_number}."}

        if track_id:
            filtered_objects = []
            for frame in video_doc["frames"]:
                for obj in frame.get("objects", []):
                    if isinstance(obj, dict) and str(obj.get("track_id")) == str(track_id):
                        filtered_objects.append(obj)
            if not filtered_objects:
                return {"result": f"No object with track_id {track_id} found in the video."}
            return {"result": filtered_objects}

        if object_type:
            filtered_objects = []
            for frame in video_doc["frames"]:
                for obj in frame.get("objects", []):
                    if obj.get("object_type") == object_type:
                        filtered_obj = {
                            "track_id": obj.get("track_id"),
                            "frame_number": obj.get("frame_number"),
                            "object_type": obj.get("object_type"),
                            "object_position": obj.get("position"),
                            "confidence": obj.get("confidence")
                        }
                        filtered_objects.append(filtered_obj)
            if not filtered_objects:
                return {"result": f"No objects of type '{object_type}' found in the video."}
            return {"result": filtered_objects}

        return {"result": "Please specify an object type, a track_id, or a frame number in your question."}

    async def get_traffic_congestion_details(self, args):
        """
        Answer a user question about the video and detected objects using the DB.
        Args:
            video_id (str): The video ID to analyze.
            question (str): The user's question (natural language).
        Returns:
            str: The answer to the question.
        """
        if isinstance(args, str):
            args = json.loads(args)
        video_id = args.get("video_id")
        question = args.get("question")
        import re
        video_doc = collection.find_one({"video_id": video_id})
        if not video_doc or "frames" not in video_doc:
            return {"result": "No data found for this video."}
        frames = video_doc["frames"]

        # Helper: get all objects by track_id
        objects_by_track = {}
        for frame in frames:
            for obj in frame.get("objects", []):
                tid = obj.get("track_id")
                if tid:
                    if tid not in objects_by_track:
                        objects_by_track[tid] = []
                    objects_by_track[tid].append(obj)

        q = question.lower()

        # 1. How many total objects were detected?
        if re.search(r"total objects|how many objects", q):
            unique_ids = set(objects_by_track.keys())
            return {"result": f"Total unique objects detected: {len(unique_ids)}"}

        # 2. How many unique object categories?
        if re.search(r"unique object categories|categories|types", q):
            categories = set()
            for objs in objects_by_track.values():
                for obj in objs:
                    categories.add(obj.get("object_type"))
            return {"result": f"Unique object categories: {', '.join(str(c) for c in categories if c)}"}

        # 3. Congestion/crowding
        if "congestion" in q or "crowding" in q or "traffic" in q:
            # If many objects in the same frame
            threshold = 10 # arbitrary
            for frame in frames:
                if len(frame.get("objects", [])) > threshold:
                    return {"result": f"Congestion detected at frame {frame['frame_number']}"}
            return {"result": "No significant congestion detected."}

        # 4. Default
        return {"result": "Sorry, I cannot answer this question yet."}    


#at testing phase
    async def get_all_object_direction(self, args):
        import math
        if isinstance(args, str):
            args = json.loads(args)

        video_id = args.get("video_id")
        specific_track_id = args.get("track_id")
        video_doc = collection.find_one({"video_id": video_id})
        if not video_doc or "frames" not in video_doc:
            return {"result": "No data found for this video."}

        track_positions = {}
        track_types = {}
        track_times = {}
        for frame in sorted(video_doc["frames"], key=lambda f: f.get("frame_number", 0)):
            frame_number = frame.get("frame_number")
            frame_time = frame.get("frame_time") if "frame_time" in frame else None
            for obj in frame.get("objects", []):
                track_id = obj.get("track_id")
                obj_type = obj.get("object_type")
                pos = obj.get("position", {})
                if track_id and all(k in pos for k in ("x", "x1", "y", "y1")):
                    cx = (pos["x"] + pos["x1"]) / 2
                    cy = (pos["y"] + pos["y1"]) / 2
                    if track_id not in track_positions:
                        track_positions[track_id] = []
                        track_types[track_id] = obj_type
                        track_times[track_id] = []
                    track_positions[track_id].append({
                        "frame_number": frame_number,
                        "frame_time": frame_time,
                        "cx": cx,
                        "cy": cy,
                        "object_type": obj_type,
                        "position": pos
                    })
                    track_times[track_id].append(frame_time)

        # Determine movement direction (angle) and towards/away classification
        directions = {}
        car_directions = {}
        towards_count, away_count = 0, 0
        towards_ids, away_ids = [], []
        car_movement_summary = {}
        for track_id, positions in track_positions.items():
            if len(positions) < 2:
                continue
            start = positions[0]
            end = positions[-1]
            dx = end["cx"] - start["cx"]
            dy = end["cy"] - start["cy"]
            angle = math.degrees(math.atan2(dy, dx))
            angle = (angle + 360) % 360
            # Angle buckets
            if 22.5 <= angle < 67.5:
                direction = "northeast"
            elif 67.5 <= angle < 112.5:
                direction = "east"
            elif 112.5 <= angle < 157.5:
                direction = "southeast"
            elif 157.5 <= angle < 202.5:
                direction = "south"
            elif 202.5 <= angle < 247.5:
                direction = "southwest"
            elif 247.5 <= angle < 292.5:
                direction = "west"
            elif 292.5 <= angle < 337.5:
                direction = "northwest"
            else:
                direction = "north"
            directions[track_id] = {
                "from_frame": start["frame_number"],
                "to_frame": end["frame_number"],
                "direction": direction,
                "angle": round(angle, 2),
                "object_type": track_types[track_id]
            }
            # Towards/away logic (assume decreasing Y = towards, increasing Y = away)
            if dy < 0:
                towards_count += 1
                towards_ids.append(track_id)
            elif dy > 0:
                away_count += 1
                away_ids.append(track_id)
            # Car-specific summary
            if track_types[track_id] == "car":
                car_directions[track_id] = direction
                car_movement_summary[direction] = car_movement_summary.get(direction, 0) + 1

        # Details for a specific track_id (e.g., truck_id)
        specific_track_info = None
        if specific_track_id is not None:
            tid = str(specific_track_id)
            if tid in track_positions:
                pos_list = track_positions[tid]
                first = pos_list[0]
                last = pos_list[-1]
                specific_track_info = {
                    "object_type": track_types[tid],
                    "positions": [{"frame_number": p["frame_number"], "cx": p["cx"], "cy": p["cy"]} for p in pos_list],
                    "detection_times": track_times[tid],
                    "first_detected_frame": first["frame_number"],
                    "last_detected_frame": last["frame_number"],
                    "direction": directions[tid]["direction"] if tid in directions else None
                }
            else:
                specific_track_info = {"error": f"track_id {tid} not found."}

        result = {
            "objects_moving_towards_camera": towards_count,
            "objects_moving_away_from_camera": away_count,
            "towards_ids": towards_ids,
            "away_ids": away_ids,
            "car_movement_summary": car_movement_summary,
            "car_directions": car_directions,
            "directions": directions,
            "specific_track_info": specific_track_info
        }
        print(result)
        return {"result": result}
        

#parcially working 
    async def object_position_confidence(self, args):
        """
        Returns position and confidence for all objects of a specific type in a specific frame.
        Args:
            video_id (str): The video ID to search for.
            object_type (str): The type of object to filter (e.g., 'car', 'person').
            frame_number (int): The frame number to search in.
            question (str): The user's question to extract additional parameters.
        Returns:
            dict: List of dicts with position and confidence for each matching object.
        """
        if isinstance(args, str):
            args = json.loads(args)

        video_id = args.get("video_id")
        object_type = args.get("object_type")
        frame_number = args.get("frame_number")
        question = args.get("question", "")

        video_doc = collection.find_one({"video_id": video_id})
        if not video_doc or "frames" not in video_doc:
            return {"result": []}

        # Extract additional parameters from the question
        track_id = args.get("track_id")
        if not track_id and question:
            match = re.search(r"track[_ ]?id[:= ]?(\d+)", question, re.IGNORECASE)
            if match:
                track_id = match.group(1)

        if not object_type and question:
            known_types = ["car", "truck", "person", "bus", "bike", "airplane"]
            for t in known_types:
                if t in question.lower():
                    object_type = t
                    break

        if not frame_number and question:
            match = re.search(r"frame[_ ]?number[:= ]?(\d+)", question, re.IGNORECASE)
            if match:
                frame_number = int(match.group(1))
            else:
                match = re.search(r"frame[\s#]*(\d+)", question, re.IGNORECASE)
                if match:
                    frame_number = int(match.group(1))

        # Helper: Determine movement direction for a track_id
        def determine_movement_direction(track_id, video_doc):
            positions = []
            for frame in video_doc["frames"]:
                for obj in frame.get("objects", []):
                    if str(obj.get("track_id")) == str(track_id):
                        positions.append((frame.get("frame_number"), obj.get("position")))
            if len(positions) < 2:
                return "Not enough data to determine movement."
            x_positions = [pos[1]['x'] for pos in positions]
            if x_positions[-1] < x_positions[0]:
                return "left"
            elif x_positions[-1] > x_positions[0]:
                return "right"
            else:
                return "stationary"

        # Helper: Check if object is stationary
        def is_object_stationary(track_id, video_doc):
            positions = []
            for frame in video_doc["frames"]:
                for obj in frame.get("objects", []):
                    if str(obj.get("track_id")) == str(track_id):
                        positions.append((frame.get("frame_number"), obj.get("position")))
            if len(positions) < 2:
                return True
            x_positions = [pos[1]['x'] for pos in positions]
            return all(x == x_positions[0] for x in x_positions)

        # Helper: Count objects moving left
        def count_objects_moving_left(video_doc, object_type=None):
            left_moving_objects = set()
            for frame in video_doc["frames"]:
                for obj in frame.get("objects", []):
                    if object_type and obj.get("object_type") != object_type:
                        continue
                    track_id = obj.get("track_id")
                    positions = []
                    for f in video_doc["frames"]:
                        for o in f.get("objects", []):
                            if str(o.get("track_id")) == str(track_id):
                                positions.append((f.get("frame_number"), o.get("position")))
                    if len(positions) >= 2:
                        x_positions = [pos[1]['x'] for pos in positions]
                        if x_positions[-1] < x_positions[0]:
                            left_moving_objects.add(track_id)
            return len(left_moving_objects)

        # Tool-call logic
        if track_id and question and "moved left" in question.lower():
            direction = determine_movement_direction(track_id, video_doc)
            return {"result": {"track_id": track_id, "movement_direction": direction}}

        if track_id and question and "stopped" in question.lower():
            stationary = is_object_stationary(track_id, video_doc)
            return {"result": {"track_id": track_id, "stationary": stationary}}
        if question and "how many object are moving on left side" in question.lower():
            count = count_objects_moving_left(video_doc, object_type)
            return {"result": {"object_type": object_type, "moving_left_count": count}}

        # Strictly filter by object_type if specified
        if frame_number is not None:
            try:
                frame_number = int(frame_number)
            except Exception:
                return {"result": "Invalid frame_number."}
            for frame in video_doc["frames"]:
                if frame.get("frame_number") == frame_number:
                    results = []
                    for obj in frame.get("objects", []):
                        if object_type:
                            if obj.get("object_type") == object_type:
                                entry = {
                                    "object_type": obj.get("object_type"),
                                    "position": obj.get("position"),
                                    "confidence": obj.get("confidence")
                                }
                                # Only include track_id if explicitly requested
                                if (track_id or "track id" in question.lower() or "which" in question.lower() or "list id" in question.lower()):
                                    entry["track_id"] = obj.get("track_id")
                                results.append(entry)
                        else:
                            entry = {
                                "object_type": obj.get("object_type"),
                                "position": obj.get("position"),
                                "confidence": obj.get("confidence")
                            }
                            if (track_id or "track id" in question.lower() or "which" in question.lower() or "list id" in question.lower()):
                                entry["track_id"] = obj.get("track_id")
                            results.append(entry)
                    return {"result": results}

        if track_id and question and "moved left" in question.lower():
            direction = determine_movement_direction(track_id, video_doc)
            return {"result": {"track_id": track_id, "movement_direction": direction}}

        if track_id and question and "stopped" in question.lower():
            stationary = is_object_stationary(track_id, video_doc)
            return {"result": {"track_id": track_id, "stationary": stationary}}

        return {"result": "Please specify a valid question or parameters."}


