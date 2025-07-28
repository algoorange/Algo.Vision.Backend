from pymongo import MongoClient
import os
import json

client = MongoClient(os.getenv("MONGO_URI"))
db = client["algo_compliance_db_2"]
collection = db["video_details"]

class VideoToolService:
    def __init__(self, request):
        self.request = request

    # async def analyze_video_behavior(args):
        import json
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                return {"result": "Invalid arguments: could not parse JSON."}
        """
        This function answers questions about a video using data from the database.
        You can extend this function to support more types of analytics.
        """
        query = args.get("query")  # User's question, e.g., 'How many cars are in the video?'
        video_id = args.get("video_id")  # Video ID to look up in the DB

        # Fetch the video data from MongoDB
        video = collection.find_one({"video_id": video_id})
        if not video:
            return {"result": f"No video found with ID: {video_id}"}

        frames = video.get("frames", [])
        video_name = video.get('video_name', 'Unknown')

        # Gather all objects from all frames
        all_objects = [obj for frame in frames for obj in frame.get("objects", [])]

        # Count objects by type (e.g., car, truck, etc.)
        type_counts = {}
        for obj in all_objects:
            obj_type = obj.get('object_type', 'unknown')
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1

        # Beginner-friendly: Simple natural language matching
        if query:
            q = query.lower()
            if 'how many cars' in q:
                car_count = type_counts.get('car', 0)
                return {"result": f"There are {car_count} cars detected in video '{video_name}'."}
            elif 'how many trucks' in q:
                truck_count = type_counts.get('truck', 0)
                return {"result": f"There are {truck_count} trucks detected in video '{video_name}'."}
            elif 'total objects' in q or 'how many objects' in q:
                return {"result": f"There are {len(all_objects)} objects detected in video '{video_name}'."}
            else:
                # Default: Just list all object types and their counts
                type_summary = ', '.join([f"{k}: {v}" for k, v in type_counts.items()])
                return {"result": f"Object counts in video '{video_name}': {type_summary}", "reformatting": True}
        else:
            return {"result": "No query provided."}

        # You can add more 'elif' blocks above for more question types!

    # async def get_object_details(self, args):
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                return {"result": "Invalid arguments: could not parse JSON."}
        """
        Returns details of all objects of a specific type in the video.
        Args must include 'object_type' and 'video_id'.
        """
        object_type = args.get("object_type")
        video_id = args.get("video_id")
        if not object_type or not video_id:
            return {"result": "Missing object_type or video_id."}

        video = collection.find_one({"video_id": video_id})
        if not video:
            return {"result": f"No video found with ID: {video_id}"}
        frames = video.get("frames", [])
        video_name = video.get('video_name', 'Unknown')

        # Gather all objects of the specified type
        matching_objects = [
            {"frame_number": frame.get("frame_number"), "frame_time": frame.get("frame_time"), **obj}
            for frame in frames for obj in frame.get("objects", [])
            if obj.get("object_type") == object_type
        ]
        if not matching_objects:
            return {"result": f"No objects of type '{object_type}' found in video '{video_name}'."}
        return {
            "result": f"Found {len(matching_objects)} '{object_type}' objects in video '{video_name}'.",
            "reformatting": True
        }

    # async def get_object_statistics(self, args):
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                return {"result": "Invalid arguments: could not parse JSON."}
        """
        Returns statistics (count, average confidence, etc.) for objects of a specific type in the video.
        Args must include 'object_type' and 'video_id'.
        """
        object_type = args.get("object_type")
        video_id = args.get("video_id")
        if not object_type or not video_id:
            return {"result": "Missing object_type or video_id."}

        video = collection.find_one({"video_id": video_id})
        if not video:
            return {"result": f"No video found with ID: {video_id}"}
        frames = video.get("frames", [])
        video_name = video.get('video_name', 'Unknown')

        matching_objects = [obj for frame in frames for obj in frame.get("objects", []) if obj.get("object_type") == object_type]
        count = len(matching_objects)
        if count == 0:
            return {"result": f"No objects of type '{object_type}' found in video '{video_name}'."}
        avg_confidence = sum(obj.get("confidence", 0) for obj in matching_objects) / count
        return {
            "result": f"Statistics for '{object_type}' in video '{video_name}': count={count}, avg_confidence={avg_confidence:.2f}",
            "reformatting": True
        }

    async def get_all_object_details(self, args):
        """
        Returns all objects (from all frames) for the given video_id.
        Args:
            video_id (str): The video ID to search for.
        Returns:
            list: List of all objects (dicts) from all frames.
        """
        if isinstance(args, str):
            args = json.loads(args)
        video_id = args.get("video_id")
        video_doc = collection.find_one({"video_id": video_id})
        if not video_doc or "frames" not in video_doc:
            return {"result": []}
        all_objects = []
        for frame in video_doc["frames"]:
            all_objects.extend(frame.get("objects", []))
        return {"result": all_objects}

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
        import json
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

    # async def object_position_confidence(self, args):
        """
        Returns position and confidence for all objects of a specific type in a specific frame.
        Args:
            video_id (str): The video ID to search for.
            object_type (str): The type of object to filter (e.g., 'car', 'person').
            frame_number (int): The frame number to search in.
        Returns:
            list: List of dicts with position and confidence for each matching object.
        """
        if isinstance(args, str):
            args = json.loads(args)
        video_id = args.get("video_id")
        object_type = args.get("object_type")
        frame_number = args.get("frame_number")
        video_doc = collection.find_one({"video_id": video_id})
        if not video_doc or "frames" not in video_doc:
            return {"result": []}
        for frame in video_doc["frames"]:
            if frame.get("frame_number") == frame_number:
                results = []
                for obj in frame.get("objects", []):
                    if obj.get("object_type") == object_type:
                        results.append({
                            "position": obj.get("position"),
                            "confidence": obj.get("confidence")
                        })
                return {"result": results}
        return {"result": []}    


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
    # async def count_left_right_moving_objects(self, args):
        """
        Counts how many unique objects move left and right across the video.
        Returns: dict with counts and lists of track_ids for left and right movers.
        """
        if isinstance(args, str):
            args = json.loads(args)
        video_id = args.get("video_id")
        video_doc = collection.find_one({"video_id": video_id})
        if not video_doc or "frames" not in video_doc:
            return {"result": {"left": 0, "right": 0, "left_ids": [], "right_ids": []}}
        # Gather all positions for each track_id
        track_positions = {}
        for frame in video_doc["frames"]:
            frame_number = frame.get("frame_number")
            for obj in frame.get("objects", []):
                tid = obj.get("track_id")
                pos = obj.get("position")
                if tid and pos and all(k in pos for k in ("x", "y", "x1", "y1")):
                    cx = (pos["x"] + pos["x1"]) / 2
                    track_positions.setdefault(tid, []).append((frame_number, cx))
        left_ids = []
        right_ids = []
        for tid, positions in track_positions.items():
            positions.sort()  # sort by frame_number
            if len(positions) >= 2:
                x_start = positions[0][1]
                x_end = positions[-1][1]
                if x_end < x_start:
                    left_ids.append(tid)
                elif x_end > x_start:
                    right_ids.append(tid)
        return {"result": {"left": len(left_ids), "right": len(right_ids), "left_ids": left_ids, "right_ids": right_ids}}


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
        import re
        import json
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
