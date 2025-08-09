from pymongo import MongoClient
import os
import json
import base64
from bson.binary import Binary

client = MongoClient(os.getenv("MONGO_URI"))
db = client["algo_compliance_db_2"]
collection = db["video_details"]
video_details_segment = db["video_details_segment"]

class AllObjectToolService:
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

        # Optional general filters to make queries user-friendly without requiring track_id
        filter_object_type = args.get("object_type")
        filter_color = args.get("color")

        def matches_filters(o):
            if filter_object_type and str(o.get("object_type", "")).strip().lower() != str(filter_object_type).strip().lower():
                return False
            # Prefer properties.color for filtering, fallback to legacy color
            def _obj_color(val):
                props = o.get("properties") or {}
                if not isinstance(props, dict):
                    props = {}
                return str(props.get("color", o.get("color", ""))).strip().lower()
            if filter_color and _obj_color(o) != str(filter_color).strip().lower():
                return False
            # If frame_number provided, many queries naturally scope to that frame
            if frame_number is not None and int(o.get("frame_number", -1)) != int(frame_number):
                return False
            return True

        # Pre-filtered view reflecting any provided filters
        filtered_all = [o for o in all_objects if matches_filters(o)]

        # ✅ Track IDs in a specific frame
        if get_frame_track_ids and frame_number is not None:
            frame_track_ids = {
                obj["track_id"]
                for obj in filtered_all
                if obj.get("frame_number") == frame_number and "track_id" in obj
            }
            results["frame_number"] = frame_number
            results["track_ids"] = list(frame_track_ids)

        # ✅ General count of unique track IDs
        if get_count:
            unique_ids = {obj.get("track_id") for obj in filtered_all if "track_id" in obj}
            results["count"] = len(unique_ids)

        # ✅ Unique object list by track ID
        if get_unique:
            seen = {}
            for obj in filtered_all:
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
                objects = [obj for obj in filtered_all if obj.get("track_id") == track_id]
                confidences = list({obj.get("confidence") for obj in objects if "confidence" in obj})
                results["track_id"] = track_id
                results["confidences"] = confidences
            else:
                confidences = list({obj.get("confidence") for obj in filtered_all if "confidence" in obj})
                results["confidences"] = confidences

        # ✅ Object count in specific frame
        if get_frame_object_count and frame_number is not None:
            unique_ids = set()
            for obj in filtered_all:
                if obj.get("frame_number") == frame_number:
                    tid = obj.get("track_id")
                    if tid is not None:
                        unique_ids.add(tid)
            results["frame_number"] = frame_number
            results["object_count"] = len(unique_ids)

        if get_position_by_frame_and_track and track_id is not None and frame_number is not None:       
            matching_objects = [
                obj for obj in filtered_all
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
            def _get_color_from_obj(obj: dict) -> str:
                props = obj.get("properties") or {}
                if not isinstance(props, dict):
                    props = {}
                return str(props.get("color", obj.get("color", "")))
            if track_id is not None and frame_number is not None:
                # Both track_id and frame_number provided: filter by both
                objects = [
                    obj for obj in filtered_all
                    if str(obj.get("track_id")) == str(track_id) and int(obj.get("frame_number")) == int(frame_number)
                ]
                colors = list({
                    _get_color_from_obj(obj)
                    for obj in objects
                    if ("properties" in obj and isinstance(obj.get("properties"), dict) and "color" in obj["properties"]) or "color" in obj
                })
                results["track_id"] = track_id
                results["frame_number"] = frame_number
                results["color"] = colors
            elif track_id is not None:
                # Only track_id provided
                objects = [obj for obj in filtered_all if str(obj.get("track_id")) == str(track_id)]
                colors = list({
                    _get_color_from_obj(obj)
                    for obj in objects
                    if ("properties" in obj and isinstance(obj.get("properties"), dict) and "color" in obj["properties"]) or "color" in obj
                })
                results["track_id"] = track_id
                results["color"] = colors
            elif frame_number is not None:
                # Only frame_number provided
                objects = [obj for obj in filtered_all if int(obj.get("frame_number")) == int(frame_number)]
                # Return all unique (object_type, track_id, color) tuples
                details = [
                    {
                        "object_type": obj.get("object_type"),
                        "track_id": obj.get("track_id"),
                        "color": _get_color_from_obj(obj)
                    }
                    for obj in objects if ("object_type" in obj and "track_id" in obj and (("properties" in obj and isinstance(obj.get("properties"), dict) and "color" in obj["properties"]) or "color" in obj))
                ]
                results["frame_number"] = frame_number
                results["object_details"] = details
            elif args.get("object_type") and args.get("color"):
                # Query for count of objects of a specific color and type (e.g., red cars)
                object_type_query = str(args.get("object_type")).strip().lower()
                color_query = str(args.get("color")).strip().lower()
                matching_track_ids = {
                    str(obj["track_id"])
                    for obj in filtered_all
                    if str(obj.get("object_type", "")).strip().lower() == object_type_query
                        and str((_get_color_from_obj(obj) or "")).strip().lower() == color_query
                        and "track_id" in obj
                }
                results[f"{color_query}_{object_type_query}_count"] = len(matching_track_ids)
                results[f"{color_query}_{object_type_query}_track_ids"] = list(matching_track_ids)
            elif args.get("color") and not any([args.get("object_type"), args.get("track_id"), args.get("frame_number")]):
                # Only color provided: return breakdown by object type
                color_query = str(args.get("color")).strip().lower()
                type_to_tracks = {}
                for obj in filtered_all:
                    if str((_get_color_from_obj(obj) or "")).strip().lower() == color_query and "object_type" in obj and "track_id" in obj:
                        obj_type = str(obj.get("object_type")).strip().lower()
                        if obj_type not in type_to_tracks:
                            type_to_tracks[obj_type] = set()
                        type_to_tracks[obj_type].add(str(obj["track_id"]))
                # Format result: e.g., { 'car': 10, 'truck': 2 }
                results[f"{color_query}_count"] = {k: len(v) for k, v in type_to_tracks.items()}
                results[f"{color_query}_track_ids"] = {k: list(v) for k, v in type_to_tracks.items()}
            else:
                # Neither provided: all unique colors in the video
                colors = list({
                    _get_color_from_obj(obj)
                    for obj in filtered_all
                    if ("properties" in obj and isinstance(obj.get("properties"), dict) and "color" in obj["properties"]) or "color" in obj
                })
                results["colors"] = colors

        # ✅ Object types
        if get_object_types:
            if track_id is not None and frame_number is not None:
                # Both track_id and frame_number provided: filter by both
                objects = [
                    obj for obj in filtered_all
                    if str(obj.get("track_id")) == str(track_id) and int(obj.get("frame_number")) == int(frame_number)
                ]
                types = list({obj.get("object_type") for obj in objects if "object_type" in obj})
                results["track_id"] = track_id
                results["frame_number"] = frame_number
                results["object_type"] = types
            elif track_id is not None:
                # Only track_id provided
                objects = [obj for obj in filtered_all if str(obj.get("track_id")) == str(track_id)]
                types = list({obj.get("object_type") for obj in objects if "object_type" in obj})
                results["track_id"] = track_id
                results["object_type"] = types
            elif frame_number is not None:
                # Only frame_number provided
                objects = [obj for obj in filtered_all if int(obj.get("frame_number")) == int(frame_number)]
                types = list({obj.get("object_type") for obj in objects if "object_type" in obj})
                results["frame_number"] = frame_number
                results["object_types"] = types   
            else:
                # Neither provided: all unique types in the video
                types = list({obj.get("object_type") for obj in filtered_all if "object_type" in obj})
                results["object_types"] = types

        # ✅ Default fallback if no flags
        if not any([get_count, get_unique, get_confidences, get_object_types, get_frame_object_count, get_frame_track_ids, get_position_by_frame_and_track, get_object_color]):
            results["result"] = self.filter_objects(
                filtered_all, fields=["object_type", "track_id", "confidence"]
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
                def _get_color_from_obj(obj: dict) -> str:
                    props = obj.get("properties") or {}
                    if not isinstance(props, dict):
                        props = {}
                    return str(props.get("color", obj.get("color", "")))
                colors = list({
                    _get_color_from_obj(obj)
                    for obj in objects
                    if ("properties" in obj and isinstance(obj.get("properties"), dict) and "color" in obj["properties"]) or "color" in obj
                })
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

