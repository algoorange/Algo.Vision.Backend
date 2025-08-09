from pymongo import MongoClient
import os
import json
import base64
from bson.binary import Binary

client = MongoClient(os.getenv("MONGO_URI"))
db = client["algo_compliance_db_2"]
collection = db["video_details"]
video_details_segment = db["video_details_segment"]

class EvidenceToolService:
    def __init__(self, request):
        self.request = request
        self.db = request.app.db if request and hasattr(request, 'app') and hasattr(request.app, 'db') else db

  
  
  
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
            if start_time is not None and end_time is not None:
                frame_time = float(frame.get("frame_time", 0))
                if not (start_time <= frame_time <= end_time):
                    continue
            for obj in frame.get("objects", []):
                if object_type and obj.get("object_type") != object_type:
                    continue
                if track_id and str(obj.get("track_id")) != str(track_id):
                    continue
                # Apply color filter at the object level (not frame level)
                obj_color = obj.get("properties", {}).get("color", obj.get("color", ""))
                if color_query and str(obj_color).lower() != str(color_query).lower():
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
            return {
                "type": "evidence",
                "images": [],
                "text": f"No evidence frames found for your query: object_type={object_type}, color={color_query}, question='{question}'"
            }
        response_text = "These are the evidence frames showing the detected " + (object_type or "objects") + "."
        return {
            "type": "evidence",
            "images": evidence_images,
            "text": response_text
        }
