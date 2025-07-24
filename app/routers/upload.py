from fastapi import APIRouter, UploadFile, File
from app.services import video_processor
import os
import uuid
from pymongo import MongoClient

router = APIRouter()

# Connect to MongoDB (adjust the URI as needed)
client = MongoClient("mongodb://localhost:27017/")
db = client["algo_compliance_db_2"]  # Your database name
zone_coordinates_collection = db["zone_coordinates"]  # Your collection name


# Ensure the uploads folder exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

from fastapi import Form
import json
import datetime

@router.post("/")
async def upload_video(
    file: UploadFile = File(...),
    zoneCoords: str = Form(None),  # Receive zoneCoords as an optional string
    previewWidth: str = Form(None),
    previewHeight: str = Form(None)
):
    """
    Uploads the video and extracts frames from the uploaded video. Generates a unique video ID and stores frames per video.
    If zoneCoords is provided, it contains the restricted zone coordinates as a JSON string.
    Uses traditional object detection and tracking methods.
    """
    video_id = str(uuid.uuid4())

    # Parse the coordinates if present
    coords = None
    if zoneCoords:
        try:
            coords = json.loads(zoneCoords)
        except Exception as e:
            coords = None  # Optionally log error or handle as needed

        # Create a new document with the video ID and zone coordinates
        doc = {
            "UID": str(uuid.uuid4()),
            "video_id": video_id,
            "zone_coords": coords,
            "created_at": datetime.datetime.now()
        }
        # Insert the document into the collection
        zone_coordinates_collection.insert_one(doc)

    # Pass the coordinates and preview size to the video processor
    preview_w = int(previewWidth) if previewWidth else None
    preview_h = int(previewHeight) if previewHeight else None
    result = await video_processor.process_video(file, video_id, coords, preview_w, preview_h)
    return {"video_id": video_id, "result": result}


# @router.post("/llava")
# async def upload_video_with_llava(
#     file: UploadFile = File(...),
#     zoneCoords: str = Form(None),  # Receive zoneCoords as an optional string
#     previewWidth: str = Form(None),
#     previewHeight: str = Form(None)
# ):
#     """
#     Uploads the video and analyzes it using LLaVA (Large Language and Vision Assistant) via Groq API.
#     Generates a unique video ID and processes video frames with advanced vision-language understanding.
#     If zoneCoords is provided, it contains the restricted zone coordinates as a JSON string.
#     """
#     video_id = str(uuid.uuid4())

#     # Parse the coordinates if present
#     coords = None
#     if zoneCoords:
#         try:
#             coords = json.loads(zoneCoords)
#         except Exception as e:
#             coords = None  # Optionally log error or handle as needed

#         # Create a new document with the video ID and zone coordinates
#         doc = {
#             "UID": str(uuid.uuid4()),
#             "video_id": video_id,
#             "zone_coords": coords,
#             "analysis_method": "llava_groq",
#             "created_at": datetime.datetime.now()
#         }
#         # Insert the document into the collection
#         zone_coordinates_collection.insert_one(doc)

#     # Pass the coordinates and preview size to the LLaVA video processor
#     preview_w = int(previewWidth) if previewWidth else None
#     preview_h = int(previewHeight) if previewHeight else None
    
#     try:
#         result = await video_processor.process_video_with_laava(file, video_id, coords, preview_w, preview_h)
#         return {
#             "video_id": video_id, 
#             "result": result,
#             "analysis_method": "LLaVA via Groq",
#             "message": "Video processed successfully with LLaVA vision analysis"
#         }
#     except Exception as e:
#         return {
#             "video_id": video_id,
#             "error": f"LLaVA processing failed: {str(e)}",
#             "analysis_method": "LLaVA via Groq",
#             "message": "Video processing failed"
#         }
    