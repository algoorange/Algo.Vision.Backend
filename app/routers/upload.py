from fastapi import APIRouter, UploadFile, File
from app.services import video_processor
import os
import uuid


router = APIRouter()


# Ensure the uploads folder exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

from fastapi import Form
import json

@router.post("/")
async def upload_video(
    file: UploadFile = File(...),
    zoneCoords: str = Form(None)  # Receive zoneCoords as an optional string
):
    """
    Uploads the video and extracts frames from the uploaded video. Generates a unique video ID and stores frames per video.
    If zoneCoords is provided, it contains the restricted zone coordinates as a JSON string.
    """
    video_id = str(uuid.uuid4())

    # Parse the coordinates if present
    coords = None
    if zoneCoords:
        try:
            coords = json.loads(zoneCoords)
        except Exception as e:
            coords = None  # Optionally log error or handle as needed

    # Pass the coordinates to the video processor (update its signature if needed)
    result = await video_processor.process_video(file, video_id, coords)
    return {"video_id": video_id, "result": result}