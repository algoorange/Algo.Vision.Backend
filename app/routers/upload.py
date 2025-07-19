from fastapi import APIRouter, UploadFile, File
from app.services import video_processor
import os
import uuid


router = APIRouter()


# Ensure the uploads folder exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.get("/test-debug")
async def test_debug():
    """
    Test endpoint for debugging - will trigger breakpoint
    """
    breakpoint()  # 🔴 This will trigger the debugger
    message = "Debug test successful!"
    return {"message": message, "debug": True}

@router.post("/")
async def upload_video(file: UploadFile = File(...)):
    """
    Uploads the video and extracts frames from the uploaded video. Generates a unique video ID and stores frames per video.
    """
    breakpoint()  # 🔴 Add breakpoint here too
    video_id = str(uuid.uuid4())
    result = await video_processor.process_video(file, video_id)
    return {"video_id": video_id, **result}