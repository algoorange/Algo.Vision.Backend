from fastapi import APIRouter, UploadFile, File
from app.services import video_processor
from app.services import video_streaming
import os


router = APIRouter()


# Ensure the uploads folder exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/")
async def upload_video(file: UploadFile = File(...)):
    """
    Uploads the video and starts streaming the processed frames
    """
    # Save video temporarily and stream processed frames
    result = await video_processor.process_video(file)
    return result


# # Stream video frames with object detection
# @router.get("/{filename}")
# async def stream_video(filename: str):
#     """
#     Stream processed video frames
#     """
#     return video_streaming.stream_video(filename)    