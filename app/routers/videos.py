import os
from fastapi import APIRouter
from fastapi.responses import JSONResponse

UPLOAD_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'uploads')
)

router = APIRouter()

from fastapi import Query

@router.get("/frames/list", response_class=JSONResponse)
def list_frames(video_id: str = Query(..., description="Unique video ID")):
    """
    Returns a list of frame image filenames for a specific video.
    Looks in the 'frames/{video_id}/' folder and returns all .jpg files.
    """
    # Get the absolute path to the 'frames' directory
    frames_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'frames'))
    # Get the path to the specific video's frames folder
    video_frames_folder = os.path.join(frames_folder, video_id)

    # If the folder does not exist, return an empty list
    if not os.path.exists(video_frames_folder):
        return []

    # List all files in the folder that end with '.jpg' (case-insensitive)
    frame_files = []
    for filename in os.listdir(video_frames_folder):
        if filename.lower().endswith('.jpg'):
            frame_files.append(filename)

    # Sort the filenames (optional, for consistent order)
    frame_files.sort()
    return frame_files

@router.get("/list", response_class=JSONResponse)
def list_videos():
    """List all video files in the uploads directory."""
    if not os.path.exists(UPLOAD_DIR):
        return []
    files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
    return files
