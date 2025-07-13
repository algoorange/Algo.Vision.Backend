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
    """List all frame image filenames for a specific video in frames/{video_id}/."""
    frames_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'frames'))
    frames_dir = os.path.join(frames_base, video_id)
    if not os.path.exists(frames_dir):
        return []
    files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith('.jpg')])
    return files

@router.get("/list", response_class=JSONResponse)
def list_videos():
    """List all video files in the uploads directory."""
    if not os.path.exists(UPLOAD_DIR):
        return []
    files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
    return files
