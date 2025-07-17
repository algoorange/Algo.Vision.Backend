import os
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pymongo import MongoClient
from fastapi import Body, Query
from typing import Optional

UPLOAD_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'uploads')
)

router = APIRouter()

# Connect to MongoDB (adjust the URI as needed)
client = MongoClient("mongodb://localhost:27017/")
db = client["algo_compliance_db_2"]  # Your database name
chat_collection = db["user_chat_history"]  # Your collection name
video_details_collection = db["video_details"]  # Your collection name

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


from typing import Optional
from fastapi import Body

@router.post("/user_chat_history", response_class=JSONResponse)
def save_user_chat_history(
    video_id: str = Body(...),
    summary: Optional[str] = Body(None),
    role: str = Body(...),  # "user" or "bot"
    text: str = Body(...)
):
    """
    Save a chat message for a specific video.
    If summary is provided and not empty, update it; otherwise, leave it unchanged.
    """
    update_fields = {
        "$push": {"messages": {"role": role, "text": text}}
    }
    if summary:
        update_fields["$set"] = {"summary": summary}

    chat_collection.update_one(
        {"video_id": video_id},
        update_fields,
        upsert=True
    )
    return {"success": True}


@router.get("/user_chat_history_fetch", response_class=JSONResponse)
def fetch_user_chat_history(video_id: str = Query(..., description="Unique video ID")):
    print("Fetching chat for video_id:", video_id)
    """
    Fetch chat history for a specific video.
    Returns a dict with 'messages' and optional 'summary'.
    """
    doc = chat_collection.find_one({"video_id": video_id}, {"_id": 0})  # Exclude MongoDB's _id
    if doc:
        print("Found chat history:", doc)
        return doc
    else:
        print("No chat history found for video_id:", video_id)
        return {"messages": []}


@router.get("/frame_details", response_class=JSONResponse)
def get_frame_details(
    video_id: str = Query(...),
    frame_id: str = Query(...)
):
    frame_id_split = frame_id.split('.')[0]
    docs = video_details_collection.find(
        {"video_id": video_id, "frames_id": frame_id_split},
        {"_id": 0}
    )
    docs_list = []
    for doc in docs:
        docs_list.append(doc)
    if docs_list:
        return docs_list
    return {"error": "Frame not found"}



