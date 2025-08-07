from fastapi import APIRouter, Body
from app.services import query_engine
from app.utils.embeddings import embedder, embedding_index, embedding_metadata
from app.services.chat_service.main_llm import ChatService
import asyncio
from pymongo import MongoClient


# 1. Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["algo_compliance_db_2"]
video_details = db["video_details"]

router = APIRouter()

# @router.post("/")
async def query_llm(data: dict = Body(...)):
    question = data.get("question")

    if not question:
        return {"error": "Question is required"}

    #Embed user query and find most relevant video summary    
    semantic_result = await query_engine.semantic_search(question)

    combined_prompt = (
        f"User Question: {question}\n"
        f"Relevant Video Summary: {semantic_result['answer']}\n"
        "Give a clear and concise answer using the semantic search result above."
    )

    #llm_response = await query_engine.query_ollama(combined_prompt)
    #use GROQ model here 
    llm_response = await query_engine.check_with_groq(combined_prompt, model="llama-3.1-8b-instant")

    return {
        "answer": llm_response,
        "source_video": semantic_result["source_video"]
    }

# New endpoint for direct LLM chat
from uuid import uuid4

@router.post("/")
async def llm_chat(data: dict = Body(...)):
    question = data.get("question")
    session_id = data.get("session_id") or str(uuid4())
    if not question:
        return {"error": "Question is required"}
    chat_service = ChatService(session_id=session_id)
    result = await chat_service.chat_query(question)
    print("final result ", result)
    if isinstance(result, dict) and result.get("type") == "evidence":
        return {"answer": result, "session_id": session_id}
    else:
        return {"answer": result, "session_id": session_id}







# @router.post("/evidance_frame")
# async def evidance_frame(data: dict = Body(...)):
#     video_id = data.get("video_id")
#     frame_number = data.get("frame_number")
#     track_id = data.get("track_id")

#     if not video_id or not frame_number or not track_id:
#         return {"error": "video_id, frame_number, and track_id are required"}

#     # 2. Find the video document
#     video_doc = video_details.find_one({"video_id": video_id})
#     if not video_doc:
#         return {"error": "Video not found"}

#     # 3. Find the frame with the given frame_number
#     frame = next((f for f in video_doc["frames"] if f["frame_number"] == int(frame_number)), None)
#     if not frame:
#         return {"error": "Frame not found"}

#     # 4. Find the object with the given track_id
#     obj = next((o for o in frame["objects"] if o["track_id"] == str(track_id)), None)
#     if not obj:
#         return {"error": "Object not found in frame"}

#     # 5. Build the image path (adjust as needed for your server setup)
#     frame_id = frame["frame_id"]
#     image_path = f"/frames/{video_id}/{frame_id}.jpg"  # This should be a URL if serving static files

#     # 6. Return the evidence
#     return {
#         "image_url": image_path,
#         "frame_number": frame["frame_number"],
#         "frame_time": frame.get("frame_time"),
#         "object": obj,
#     }

