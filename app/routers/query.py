from fastapi import APIRouter, Body
from app.services import query_engine
from app.services.chat_service.main_llm import ChatService
from app.services.chat_service.video_tool_service import VideoToolService

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
@router.post("/")
async def llm_chat(data: dict = Body(...)):
    question = data.get("question")
    if not question:
        return {"error": "Question is required"}
    user_id = "user1"  # TODO: Replace with real user/session ID in production
    video_tool_service = VideoToolService(request=None)
    memory = video_tool_service.get_chat_memory(user_id)
    chat_service = ChatService(memory=memory)
    result = await chat_service.chat_query(question)
    video_tool_service.set_chat_memory(user_id, chat_service.memory)
    return {"answer": result}
