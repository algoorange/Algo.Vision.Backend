from fastapi import APIRouter, Body
from app.services import query_engine
from app.utils.embeddings import embedder, embedding_index, embedding_metadata

router = APIRouter()

@router.post("/")
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
