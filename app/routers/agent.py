# from fastapi import APIRouter, HTTPException, Body
# import httpx
# from pydantic import BaseModel
# from typing import Optional, Dict, Any
# import torch
# from app.services import object_detector
# import httpx
# import logging
# from app.utils.embeddings import embedder, embedding_index, embedding_metadata
# from app.services.google_adk_runner_service import get_google_adk_runner
# import os

# router = APIRouter()


# @router.post("/")
# async def google_adk_agent(data: dict = Body(...)):
     
#     try:
#         google_adk_runner = get_google_adk_runner()
#         question = data.get("question")
#         user_id= "samit007"
    

#         result = await google_adk_runner.execute_agent(user_id, question)

#         if result.get("success"):
#                 return {
#                     "answer": result.get("response"),
#                 }
#         else:
#                 return {
#                     "error": result.get("error", "Google ADK agent execution failed"),
#                     "system": "google_adk_agent"
#                 }
#     except Exception as e:
#         logging.error(f"Error in google_adk_agent: {str(e)}")
#         # Fallback to GROQ API if google_adk_agent fails
#         prompt = "Hello, how are you?"
#         model = "llama-3.1-8b-instant"
#         api_key = os.getenv("GROQ_API_KEY")
#         if not api_key:
#             return "Error: API key not configured. Please set the GROQ_API_KEY environment variable."

#         url = "https://api.groq.com/openai/v1/chat/completions"
#         headers = {
#             "Authorization": f"Bearer {api_key}",
#             "Content-Type": "application/json"
#         }

#         payload = {
#             "model": model,
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": 0.7,
#             "max_tokens": 1000
#         }

#         try:
#             async with httpx.AsyncClient(timeout=120.0) as client:
#                 response = await client.post(url, json=payload, headers=headers)

#                 if response.status_code == 401:
#                     return "Error: Invalid API key. Please check your GROQ_API_KEY environment variable."

#                 response.raise_for_status()
#                 data = response.json()

#                 # Get the response content
#                 if not data.get("choices"):
#                     return "Error: No response from the API. Please try again later."

#                 content = data["choices"][0].get("message", {}).get("content", "")
#                 return content.strip() or "No content generated"

#         except httpx.HTTPStatusError as http_err:
#             return f"HTTP Error: {str(http_err)}"
#         except Exception as ex:
#             return f"Error: {str(ex)}"
        

