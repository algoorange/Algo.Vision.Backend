import httpx
import logging
from app.utils.embeddings import embedder, embedding_index, embedding_metadata
import os

async def semantic_search(prompt: str):
    try:
        # Check if embedding index is empty
        if embedding_index.ntotal == 0:
            return {
                "answer": "No video data available for search. Please upload some videos first.",
                "source_video": None
            }
        
        query_embedding = embedder.encode(prompt)
        scores, indices = embedding_index.search(query_embedding[None, :], k=1)

        # Check if we got any results
        if len(indices[0]) == 0 or indices[0][0] == -1:
            return {
                "answer": "No relevant video found for your query.",
                "source_video": None
            }

        if scores and scores[0][0] > 0.4:  # lowered threshold for more matches
            # Safely access the metadata
            if indices[0][0] < len(embedding_metadata):
                metadata = embedding_metadata[indices[0][0]]
                return {
                    "answer": metadata["summary"],
                    "source_video": metadata["video"]
                }
            else:
                return {
                    "answer": "Video data found but metadata is incomplete.",
                    "source_video": None
                }
        else:
            return {
                "answer": "No relevant video found with sufficient similarity.",
                "source_video": None
            }
    except Exception as e:
        logging.error(f"Error in semantic_search: {e}")
        return {
            "answer": f"Error during search: {str(e)}",
            "source_video": None
        }

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def query_ollama(prompt: str, model: str = "llama3:8b"):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}]
    }

    logger.info("Sending request to Ollama API with prompt: %s", prompt)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()

                final_response = ""
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = httpx.Response(200, content=line).json()
                            content_piece = data.get("message", {}).get("content", "")
                            final_response += content_piece
                        except Exception as e:
                            logger.warning("Skipping malformed chunk: %s", line)

                return final_response or "No content returned."

    except httpx.HTTPStatusError as e:
        logger.error("HTTP error occurred: %s", e)
        return f"HTTP error occurred: {str(e)}"
    except httpx.RequestError as e:
        logger.error("Request error occurred: %s", e)
        return f"Request error occurred: {str(e)}"
    except Exception as e:
        logger.error("Unexpected error occurred: %s", e)
        return f"Unexpected error occurred: {str(e)}"


async def check_with_groq(prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    """
    Generate a summary using Groq's API.
    
    Args:
        prompt: The input text to summarize
        model: The model to use (default: "llama-3.1-8b-instant")
        
    Returns:
        Returns:
            Generated summary or error message
        """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: API key not configured. Please set the GROQ_API_KEY environment variable."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            
            if response.status_code == 401:
                return "Error: Invalid API key. Please check your GROQ_API_KEY environment variable."
                
            response.raise_for_status()
            data = response.json()
            
            # Get the response content
            if not data.get("choices"):
                return "Error: No response from the API. Please try again later."
                
            content = data["choices"][0].get("message", {}).get("content", "")
            return content.strip() or "No content generated"
            
    except httpx.HTTPStatusError as e:
        return f"HTTP Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"
