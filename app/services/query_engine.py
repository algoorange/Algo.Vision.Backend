import httpx
import logging
from app.utils.embeddings import embedder, embedding_index, embedding_metadata


async def semantic_search(prompt: str):
    query_embedding = embedder.encode(prompt)
    scores, indices = embedding_index.search(query_embedding[None, :], k=1)

    if scores and scores[0][0] > 0.4:  # lowered threshold for more matches
        metadata = embedding_metadata[indices[0][0]]
        return {
            "answer": metadata["summary"],
            "source_video": metadata["video"]
        }
    else:
        return {
            "answer": "No relevant video found.",
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
