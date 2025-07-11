import os
import httpx
import asyncio
from typing import Optional

def get_api_key() -> Optional[str]:
    """Get the API key from environment variables."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable is not set.")
        print("Please set it using:")
        print("1. For Windows (Command Prompt): set GROQ_API_KEY=your_api_key_here")
        print("2. For Windows (PowerShell): $env:GROQ_API_KEY='your_api_key_here'")
        print("3. For Linux/MacOS: export GROQ_API_KEY=your_api_key_here")
    return api_key

async def generate_summary(prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    """
    Generate a summary using Groq's API.
    
    Args:
        prompt: The input text to summarize
        model: The model to use (default: "llama-3.1-8b-instant")
        
    Returns:
        Generated summary or error message
    """
    api_key = get_api_key()
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

# Example usage
if __name__ == "__main__":
    prompt = """
    Summarize the following object movements:
    - ID: 1, Label: car, Actions: Moving right, then stopped
    - ID: 2, Label: person, Actions: Crossing the road
    - ID: 3, Label: truck, Actions: Moving left
    """
    
    summary = asyncio.run(generate_summary(prompt))
    print(summary)
