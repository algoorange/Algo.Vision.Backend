import os
import httpx
import asyncio
from typing import Optional, Dict, Any, List

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


# ---------------- Segment helpers ----------------
def build_segment_summary_prompt(segment: Dict[str, Any]) -> str:
    """Build a concise, structured prompt for LLM from a segment document.

    Expected keys in segment:
      - summary: { start_time, end_time }
      - object_counts: { type: count }
      - frame_count: int
      - objects: list of filtered objects (may include color, start/end info)
    """
    summary = segment.get("summary", {})
    start_time = summary.get("start_time")
    end_time = summary.get("end_time")
    frame_count = segment.get("frame_count", 0)
    object_counts = segment.get("object_counts", {})

    # Lightly sample object examples to keep prompt short
    objs: List[Dict[str, Any]] = segment.get("objects", [])
    sample = objs[:8]  # up to 8 exemplars
    example_lines = []
    for o in sample:
        parts = []
        if o.get("object_type"):
            parts.append(f"type: {o['object_type']}")
        if o.get("color") and o.get("color") != "unknown":
            parts.append(f"color: {o['color']}")
        if o.get("frame_time") is not None:
            parts.append(f"first_seen_time: {o['frame_time']}")
        if o.get("start_time") is not None and o.get("end_time") is not None:
            parts.append(f"track_window: {o['start_time']}–{o['end_time']}")
        example_lines.append("- " + ", ".join(parts))

    object_counts_str = "; ".join([f"{k}: {v}" for k, v in object_counts.items()]) or "none"
    examples_str = "\n".join(example_lines) if example_lines else "- (no exemplars)"

    prompt = f"""
You are a vision analysis assistant. Write a concise, descriptive summary for a video segment suitable for a human reader.

Segment window: {start_time}s to {end_time}s
Frames in segment: {frame_count}
Object counts (unique tracks): {object_counts_str}
Example objects:\n{examples_str}

Guidelines:
- Describe the overall scene in one or two sentences.
- Mention the main object categories and relative amounts.
- Note colors only when salient.
- Briefly mention movement direction or notable actions if inferable.
- Avoid speculation; do not invent details.
- Keep to 1–3 sentences maximum.
""".strip()
    return prompt


async def generate_segment_description(segment: Dict[str, Any], model: str = "llama-3.1-8b-instant") -> str:
    """Create a natural-language description for a segment using Groq LLM."""
    prompt = build_segment_summary_prompt(segment)
    return await generate_summary(prompt, model=model)

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
