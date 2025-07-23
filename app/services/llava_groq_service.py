import os
import cv2
import base64
import httpx
import asyncio
from typing import Optional, List, Dict, Any
import numpy as np

def get_api_key() -> Optional[str]:
    """Get the Groq API key from environment variables."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable is not set.")
        print("Please set it using:")
        print("1. For Windows (Command Prompt): set GROQ_API_KEY=your_api_key_here")
        print("2. For Windows (PowerShell): $env:GROQ_API_KEY='your_api_key_here'")
        print("3. For Linux/MacOS: export GROQ_API_KEY=your_api_key_here")
    return api_key

def encode_image_to_base64(image: np.ndarray) -> str:
    """
    Encode OpenCV image (numpy array) to base64 string.
    
    Args:
        image: OpenCV image as numpy array
        
    Returns:
        Base64 encoded string of the image
    """
    # Encode image to JPEG format
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    # Convert to base64
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64

async def analyze_frame_with_llava(image: np.ndarray, prompt: str, model: str = "meta-llama/llama-4-maverick-17b-128e-instruct") -> str:
    """
    Analyze a single frame using LLaVA via Groq API.
    
    Args:
        image: OpenCV image as numpy array
        prompt: Text prompt for the vision analysis
        model: Groq model to use (default: Llama 4 Maverick multimodal)
        
    Returns:
        Analysis result from LLaVA model
    """
    api_key = get_api_key()
    if not api_key:
        return "Error: API key not configured. Please set the GROQ_API_KEY environment variable."

    # Encode image to base64
    image_base64 = encode_image_to_base64(image)
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Prepare the message with image and text
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
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

async def analyze_video_frames_with_llava(frames: List[np.ndarray], prompts: List[str], model: str = "meta-llama/llama-4-maverick-17b-128e-instruct") -> List[str]:
    """
    Analyze multiple video frames using LLaVA via Groq API.
    
    Args:
        frames: List of OpenCV images as numpy arrays
        prompts: List of text prompts for each frame analysis
        model: Groq model to use (default: Llama 4 Maverick multimodal)
        
    Returns:
        List of analysis results from LLaVA model
    """
    if len(frames) != len(prompts):
        raise ValueError("Number of frames must match number of prompts")
    
    # Process frames concurrently for better performance
    tasks = []
    for frame, prompt in zip(frames, prompts):
        task = analyze_frame_with_llava(frame, prompt, model)
        tasks.append(task)
    
    # Wait for all analyses to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Convert exceptions to error strings
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            processed_results.append(f"Error: {str(result)}")
        else:
            processed_results.append(result)
    
    return processed_results

def create_default_prompts(num_frames: int) -> List[str]:
    """
    Create default prompts for video frame analysis.
    
    Args:
        num_frames: Number of frames to create prompts for
        
    Returns:
        List of default prompts
    """
    base_prompts = [
        "Describe what you see in this video frame. Focus on objects, people, and their activities.",
        "What objects and people are visible in this frame? Describe their positions and actions.",
        "Analyze this video frame and describe any movement or activity you can observe.",
        "What is happening in this frame? Describe the scene, objects, and any notable details.",
        "Examine this frame and identify all visible objects, people, and their interactions."
    ]
    
    # Cycle through base prompts to cover all frames
    prompts = []
    for i in range(num_frames):
        prompt_index = i % len(base_prompts)
        frame_number = i + 1
        prompt = f"Frame {frame_number}: {base_prompts[prompt_index]}"
        prompts.append(prompt)
    
    return prompts

async def generate_video_summary_with_llava(frame_analyses: List[str], video_metadata: Dict[str, Any]) -> str:
    """
    Generate a comprehensive video summary using LLaVA frame analyses.
    
    Args:
        frame_analyses: List of individual frame analysis results
        video_metadata: Video metadata (duration, fps, etc.)
        
    Returns:
        Comprehensive video summary
    """
    # Combine all frame analyses
    combined_analysis = "\n\n".join([f"Analysis {i+1}: {analysis}" for i, analysis in enumerate(frame_analyses)])
    
    # Create summary prompt
    summary_prompt = f"""
Based on the following frame-by-frame analysis of a video, provide a comprehensive summary:

Video Duration: {video_metadata.get('duration', 'Unknown')} seconds
Frame Rate: {video_metadata.get('fps', 'Unknown')} FPS
Number of analyzed frames: {len(frame_analyses)}

Frame Analyses:
{combined_analysis}

Please provide a detailed summary that includes:
1. Overall scene description
2. Key objects and people identified
3. Main activities and movements observed
4. Timeline of events if applicable
5. Any patterns or interesting observations

Summary:
"""

    # Use text-only model for summary generation
    api_key = get_api_key()
    if not api_key:
        return "Error: API key not configured for summary generation."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.3-70b-versatile",  # Use text model for summary
        "messages": [{"role": "user", "content": summary_prompt}],
        "temperature": 0.7,
        "max_tokens": 1500
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            
            if response.status_code == 401:
                return "Error: Invalid API key for summary generation."
                
            response.raise_for_status()
            data = response.json()
            
            if not data.get("choices"):
                return "Error: No response from summary API."
                
            content = data["choices"][0].get("message", {}).get("content", "")
            return content.strip() or "Unable to generate summary."
            
    except httpx.HTTPStatusError as e:
        return f"Summary HTTP Error: {str(e)}"
    except Exception as e:
        return f"Summary Error: {str(e)}" 