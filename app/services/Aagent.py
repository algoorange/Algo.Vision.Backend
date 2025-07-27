import os
import json
from pydantic import BaseModel, Field
from pymongo import MongoClient
from app.services.query_engine import semantic_search
from app.services.chromadb_service import chromadb_service

try:
    from dotenv import load_dotenv
    load_dotenv()

    MODEL_NAME = os.getenv("GOOGLE_GENAI_MODEL","gemini-2.0-flash")
    API_KEY = "AIzaSyAatrQTxYHF8g09zi3smeycS2xtYlVOJDw"
    
    # Validate API key
    if not API_KEY:
        raise ValueError("GOOGLE_API_KEY_2 environment variable is not set")
    
    print(f"✅ Model: {MODEL_NAME}")
    print(f"✅ API Key: {API_KEY[:10]}..." if API_KEY else "❌ No API Key")
    
except Exception as e:
    print(f"Error loading environment variables: {e}")
    API_KEY = None
    MODEL_NAME = "gemini-2.0-flash"

from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.adk.planners import PlanReActPlanner
from google.adk.models.lite_llm import LiteLlm

groq_llm = LiteLlm(
    model="groq/llama-3.3-70b-versatile",          # MUST include the "groq/" prefix
    api_key=os.getenv("GROQ_API_KEY"),
    api_base="https://api.groq.com/openai/v1"
)   

def chromadb_semantic_search(query: str, search_type: str = "both", n_results: int = 5):
    """
    Search ChromaDB for video analysis and frame data based on user query
    
    Args:
        query: User's search query
        search_type: "videos", "frames", or "both"
        n_results: Number of results to return
    
    Returns:
        Dictionary with search results
    """
    try:
        results = {}
        
        if search_type in ["videos", "both"]:
            video_results = chromadb_service.search_videos(query, n_results)
            if video_results:
                results["videos"] = {
                    "documents": video_results.get("documents", []),
                    "metadatas": video_results.get("metadatas", []),
                    "distances": video_results.get("distances", [])
                }
        
        if search_type in ["frames", "both"]:
            frame_results = chromadb_service.search_frames(query, n_results)
            if frame_results:
                results["frames"] = {
                    "documents": frame_results.get("documents", []),
                    "metadatas": frame_results.get("metadatas", []),
                    "distances": frame_results.get("distances", [])
                }
        
        return {
            "success": True,
            "query": query,
            "search_type": search_type,
            "results": results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query
        }
def video_data_using_mongodb(video_id: str):
    """
    Get video data from MongoDB using the video ID
    
    Args:
        video_id: ID of the video to retrieve
    """

    try:
        mongo_client = MongoClient("mongodb://localhost:27017/")
        db = mongo_client["algo_compliance_db_2"]
        video_details_collection = db["video_details"]
        video_data = video_details_collection.find_one({"video_id": video_id},{"_id":0})
        return video_data
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "video_id": video_id
        }


# Create the agent with proper configuration according to ADK docs
if API_KEY:
    Google_ADK_Agent = LlmAgent(
        name="google_adk_agent",
        model=groq_llm,
        planner=PlanReActPlanner(),
        description="A vision assistant that can analyze video data and answer questions using ChromaDB semantic search",
        instruction="""You are a vision assistant that can analyze video data and answer questions about video content.

        When a user asks a question about videos or video analysis:
        1. Understand the user's query about video content, objects, or activities
        2. Use the video_data_using_mongodb tool to find relevant video analysis data
        3. Analyze the search results to provide accurate answers
        4. Provide detailed responses based on the video analysis data
        
        You have access to:
        - video_data_using_mongodb: Search for video analysis and frame data
        - if data is not enough in video_data_using_mongodb serach then use the mogno db analysis and answer for the question
        
        Focus on providing accurate information about video content, detected objects, activities, and analysis results.
        """,
        tools=[video_data_using_mongodb],  # Include both tools
        output_key="google_adk_agent_response"
    )
    print("✅ Google ADK Agent created successfully with ChromaDB search capability")
else:
    print("❌ Cannot create Google ADK Agent - no API key available")
    Google_ADK_Agent = None

# Export the agent as root_agent
root_agent = Google_ADK_Agent

INSTRUCTIONS = """
You are a vision assistant. Your job is to correctly analyze the data from the tools, get the proper data, and answer user questions about video content, detected objects, and video analysis results.
"""
