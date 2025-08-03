from dotenv import load_dotenv
import groq
import os
from pymongo import MongoClient
from groq.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
)
from langchain.memory import ConversationBufferMemory
from app.services.chat_service.video_tool_discription import video_tool_description
from app.services.chat_service.video_tool_service import VideoToolService

load_dotenv()
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["algo_compliance_db_2"]
collection = db["video_details"]

main_prompt = """
You are a helpful assistant that can answer questions about the video.
"""

class ChatService:
    def __init__(self, video_tool_service=None, llm_client=None, model_name=None, reformat_prompt=None, memory=None):
        """
        ChatService for handling chat interactions and tool calls.
        Args:
            video_tool_service: Service for handling video analysis tool calls (must implement analyze_video_behavior, get_object_details, get_object_statistics)
            llm_client: Optional, client for LLM follow-up (defaults to groq.Client)
            model_name: Optional, model name for follow-up LLM calls
            reformat_prompt: Optional, prompt for result reformatting
            memory: Optional, ConversationBufferMemory for chat history
        """
        self.client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))
        self.video_tool_service = video_tool_service
        self.llm_client = llm_client or self.client
        self.model_name = model_name or "llama-3.3-70b-versatile"
        self.reformat_prompt = reformat_prompt or "Please reformat the tool result for the user."
        self.memory = memory or ConversationBufferMemory(return_messages=True)

    async def chat_query(self, query: str):
        """
        Sends a user query to the LLM and returns the response.
        Handles tool calls if present in the response.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    ChatCompletionSystemMessageParam(
                        role="system",
                        content=main_prompt
                    ),
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=query
                    ),
                ],
                tools=video_tool_description
            )
            message = response.choices[0].message
            # If LLM triggers a tool call, handle it
            if hasattr(message, "tool_calls") and message.tool_calls:
                return await self.handle_tool_call(message, query)
            return message.content
        except Exception as e:
            return str(e)

    async def handle_tool_call(self, message, user_query):
        """
        Handles LLM tool calls by dispatching to the appropriate video_tool_service method.
        """
        tool_call = message.tool_calls[0]
        agent_name = tool_call.function.name
        args = tool_call.function.arguments if hasattr(tool_call.function, "arguments") else {}
        tool_name = agent_name  # For message construction
        try:
            video_tool_service = VideoToolService(request=None)
            if agent_name == "get_all_object_details":
                tool_result = await video_tool_service.get_all_object_details(args)
            elif agent_name == "get_specific_object_type":
                tool_result = await video_tool_service.get_specific_object_type(args)
            elif agent_name == "get_traffic_congestion_details":
                tool_result = await video_tool_service.get_traffic_congestion_details(args)
            elif agent_name == "object_position_confidence":
                tool_result = await video_tool_service.object_position_confidence(args)
            elif agent_name == "get_all_object_direction":
                tool_result = await video_tool_service.get_all_object_direction(args)
            elif agent_name == "object_position_confidence_using_track_id":
                if "frame_number" in args and args["frame_number"] is not None:
                    try:    
                        args["frame_number"] = int(args["frame_number"])
                    except (ValueError, TypeError):
                        return {"error": "frame_number must be an integer or convertible to int."}
                tool_result = await video_tool_service.object_position_confidence_using_track_id(args)
            else:
                return {"error": f"Unknown agent name: {agent_name}"}

            # If no reformatting requested, return result directly
            if not tool_result.get("reformatting", True):
                return tool_result.get("result")

            # Otherwise, reformat using LLM
            import time
            retries = 3
            for attempt in range(retries):
                try:
                    follow_up = self.llm_client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            ChatCompletionSystemMessageParam(
                                role="system",
                                content=self.reformat_prompt
                            ),
                            ChatCompletionUserMessageParam(
                                role="user",
                                content=f"Current user query is : {user_query}"
                            ),
                            ChatCompletionAssistantMessageParam(
                                role="assistant",
                                tool_calls=[tool_call]
                            ),
                            ChatCompletionToolMessageParam(
                                role="tool",
                                tool_call_id=tool_call.id,
                                name=tool_name,
                                content=str(tool_result.get("result"))
                            ),
                        ],
                        max_tokens=1000,
                        timeout=60  # Set timeout to 60 seconds
                    )
                    result = follow_up.choices[0].message
                    return result.content
                except Exception as e:
                    if 'timed out' in str(e).lower() and attempt < retries - 1:
                        time.sleep(2)  # Wait before retrying
                        continue
                    return str(e)
        except Exception as e:
            return str(e)
