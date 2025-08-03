from dotenv import load_dotenv
import os
import time
from pymongo import MongoClient
import groq

from groq.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
)

from langchain.memory import ConversationSummaryBufferMemory
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage

from app.services.chat_service.video_tool_discription import video_tool_description
from app.services.chat_service.video_tool_service import VideoToolService

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["algo_compliance_db_2"]
collection = db["video_details"]

main_prompt = """
You are a helpful assistant that can answer questions about the video.
If the user asks about object counts, analytics, movements, or any question that requires database or video analysis, ALWAYS use the available tools/functions provided. Do NOT try to answer from your own knowledge—use a tool call for anything requiring data lookup or analytics.
"""

import threading

_session_memories = {}
_session_lock = threading.Lock()

def get_session_memory(session_id, llm):
    with _session_lock:
        if session_id not in _session_memories:
            _session_memories[session_id] = ConversationSummaryBufferMemory(
                llm=llm,
                max_token_limit=200,
                return_messages=True
            )
        return _session_memories[session_id]

class ChatService:
    def __init__(self, session_id, video_tool_service=None, llm_client=None, model_name=None, reformat_prompt=None):
        """
        ChatService for handling chat interactions and tool calls, with session-based memory.
        """
        self.session_id = session_id
        self.client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))
        self.video_tool_service = video_tool_service
        self.llm_client = llm_client or self.client
        self.model_name = model_name or "llama-3.3-70b-versatile"
        self.reformat_prompt = reformat_prompt or "Please reformat the tool result for the user."
        self.memory = get_session_memory(
            session_id=session_id,
            llm=self.client_wrapper()
        )

    def client_wrapper(self):
        """
        Wraps the Groq client into a LangChain-compatible ChatGroq client.
        """
        return ChatGroq(
            api_key=api_key,
            model_name=self.model_name,
        )

    async def chat_query(self, query: str):
        """
        Sends a user query to the LLM and returns the response.
        Handles tool calls if present in the response.
        """
        try:
            # Add user message to memory
            self.memory.chat_memory.add_user_message(query)

            # Build full message history from memory with correct roles
            chat_history = self.memory.chat_memory.messages
            memory_messages = [
                {"role": "system", "content": main_prompt}
            ] + [
                {
                    "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                    "content": msg.content
                }
                for msg in chat_history
            ]

            # Send request to Groq
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=memory_messages,
                tools=video_tool_description
            )

            message = response.choices[0].message
            print("DEBUG LLM MESSAGE:", message)  # Debug: Print full LLM message

            # Add assistant message to memory (handle None content for tool calls)
            if message.content:
                self.memory.chat_memory.add_ai_message(message.content)
            else:
                # For tool calls, add a placeholder message
                self.memory.chat_memory.add_ai_message("Tool call executed")

            # Handle tools
            if hasattr(message, "tool_calls") and message.tool_calls:
                print("DEBUG TOOL CALLS:", message.tool_calls)  # Debug: Print tool call details
                return await self.handle_tool_call(message, query)

            return message.content or "No answer generated."
        except Exception as e:
            print("DEBUG EXCEPTION:", e)
            return str(e)

    async def handle_tool_call(self, message, user_query):
        """
        Handles LLM tool calls by dispatching to the appropriate video_tool_service method.
        """
        tool_call = message.tool_calls[0]
        agent_name = tool_call.function.name
        args = tool_call.function.arguments if hasattr(tool_call.function, "arguments") else {}

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
            elif agent_name == "get_video_segment_details":
                tool_result = await video_tool_service.get_video_segment_details(args)
            elif agent_name == "object_position_confidence_using_track_id":
                if "frame_number" in args and args["frame_number"] is not None:
                    try:
                        args["frame_number"] = int(args["frame_number"])
                    except (ValueError, TypeError):
                        return {"error": "frame_number must be an integer or convertible to int."}
                tool_result = await video_tool_service.object_position_confidence_using_track_id(args)
            else:
                return {"error": f"Unknown agent name: {agent_name}"}

            # Always reformat using LLM for user-friendly output
            reformat_prompt = (
                self.reformat_prompt or
                "Please reformat the tool result for the user."
            )
            reformat_prompt += (
                "\nYour job is to present the tool result in a clear, concise, and user-friendly way. "
                "Do not include raw JSON or technical fields. Use natural language and summarize the most important information for the user. "
                "Format your answer using advanced Markdown for clarity and beauty: use bullet points, numbered lists, tables, headings, bold, italics where appropriate. ignore the code blocks, and LaTeX math blocks."
                "Make the output visually appealing and easy to read."
                "show only the final answer, do not show the technical fields and technical results."
            )

            retries = 3
            for attempt in range(retries):
                try:
                    follow_up = self.llm_client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            ChatCompletionSystemMessageParam(
                                role="system",
                                content=reformat_prompt
                            ),
                            ChatCompletionUserMessageParam(
                                role="user",
                                content=f"User Question: {user_query}\nRaw Tool Result: {tool_result}"
                            )
                        ],
                        max_tokens=1000,
                        timeout=60
                    )
                    result = follow_up.choices[0].message
                    # Store reformatted assistant message in memory
                    if result.content:
                        self.memory.chat_memory.add_ai_message(result.content)
                        return result.content
                    else:
                        self.memory.chat_memory.add_ai_message("Tool result processed")
                        return "Tool result processed"
                except Exception as e:
                    if 'timed out' in str(e).lower() and attempt < retries - 1:
                        time.sleep(2)
                        continue
                    return str(e)
        except Exception as e:
            return str(e)