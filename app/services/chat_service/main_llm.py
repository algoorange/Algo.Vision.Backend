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
from app.services.chat_service.all_tools.all_object_tool import AllObjectToolService
from app.services.chat_service.all_tools.video_segment_tool import VideoSegmentToolService
from app.services.chat_service.all_tools.evidence_tool import EvidenceToolService
from app.services.chat_service.chathistory import ChatMemoryForLLMService, generate_chat_id

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["algo_compliance_db_2"]
collection = db["video_details"]

main_prompt = """
You are a helpful assistant for answering questions about a specific video.

- Always ground answers in database/tool outputs. Never guess or invent values.
- Persistent chat history is maintained for this session. When a question is a follow-up or uses references/pronouns (e.g., "that car", "previous frame"), first consult the chat history to resolve context and infer missing parameters (e.g., video_id, segment/frame ranges, object_type, color, track_id, time windows).
- Based on the inferred intent, choose and call the appropriate tool(s):
  - all_object_tool: object types, colors, track IDs, per-frame info, positions.
  - video_segment_tool: counts by segment/time windows, unique counts by color/type, track ranges.
  - evidence_tool: visual evidence (snapshots/clips) when the user asks for proof.
- STRICT TOOL-FIRST POLICY: For any request involving counts, analytics, movements, IDs, ranges, or database lookups, you MUST call a tool. Only summarize/explain after reading the tool output.
- If history does not fully disambiguate the request, ask a brief clarifying question before or after a minimal tool call.
- Be concise. When returning an answer from a tool call, summarize key results and explicitly state any parameters inferred from chat history.
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
        self.chatMemoryForLLMService = ChatMemoryForLLMService()

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
            # Use the provided session_id as the stable chat_id for session-based history
            chat_id = self.session_id or generate_chat_id()
            # Add user message to memory
            # self.memory.chat_memory.add_user_message(query)
            chathistory=(self.chatMemoryForLLMService.get_chat_history_in_LLM_feedable_form(chat_id))
            print("Chat history:", chathistory)  # Debug: Print full chat history

            # Build full message history from memory with correct roles
            # chat_history = self.memory.chat_memory.messages
            # memory_messages = [
            #     {"role": "system", "content": main_prompt}
            # ] + [
            #     {
            #         "role": "user" if isinstance(msg, HumanMessage) else "assistant",
            #         "content": msg.content
            #     }
            #     for msg in chat_history
            # ]

            # Send request to Groq with proper ordering: system -> history -> current user
            messages = [{"role": "system", "content": main_prompt}]
            if chathistory:
                # chathistory already contains alternating user/assistant messages
                messages += chathistory
            messages.append({"role": "user", "content": query})
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=video_tool_description
            )

            message = response.choices[0].message
            print("DEBUG LLM MESSAGE:", message)  # Debug: Print full LLM message

            # Add assistant message to memory (handle None content for tool calls)
            # if message.content:
            #     self.memory.chat_memory.add_ai_message(message.content)
            # else:
            #     # For tool calls, add a placeholder message
            #     self.memory.chat_memory.add_ai_message("Tool call executed")

            

            # Handle tools
            if hasattr(message, "tool_calls") and message.tool_calls:
                print("DEBUG TOOL CALLS:", message.tool_calls)  # Debug: Print tool call details
                result = await self.handle_tool_call(message, query)
                # Only add string/markdown to chat memory
                if isinstance(result, dict) and result.get("type") == "evidence":
                    # Optionally, add a summary string to chat history
                    self.chatMemoryForLLMService.add_message(chat_id, query, "Displayed evidence frames.")
                    return result
                else:
                    self.chatMemoryForLLMService.add_message(chat_id, query, result)
                   
                    return result

            self.chatMemoryForLLMService.add_message(chat_id,query,message.content)
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
            video_tool_service = AllObjectToolService(request=None)
            video_segment_tool_service = VideoSegmentToolService(request=None)
            evidence_tool_service = EvidenceToolService(request=None)
            if agent_name == "get_all_object_details":
                tool_result = await video_tool_service.get_all_object_details(args)
            elif agent_name == "show_evidence":
                tool_result = await evidence_tool_service.show_evidence(args)
                return tool_result
            elif agent_name == "get_video_segment_details":
                tool_result = await video_segment_tool_service.get_video_segment_details(args)
            elif agent_name == "fall_back":
                if "frame_number" in args and args["frame_number"] is not None:
                    try:
                        args["frame_number"] = int(args["frame_number"])
                    except (ValueError, TypeError):
                        return {"error": "frame_number must be an integer or convertible to int."}
                tool_result = await video_tool_service.fall_back(args)
            else:
                return {"error": f"Unknown agent name: {agent_name}"}
            if self.reformat_prompt:
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
            else:
                return tool_result
        except Exception as e:
            return str(e)        