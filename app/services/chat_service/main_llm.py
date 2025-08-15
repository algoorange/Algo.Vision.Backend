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
from app.services.chat_service.all_tools.fall_back_tool import FallBackToolService
from app.services.chat_service.all_tools.generic_responce_tool import GenericResponceToolService
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
  - generic_responce_tool: generic/overview questions that are not related to the vision project and the user uploaded video. Examples of generic questions may include:
    - Greetings (e.g. "Hi there", "hello", "hi", "hey", etc...)
    - Introductions (e.g. "I'm John Doe", "My name is Jane Doe", etc...)
    - Small talk (e.g. "How are you doing?", "What's up?", etc...)
    - Jokes or humor (e.g. "Knock knock", "Why was the math book sad?", etc...)
    - Random questions (e.g. "What's the weather like today?", "What's your favorite color?", etc...)
    - Conversation starters (e.g. "How was your day?", "What did you do today?", etc...)
    - Goodbyes (e.g. "Goodbye", "See you later", "Bye", etc...)
    CALL THIS TOOL ONLY IF THE QUERY IS NOT RELATED TO THE VISION PROJECT AND THE USER UPLOADED VIDEO.
- For generic/overview questions (e.g., "What's happening in the video?"), CALL THE FALLBACK TOOL to synthesize an overall summary by combining all segment descriptions in order and aggregating object counts across segments. Return a concise overall summary plus brief per-segment notes when helpful.
- STRICT TOOL-FIRST POLICY: For any request involving counts, analytics, movements, IDs, ranges, or database lookups, you MUST call a tool. Only summarize/explain after reading the tool output.
- If history does not fully disambiguate the request, ask a brief clarifying question before or after a minimal tool call.
- Be concise. When returning an answer from a tool call, summarize key results and explicitly state any parameters inferred from chat history.

- Fallback behavior:
  1. If no existing tool applies to the user's question, if the question is unrelated to the project scope, or if there is no tool available for the request, CALL THE FALLBACK FUNCTION. Use it to provide a safe, concise response and optionally ask for clarification or suggest supported queries. Do not fabricate analytics without a tool call. 

- HANDLING TOOL INTERACTIONS
            - If the user's message is a greeting or social pleasantry, respond warmly **without calling any tool**.
            - Use tools only when the user input or prior history indicates vision project-related content (e.g., object types, colors, track IDs, video ID, segment/frame ranges, object_type, color, track_id, time windows).
            - Ensure all required inputs are present.
            - Never display tool calls as raw text—always execute and interpret results.
            - If an error occurs, respond with constructive suggestions (e.g., missing fields, formatting issues).
            ---

- HANDLING TOOL INTERACTIONS
            - If the user's message is a greeting or social pleasantry, respond warmly **without calling any tool**.
            - Use tools only when the user input or prior history indicates vision project-related content (e.g., object types, colors, track IDs, video ID, segment/frame ranges, object_type, color, track_id, time windows).
            - Ensure all required inputs are present.
            - Never display tool calls as raw text—always execute and interpret results.
            - If an error occurs, respond with constructive suggestions (e.g., missing fields, formatting issues).
            ---
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
            chat_id = self.session_id or generate_chat_id()
            chathistory = self.chatMemoryForLLMService.get_chat_history_in_LLM_feedable_form(chat_id)
            print("Chat history:", chathistory)

            # Base messages
            base_messages = [{"role": "system", "content": main_prompt}]
            if chathistory:
                base_messages += chathistory
            base_messages.append({"role": "user", "content": query})

            max_llm_retries = 3
            last_error_text = None

            for attempt in range(max_llm_retries):
                messages = list(base_messages)
                if attempt > 0:
                    corrective_note = (
                        "Previous response failed or did not call a tool. "
                        "You MUST call the correct tool with minimal, correct arguments inferred from chat history. "
                        "Do not answer directly without a tool call for analytics."
                    )
                    messages.insert(0, {"role": "system", "content": corrective_note})
                    if last_error_text:
                        messages.append({"role": "assistant", "content": f"Last error: {last_error_text}"})

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=video_tool_description
                )

                message = response.choices[0].message
                print("DEBUG LLM MESSAGE:", message)

                # If tool call present, try to execute
                if hasattr(message, "tool_calls") and message.tool_calls:
                    print("DEBUG TOOL CALLS:", message.tool_calls)
                    tool_result = await self.handle_tool_call(message, query)

                    # Detect tool failure; if so, retry prompting LLM to correct args
                    if self._is_failure_output(tool_result):
                        last_error_text = tool_result if isinstance(tool_result, str) else str(tool_result)
                        if attempt < max_llm_retries - 1:
                            time.sleep(1)
                            continue
                    # Success path: store and return
                    if isinstance(tool_result, dict) and tool_result.get("type") == "evidence":
                        self.chatMemoryForLLMService.add_message(chat_id, query, "Displayed evidence frames.")
                        return tool_result
                    else:
                        self.chatMemoryForLLMService.add_message(chat_id, query, tool_result)
                        return tool_result

                # If message indicates failure or lacks tool use for an analytical query, retry
                if not message.content or self._is_failure_text(message.content):
                    last_error_text = message.content or "Empty response"
                    if attempt < max_llm_retries - 1:
                        time.sleep(1)
                        continue
                # Accept final non-tool answer only as last resort
                self.chatMemoryForLLMService.add_message(chat_id, query, message.content)
                return message.content or "No answer generated."

            # Fallback if loop exits without return
            return last_error_text or "No answer generated."
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
            fall_back_tool_service = FallBackToolService(request=None)
            generic_responce_tool_service = GenericResponceToolService(request=None)
            if agent_name == "get_all_object_details":
                tool_result = await video_tool_service.get_all_object_details(args)
            elif agent_name == "show_evidence":
                tool_result = await evidence_tool_service.show_evidence(args)
                return tool_result
            elif agent_name == "generic_responce":
                tool_result = await generic_responce_tool_service.generic_responce(args)
            elif agent_name == "get_video_segment_details":
                tool_result = await video_segment_tool_service.get_video_segment_details(args)
            else:
                tool_result = await fall_back_tool_service.fall_back(args)
            # If tool_result indicates an error/failure, surface it for outer retry logic
            if self._is_failure_output(tool_result):
                return tool_result
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

    def _is_failure_text(self, text: str) -> bool:
        """
        Heuristic to detect failure/error texts from LLM.
        """
        if not text:
            return True
        t = text.lower()
        keywords = ["error", "failed", "failure", "unknown tool", "cannot", "unable", "timed out", "exception", "traceback","failed to call tool","<function=","function:","args:","args: {"]
        return any(k in t for k in keywords)

    def _is_failure_output(self, result) -> bool:
        """
        Detect if tool output indicates failure. Works for strings or dicts.
        """
        if result is None:
            return True
        if isinstance(result, str):
            return self._is_failure_text(result)
        if isinstance(result, dict):
            # Common patterns: {"error": ...} or {"status":"failed"}
            if any(k in result for k in ("error", "errors")):
                return True
            status = result.get("status") if isinstance(result.get("status"), str) else None
            if status and status.lower() in ("error", "failed", "failure"):
                return True
        return False