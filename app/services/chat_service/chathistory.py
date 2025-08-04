import os
from langchain.memory import ConversationSummaryBufferMemory
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage, BaseMessage
from typing import Dict, List, Union, Optional, Any
import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from groq.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartParam,
)
from pydantic import SecretStr

MONGO_URI = os.environ.get("MONGO_URI", "")
DB_NAME = os.environ.get("MONGO_DB_NAME", "")

# Initialize Groq LLM
llm = ChatGroq(
    api_key=SecretStr(os.getenv("GROQ_API_KEY", "")),
    model="llama-3.3-70b-versatile"
)

# Initialize memory with reasonable defaults
memory = ConversationSummaryBufferMemory(
    llm=llm,
    return_messages=True,
    max_token_limit=500
)

# Global state
chat_memories: Dict[str, ConversationSummaryBufferMemory] = {}
chat_metadata: Dict[str, List[Dict[str, Any]]] = {}
message_count: Dict[str, int] = {}
chat_contexts: Dict[str, Dict[str, Any]] = {}
max_token_limit: int = 100

def generate_chat_id(
    user_id: Optional[str] = None,
    chatTab_id: Optional[str] = None,
    chat_box_id: Optional[str] = None,
    entity_id: Optional[str] = None,
    entity_type: Optional[str] = None
) -> str:
    """Generate a unique chat ID based on provided parameters.
    
    Args:
        user_id: User identifier
        chatTab_id: Chat tab identifier
        chat_box_id: Chat box identifier
        entity_id: Entity identifier
        entity_type: Entity type
        
    Returns:
        str: Generated chat ID
    """
    user_id = user_id or "manojkke"
    chat_box_id = chat_box_id or "gen_chat"
    chatTab_id = chatTab_id or "tb"
    entity_id = entity_id or "eid"
    entity_type = entity_type or "etype"

    parts = [user_id, chatTab_id, chat_box_id, entity_id, entity_type]
    return "_".join(parts)

class ChatMemoryForLLMService:
    def __init__(self):
        """Initialize the chat memory service."""
        self.llm = llm
        self.mongo_client = AsyncIOMotorClient(MONGO_URI)
        self.db = self.mongo_client[DB_NAME]
        self.chat_collection = self.db.chat_history
        self.context_collection = self.db.chat_contexts

    def _init_chat_if_needed(self, chat_id: str) -> None:
        """Initialize chat memory and metadata if not exists.
        
        Args:
            chat_id: Unique chat identifier
        """
        if chat_id not in chat_memories:
            chat_memories[chat_id] = ConversationSummaryBufferMemory(
                llm=self.llm,
                return_messages=True,
                max_token_limit=max_token_limit,
                memory_key="history",
                input_key="input"
            )
            chat_metadata[chat_id] = []
            message_count[chat_id] = 0
            chat_contexts[chat_id] = {}

    def add_message(self, chat_id: str, human_content: Optional[str], ai_content: Optional[str]) -> None:
        """Add a new message pair to the chat history.
        
        Args:
            chat_id: Unique chat identifier
            human_content: User message content
            ai_content: AI response content
        """
        self._init_chat_if_needed(chat_id)
        timestamp = datetime.datetime.utcnow().isoformat()

        if human_content:
            message_entry = {
                "role": "user",
                "content": human_content,
                "timestamp": timestamp
            }
            chat_metadata[chat_id].append(message_entry)
            message_count[chat_id] += 1
            chat_memories[chat_id].chat_memory.add_user_message(human_content)

        if ai_content:
            message_entry = {
                "role": "assistant",
                "content": ai_content,
                "timestamp": timestamp
            }
            chat_metadata[chat_id].append(message_entry)
            message_count[chat_id] += 1
            chat_memories[chat_id].chat_memory.add_ai_message(ai_content)

        # Trigger summarization if needed
        self.summarize_if_needed(chat_id)

    def get_messages(self, chat_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a chat session.
        
        Args:
            chat_id: Unique chat identifier
            
        Returns:
            List of message dictionaries
        """
        self._init_chat_if_needed(chat_id)
        return chat_metadata.get(chat_id, [])

    # def summarize_if_needed(self, chat_id: str) -> None:
    #     """Trigger chat summarization if message count threshold reached.
        
    #     Args:
    #         chat_id: Unique chat identifier
    #     """
    #     self._init_chat_if_needed(chat_id)
    #     if message_count[chat_id] >= 5:
    #         _ = chat_memories[chat_id].load_memory_variables({})
    #         message_count[chat_id] = 0

    def summarize_if_needed(self, chat_id: str) -> None:
        """Summarize history if token count exceeds limit, keeping last 5 messages intact."""
        self._init_chat_if_needed(chat_id)
        messages = chat_memories[chat_id].chat_memory.messages

        if self._count_tokens(messages) < max_token_limit:
            return  # No need to summarize

        last_five = self._get_last_messages(chat_id, 3)
        to_summarize = messages[:-3] if len(messages) > 3 else []

        if not to_summarize:
            return  # Nothing to summarize

        # Summarize using LLM
        summary = chat_memories[chat_id].predict_new_summary(
            existing_summary="",
            messages=to_summarize
        )

        # Replace chat memory with summarized + last 5
        summarized_history = [AIMessage(content=summary)] + last_five
        chat_memories[chat_id].chat_memory.messages = summarized_history

    def _count_tokens(self, messages: List[Union[HumanMessage, AIMessage]]) -> int:
        count = sum(len(str(m.content).split()) for m in messages)  # Rough token count
        print(f"Token count: {count}")
        return count


    def _get_last_messages(self, chat_id: str, count: int = 3) -> List[BaseMessage]:
        """Get last N messages from chat memory."""
        return chat_memories[chat_id].chat_memory.messages[-count:]

    def reset_chat(self, chat_id: str) -> None:
        """Reset all chat data for a session.
        
        Args:
            chat_id: Unique chat identifier
        """
        if chat_id in chat_memories:
            del chat_memories[chat_id]
        if chat_id in chat_metadata:
            del chat_metadata[chat_id]
        if chat_id in message_count:
            del message_count[chat_id]
        if chat_id in chat_contexts:
            del chat_contexts[chat_id]

    def get_chat_history_in_LLM_feedable_form(self, chat_id: str) -> List[ChatCompletionMessageParam]:
        """Get chat history formatted for LLM input.
        
        Args:
            chat_id: Unique chat identifier
            
        Returns:
            List of properly typed chat messages
        """
        self._init_chat_if_needed(chat_id)
        all_messages = chat_memories[chat_id].chat_memory.messages
        messages: List[ChatCompletionMessageParam] = []
        
        for msg in all_messages:
            if isinstance(msg, HumanMessage):
                messages.append(ChatCompletionUserMessageParam(
                    role="user",
                    content=str(msg.content)  # Ensure content is string
                ))
            elif isinstance(msg, AIMessage):
                messages.append(ChatCompletionAssistantMessageParam(
                    role="assistant", 
                    content=str(msg.content)  # Ensure content is string
                ))
                
        return messages

    def get_context(self, chatTab_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get context for a chat session.
        
        Args:
            chatTab_id: Chat tab identifier
            user_id: User identifier
            
        Returns:
            Context dictionary if exists, None otherwise
        """
        chat_id = generate_chat_id(user_id=user_id, chatTab_id=chatTab_id)
        return chat_contexts.get(chat_id)

    def set_context(self, chatTab_id: str, user_id: str, context: Dict[str, Any]) -> None:
        """Set context for a chat session.
        
        Args:
            chatTab_id: Chat tab identifier
            user_id: User identifier
            context: Context data to store
        """
        chat_id = generate_chat_id(user_id=user_id, chatTab_id=chatTab_id)
        self._init_chat_if_needed(chat_id)
        chat_contexts[chat_id] = context

    async def persist_context(self, chatTab_id: str, user_id: str) -> None:
        """Persist chat context to database.
        
        Args:
            chatTab_id: Chat tab identifier
            user_id: User identifier
        """
        chat_id = generate_chat_id(user_id=user_id, chatTab_id=chatTab_id)
        if context := chat_contexts.get(chat_id):
            await self.context_collection.update_one(
                {"chat_id": chat_id},
                {
                    "$set": {
                        "context": context,
                        "updated_at": datetime.datetime.utcnow()
                    }
                },
                upsert=True
            )

    async def load_context(self, chatTab_id: str, user_id: str) -> None:
        """Load chat context from database.
        
        Args:
            chatTab_id: Chat tab identifier
            user_id: User identifier
        """
        chat_id = generate_chat_id(user_id=user_id, chatTab_id=chatTab_id)
        if context_doc := await self.context_collection.find_one({"chat_id": chat_id}):
            chat_contexts[chat_id] = context_doc.get("context", {})


 
    