# """
# Google ADK Runner Service
# Handles execution of Google ADK agents with proper session management
# """

# import asyncio
# import logging
# import time
# from typing import Dict, Any, Optional
# from datetime import datetime

# from google.adk.runners import Runner
# from google.adk.memory import InMemoryMemoryService
# from google.adk.sessions import InMemorySessionService
# from google.adk.artifacts import InMemoryArtifactService
# from google.genai import types

# # Import the agent from Aagent.py
# # from app.services import root_agent

# logger = logging.getLogger(__name__)

# class GoogleADKRunnerService:
#     """
#     Service for running Google ADK agents with proper session management
#     """
    
#     def __init__(self):
#         self.session_service = InMemorySessionService()
#         self.memory_service = InMemoryMemoryService()
#         self.artifact_service = InMemoryArtifactService()

#         # Initialize the Google ADK agent
#         try:
#             # Use the agent from Aagent.py
#             self.agent = root_agent
#             logger.info("✅ Google ADK Agent loaded successfully")
#             print(f"Agent loaded: {self.agent.name if self.agent else 'None'}")
#         except Exception as e:
#             logger.error(f"❌ Failed to load Google ADK Agent: {e}")
#             self.agent = None
#             print("error", e)
    
#     async def execute_agent(self, user_id: str, message: str) -> Dict[str, Any]:
#         """
#         Execute the Google ADK agent with proper session management
#         """
#         if not self.agent:
#             return {"error": "Google ADK Agent not available"}
        
#         try:
#             start_time = time.time()
            
#             # Generate session ID for this conversation
#             session_id = self._get_session_id(user_id)
            
#             # Ensure session exists before running
#             await self._ensure_session_exists(user_id, session_id)
            
#             # Create runner with session
#             runner = Runner(
#                 app_name="google_adk_agent",
#                 agent=self.agent,
#                 session_service=self.session_service,
#                 memory_service=self.memory_service,
#                 artifact_service=self.artifact_service
#             )

#             new_message = types.Content(
#                 role="user",
#                 parts=[types.Part(text=message)]
#             )
            
#             # Execute the agent and collect all responses
#             agent_responses = []
#             async for response in runner.run_async(
#                 user_id=user_id,
#                 session_id=session_id,
#                 new_message=new_message
#             ):
#                 # Handle different response types with proper None checking
#                 try:
#                     if response is None:
#                         logger.warning("Received None response from agent")
#                         continue
                        
#                     if isinstance(response, str):
#                         if response and response.strip():
#                             agent_responses.append(response)
#                     elif hasattr(response, 'content'):
#                         if hasattr(response.content, 'parts'):
#                             for part in response.content.parts:
#                                 if hasattr(part, 'text') and part.text and part.text.strip():
#                                     agent_responses.append(part.text)
#                         else:
#                             content_str = str(response.content)
#                             if content_str and content_str.strip():
#                                 agent_responses.append(content_str)
#                     else:
#                         response_str = str(response)
#                         if response_str and response_str.strip():
#                             agent_responses.append(response_str)
#                 except Exception as response_error:
#                     logger.warning(f"Error processing response: {response_error}")
#                     continue

#             # Combine responses
#             valid_responses = [resp for resp in agent_responses if resp is not None and resp.strip()]
#             if valid_responses:
#                 response_text = "\n".join(valid_responses)
#             else:
#                 response_text = "No response received from agent. Please try again."

#             execution_time = time.time() - start_time

#             print(f"Result: {response_text}")
#             print(f"Session ID: {session_id}")
#             print(f"Execution time: {execution_time}")
#             return {
#                 "success": True,
#                 "response": response_text,
#                 "execution_time": execution_time,
#                 "session_id": session_id,
#                 "system": "google_adk_agent"
#             }
            
#         except Exception as e:
#             logger.error(f"❌ Google ADK execution failed: {e}")
#             return {
#                 "success": False,
#                 "error": f"Google ADK execution failed: {e}",
#                 "system": "google_adk_agent"
#             }
    
#     def _get_session_id(self, user_id: str) -> str:
#         """Generate a consistent session ID for the conversation"""
#         return f"google_adk_{user_id}"
    
#     def get_agent_status(self) -> Dict[str, Any]:
#         """Get the status of the Google ADK agent"""
#         return {
#             "agent_loaded": self.agent is not None,
#             "agent_name": self.agent.name if self.agent else None,
#             "sub_agents": len(self.agent.sub_agents) if self.agent and hasattr(self.agent, 'sub_agents') else 0,
#             "system": "google_adk_agent"
#         }
    
#     async def _ensure_session_exists(self, user_id: str, session_id: str):
#         """Ensure a session exists for the given user and session ID"""
#         try:
#             existing_session = await self.session_service.get_session(
#                 app_name="google_adk_agent",
#                 user_id=user_id,
#                 session_id=session_id
#             )
            
#             if not existing_session:
#                 await self.session_service.create_session(
#                     app_name="google_adk_agent",
#                     user_id=user_id,
#                     state={},
#                     session_id=session_id
#                 )
#                 logger.info(f"Created session {session_id} for user {user_id}")
#             else:
#                 logger.info(f"Session {session_id} already exists for user {user_id}")
                
#         except Exception as e:
#             logger.warning(f"Error ensuring session exists: {e}")

# # Global instance for singleton pattern
# _google_adk_runner_instance = None

# def get_google_adk_runner() -> GoogleADKRunnerService:
#     """Get or create the global Google ADK runner instance"""
#     global _google_adk_runner_instance
#     if _google_adk_runner_instance is None:
#         _google_adk_runner_instance = GoogleADKRunnerService()
#     return _google_adk_runner_instance 