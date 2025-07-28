import json
from app.infrastructure.services.chat_services.base_agent import BaseAgent
from app.domain.models import ChatEntityRequest
from .video_tool_service import VideoToolService
from .video_tool_description import video_tool_description

class VideoAgent(BaseAgent):
    def __init__(self, request: ChatEntityRequest):
        super().__init__(request)
        # This must match the list in video_tool_description.py
        self.tool_descriptions = video_tool_description
        self.system_prompt = (
            "You are a Video Intelligence Agent that can analyze object movement and behavior from surveillance video data."
        )
        # Service that actually does the analytics
        self.video_tool_service = VideoToolService(request)

    async def _handle_tool_call(self, tool_call):
        """
        This function receives a tool call from the AI and calls the correct method.
        Add more 'elif' blocks if you add more tools in video_tool_service.py and video_tool_description.py
        """
        args = json.loads(tool_call.function.arguments)
        tool_name = tool_call.function.name

        if tool_name == "analyze_video_behavior":
            # Call the service to analyze the video
            return await self.video_tool_service.analyze_video_behavior(args)
        else:
            # If you add more tools, handle them here
            return {"error": f"Unknown tool: {tool_name}"}

    async def process_request(self):
        """
        This is the main entry point. It sends the user message to the LLM, which may call a tool.
        """
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.request.message}
            ]

            # Call the LLM with the tool descriptions
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=self.tool_descriptions,
                max_tokens=500
            )
            message = response.choices[0].message
            message = await self._handle_malformed_tool_call(message)

            # If the LLM wants to call a tool, handle it
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_call = message.tool_calls[0]
                return await self._handle_tool_call(tool_call)

            # Otherwise, just return the message content
            return message.content
        except Exception as e:
            return f"Error processing video agent request: {e}"
