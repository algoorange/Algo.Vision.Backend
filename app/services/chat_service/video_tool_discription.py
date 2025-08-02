# This file describes what tools (functions) the AI can call.
# Each tool must have a unique 'name' and a description of its parameters.
# Add more tools here if you want to support more analytics in the future.

# This file describes what tools (functions) the AI can call.
# Add more tools to this list as you add more analytics!

video_tool_description = [
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "analyze_video_behavior",
    #         "description": "Answer user questions about detected objects and their counts in a specific video. Use this tool for queries like 'How many cars are in the video?', 'How many trucks?', or 'What objects are present in the video?'.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "query": {
    #                     "type": "string",
    #                     "description": "A natural language question about video analytics (e.g., 'How many cars are in the video?')",
    #                 },
    #                 "video_id": {
    #                     "type": "string",
    #                     "description": "The unique video ID to analyze (from your database)",
    #                 }
    #             },
    #             "required": ["query", "video_id"]
    #         }
    #     }
    # },
    {
        "type": "function",
        "function": {
            "name": "get_all_object_details",
            "description": "Retrieve detailed information about every detected object in a given video, including their frame, position, and complete tracking details. This tool supports both general object listing and advanced time-specific or question-based analysis. If the user's question includes a time frame (e.g., 'first 10 seconds', 'last 30 seconds', 'between 5-15 seconds') or a specific natural language query, the tool will intelligently parse the question, fetch relevant objects from segmented video analysis, and provide detailed answers about counts, colors, movements, and more within the specified time range. Use this tool for both general and time-specific object queries, including temporal analytics and segment-based insights.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "The unique video ID to retrieve object details for (from your database)."
                    },
                    "question": {
                        "type": "string",
                        "description": "Optional: Natural language question that may include time frame references (e.g., 'How many cars in the first 10 seconds?', 'What colors appear between 5-15 seconds?', 'Show movement in the last 30 seconds'). The tool will parse time frames and generate appropriate answers."
                    },
                    "segment_duration": {
                        "type": "number",
                        "description": "Optional: Filter segments by specific duration in seconds (5.0, 10.0, 30.0, or 60.0). If not provided, returns all segment durations.",
                        "enum": [5.0, 10.0, 30.0, 60.0]
                    },
                    "segment_index": {
                        "type": "integer",
                        "description": "Optional: Get specific segment by index within the chosen duration. If not provided, returns all segments."
                    }
                },
                "required": ["video_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_specific_object_type",
            "description": "Retrieve detailed information about all detected objects of a specific type in a given video, including their frame and position details. Use this tool when the user asks for a list or details of all objects of a specific type in a video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "The unique video ID to get object details for (from your database).",
                    },
                    "object_type": {
                        "type": "string",
                        "description": "The type of object to get object details for (e.g., 'car', 'truck').",
                    }
                },
                "required": ["video_id", "object_type"]
            }
        }
    },
        {
        "type": "function",
        "function": {
            "name": "get_traffic_congestion_details",
            "description": "Retrieve detailed information about traffic congestion, crowding, and object counts in a given video. The tool returns the total object count and unique object count by category. Use this tool when the user asks for a list or details of objects in a video, or questions about traffic congestion or crowding.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "The unique video ID to get object details for (from your database).",
                    },
                    "question": {
                        "type": "string",
                        "description": "The natural language question about traffic congestion, crowding, or object counts in a video.",
                    }
                },
                "required": ["video_id", "question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "object_position_confidence_using_track_id",
            "description": "Retrieve the position, confidence, object type, and movement direction (towards or away from the camera) for the object with the given track_id in a specific frame. Use this tool when the user asks for a list or details of all objects of a specific type in a video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "The unique video ID to get object details for (from your database).",
                    },
                    "track_id": {
                        "type": "string",
                        "description": "The track ID of the object to get object details for (from your database).",
                    },
                    "frame_number": {
                        "type": "integer",
                        "description": "The frame number in the video to get object details for."
                    }
                },
                "required": ["video_id", "track_id", "frame_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_left_right_moving_objects",
            "description": "Get the count of objects going left and right across the video. Use this tool when the user asks for a list or details of all objects of a specific type in a video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "The unique video ID to get object details for (from your database).",
                    }
                },
                "required": ["video_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_object_direction",
            "description": "Get the direction (east, west, north, south) of all objects in a video using their position in the video. Use this tool when the user asks for a list or details of all objects of a specific type in a video. The tool takes a video_id as a parameter and returns the direction of all objects in that video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "The unique video ID to get object details for (from your database).",
                    }
                },
                "required": ["video_id"]
            }
        }
    },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "get_segmented_object_details",
    #         "description": "Retrieve comprehensive video segment analysis data with intelligent time frame parsing and question answering capabilities. This tool can parse natural language time references from user questions (e.g., 'first 10 seconds', 'between 5-15 seconds', 'last 30 seconds', 'at 20 seconds') and filter segments accordingly. It provides detailed insights into video segments of different durations (5s, 10s, 30s, 60s) with complete object lifecycle tracking. Each segment contains: object counts by type, total detection counts, frame-by-frame object details with positions, confidence scores, colors, and complete tracking information (start/end times, start/end frames, start/end positions). The tool can answer specific questions about object counts, colors, movements, and directions within specified time frames. Use this when users ask time-specific questions about video content or need detailed temporal analysis.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "video_id": {
    #                     "type": "string",
    #                     "description": "The unique video ID to retrieve segmented analysis data for (from video_details_segment collection).",
    #                 },
    #                 "question": {
    #                     "type": "string",
    #                     "description": "Optional: Natural language question that may include time frame references (e.g., 'How many cars in the first 10 seconds?', 'What colors appear between 5-15 seconds?', 'Show movement in the last 30 seconds'). The tool will parse time frames and generate appropriate answers."
    #                 },
    #                 "segment_duration": {
    #                     "type": "number",
    #                     "description": "Optional: Filter segments by specific duration in seconds (5.0, 10.0, 30.0, or 60.0). If not provided, returns all segment durations.",
    #                     "enum": [5.0, 10.0, 30.0, 60.0]
    #                 },
    #                 "segment_index": {
    #                     "type": "integer",
    #                     "description": "Optional: Get specific segment by index within the chosen duration. If not provided, returns all segments."
    #                 }
    #             },
    #             "required": ["video_id"]
    #         }
    #     }
    # }

]
