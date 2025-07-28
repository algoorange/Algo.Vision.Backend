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
            "description": "Retrieve detailed information about every detected object in a given video, including their frame and position details. Use this tool when the user asks for a list or details of all objects in a video.",
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

]
