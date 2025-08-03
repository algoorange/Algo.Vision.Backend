# This file describes what tools (functions) the AI can call.
# Each tool must have a unique 'name' and a description of its parameters.
# Add more tools here if you want to support more analytics in the future.

# This file describes what tools (functions) the AI can call.
# Add more tools to this list as you add more analytics!

video_tool_description = [
    {
        "type": "function",
        "function": {
            "name": "get_all_object_details",
            "description": "Returns specific analytics about detected and tracked objects in a video, based on structured flags. This tool can return the total object count, unique objects by track ID, confidence values, and object types either globally or for a specific track ID. It is used to answer detailed questions about object tracking in a video without relying on natural language interpretation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "The unique ID of the video from which object data should be retrieved. Required for all queries."
                    },
                    "get_count": {
                        "type": "boolean",
                        "description": "Set to true to return the count of unique tracked objects (based on distinct track IDs)."
                    },
                    "get_unique_objects": {
                        "type": "boolean",
                        "description": "Set to true to return a list of unique tracked objects with their track_id, object_type, and confidence."
                    },
                    "get_frame_track_ids": {
                        "type": "boolean",
                        "description": "Set to true to return a list of track IDs for a specific frame."
                    },
                    "get_confidences": {
                        "type": "boolean",
                        "description": "Set to true to return the confidence score(s) for detected objects. If 'track_id' is provided, returns confidence for that specific object only."
                    },
                    "get_object_types": {
                        "type": "boolean",
                        "description": "Set to true to return object type(s). If 'track_id' is provided, returns the object type for that specific track only."
                    },
                    "get_frame_object_count": {
                        "type": "boolean",
                        "description": "Set to true to get count of objects in a specific frame."
                    },
                    "frame_number": {
                        "type": "integer",
                        "description": "Frame number to query for object count. Required if get_frame_object_count is true."
                    },
                    "track_id": {
                        "type": "integer",
                        "description": "Optional. Use to get object type or confidence specific to this tracked object."
                    }
                },
                "required": ["video_id"]
            }
        }
    },
{
  "type": "function",
  "function": {
    "name": "get_video_segment_details",
    "description": "Used to answer questions about specific time-based segments of a video. Supports queries like 'how many objects were present in the first 5 seconds', 'what object types appeared between 0 and 10 seconds', 'give me the frame range of object 23', and 'what was the position of a car at the start and end of the video'. This tool analyzes fixed-duration segments of the video (e.g., 5s, 10s, 30s) and returns detailed insights about tracked objects.",
    "parameters": {
      "type": "object",
      "properties": {
        "video_id": {
          "type": "string",
          "description": "The unique ID of the video to analyze. This ID is used to retrieve video data from the database."
        },
        "segment_duration": {
          "type": ["number", "string"],
          "description": "The duration (in seconds) of each segment of the video. Must match the segmentation duration used during preprocessing. Example: 5 means every 5 seconds is a segment."
        },
        "count_within_seconds": {
          "type": ["number", "string"],
          "description": "Time range (in seconds) to count object detections from the beginning of the video. Accepts values like 5, 10, 60 (1 minute)."
        },
        "get_segment_object_counts": {
          "type": "boolean",
          "description": "Set to true to return the number of unique objects (by track ID) present in each segment. Useful for counting how many objects were seen in a time range like 'first 5 seconds'."
        },
        "get_segment_object_types": {
          "type": "boolean",
          "description": "Set to true to get a list of object types (like car, person, truck, etc.) that appeared in each segment."
        },
        "get_track_frame_range": {
          "type": "boolean",
          "description": "Returns the first and last frame number where the object with the specified track ID appears."
        },
        "get_track_position_range": {
          "type": "boolean",
          "description": "Returns the start and end position coordinates (x, y) of a specific tracked object by its track ID."
        },
        "get_track_time_range": {
          "type": "boolean",
          "description": "Returns the start and end time (in seconds) when a specific object appeared in the video."
        },
        "track_id": {
          "type": "string",
          "description": "The unique track ID of the object to retrieve details for (required for track-level queries such as time, frame, or position range)."
        }
      },
      "required": ["video_id"]
    }
  }
}
 
    ]