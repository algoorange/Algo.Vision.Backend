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
    "description": "Returns structured analytics from object detection and tracking in a video. This tool allows querying total object counts, unique objects, frame-wise analytics, confidence scores, object types, and filtered details by track ID or frame number. It is designed for precise, structured data access, not general natural language understanding. Use this tool to answer questions such as:\n\n• 'How many objects were detected in the video?'\n• 'What object types appeared in the video?'\n• 'Show me the confidence level of track ID 23.'\n• 'What is the object type of track ID 12?'\n• 'How many objects were detected in frame 45?'\n• 'Which track IDs are present in frame 10?'\n• 'What is the confidence and object type of track ID 5 in frame 15?'\n\nThis tool supports frame-specific queries and track ID-specific lookups. All queries require a valid video_id.",
    "parameters": {
      "type": "object",
      "properties": {
        "video_id": {
          "type": "string",
          "description": "The unique ID of the video from which object data should be retrieved. Required for all queries."
        },
        "get_count": {
          "type": "boolean",
          "description": "Set to true to return the total count of unique objects detected in the video (based on distinct track IDs)."
        },
        "get_unique_objects": {
          "type": "boolean",
          "description": "Set to true to return details of all unique tracked objects, including their track_id, object_type, and confidence score."
        },
        "get_frame_track_ids": {
          "type": "boolean",
          "description": "Set to true to retrieve a list of track IDs present in a specific frame. Requires 'frame_number'."
        },
        "get_confidences": {
          "type": "boolean",
          "description": "Set to true to retrieve confidence scores. If 'track_id' is provided, it returns confidence for that specific tracked object."
        },
        "get_object_color": {
          "type": "boolean",
          "description": "Set to true to retrieve object color. If 'track_id' is provided, it returns the color for that specific tracked object. if the track id and frame number is provided, it returns the color for that specific tracked object in that specific frame eg: how many red cars passed and give me the count."
        },
        "get_object_types": {
          "type": "boolean",
          "description": "Set to true to retrieve object types. If 'track_id' is provided, it returns the object type for that specific tracked object. if the track id and frame number is provided, it returns the object type for that specific tracked object in that specific frame"
        },
        "get_frame_object_count": {
          "type": "boolean",
          "description": "Set to true to return the count of all objects present in a specific frame. Requires 'frame_number'."
        },
        "get_position_by_frame_and_track": {
          "type": "boolean",
          "description": "Set to true to retrieve the position of a specific tracked object in a specific frame. Requires 'frame_number' and 'track_id'."
        },
        "frame_number": {
          "type": "integer",
          "description": "The frame number to query. Required for frame-specific queries like object count or track ID lookup."
        },
        "track_id": {
          "type": "integer",
          "description": "The ID of the tracked object. Optional, used to filter confidence or object type results to a specific object."
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
    "description": "Used to answer questions about specific time-based segments of a video. Supports queries like 'how many objects were present in the first 5 seconds', 'what object types appeared between 0 and 10 seconds', 'give me the frame range of object 23','in which frame to which frame object 23 was present or detected',' in which time to which time did the object 23 was present or detected', and 'what was the position of a car at the start and end of the video'. This tool analyzes fixed-duration segments of the video (e.g., 5s, 10s, 30s) and returns detailed insights about tracked objects.",
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
          "description": "set true to get the number of unique objects (by track ID) present in each segment. Useful for counting how many objects were seen in a time range like 'first 5 seconds'."
        },
        "get_segment_object_types": {
          "type": "boolean",
          "description": "set true to get a list of object types (like car, person, truck, etc.) that appeared in each segment."
        },
        "get_track_frame_range": {
          "type": "boolean",
          "description": "set true to get the first and last frame number where the object with the specified track ID appears."
        },
        "get_track_position_range": {
          "type": "boolean",
          "description": "set true to get the start and end position coordinates (x, y) of a specific tracked object by its track ID."
        },
        "get_track_time_range": {
          "type": "boolean",
          "description": "set true to get the start and end time (in seconds) when a specific object appeared in the video."
        },
        "time_range_start": {
          "type": ["number", "string"],
          "description": "Start of the time range (in seconds) to filter objects. Example: 10"
        },
        "time_range_end": {
          "type": ["number", "string"],
          "description": "End of the time range (in seconds). Example: 30. Use this along with time_range_start to get objects between times."
        },
        "last_n_seconds": {
          "type": ["number", "string"],
          "description": "Retrieve data from only the last N seconds of the video (e.g., 5, 10, 30). Ignored if time_range_start/time_range_end is used."
        },
        "track_id": {
          "type": "string",
          "description": "The unique track ID of the object to retrieve details for (required for track-level queries such as time, frame, or position range)."
        }
      },
      "required": ["video_id"]
    }
  }
},
{
  "type": "function",
  "function": {
    "name": "show_evidence",
    "description": "Retrieves visual evidence (e.g., video frames and captions) related to detected and tracked objects in a video. This tool intelligently handles a wide range of natural language queries to provide visual proof based on object type, color, location, movement, frame number, detection confidence, or track ID. It supports both direct filtering and contextual question-based prompts to return accurate evidence images.",
    "parameters": {
      "type": "object",
      "properties": {
        "video_id": {
          "type": "string",
          "description": "The unique identifier of the video from which to retrieve visual evidence."
        },
        "get_evidence": {
          "type": "boolean",
          "description": "Set this to true to retrieve and return evidence frames (images + relevant metadata) from the video."
        },
        "object_type": {
          "type": "string",
          "description": "The type of object to search for (e.g., car, truck, person, bus, bike). Optional. If not provided, all types are considered."
        },
        "track_id": {
          "type": "string",
          "description": "The specific track ID of the object for which evidence should be retrieved. Optional."
        },
        "frame_number": {
          "type": ["integer", "string"],
          "description": "The specific frame number to analyze. Optional. If omitted, all frames will be searched."
        },
        "question": {
          "type": "string",
          "description": "A natural language question from the user, such as 'Show the red car near the entrance' or 'Was a white truck detected?'. This field provides context to help extract the correct filters (color, object type, etc.) and return relevant visual evidence. Optional but useful when filters are not explicitly provided."
        }
      },
      "required": ["video_id", "get_evidence"]
    }
  }
}
    ]