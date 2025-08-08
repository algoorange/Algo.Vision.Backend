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
          "description": "Set to true to retrieve object color details. For overall totals (no time window), combine with 'object_type' and 'color' to get unique counts by track_id. Example (overall): get_object_color=true, object_type='car', color='black' → answers 'how many black cars detected?'. If 'track_id' is provided, returns the color for that specific tracked object; if both 'track_id' and 'frame_number' are provided, returns the color for that object in that frame. Note: for time-window or segment-specific color counts, use get_video_segment_details instead."
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
        },
        "object_type": {
          "type": "string",
          "description": "Optional. Filter by object type when using get_object_color to compute overall color/type counts (e.g., object_type='car')."
        },
        "color": {
          "type": "string",
          "description": "Optional. Filter by color when using get_object_color to compute overall color/type counts (e.g., color='black')."
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
    "description": "Used to answer questions about specific time-based segments of a video. It supports generic queries (object type, color, time windows) and precise track-level lookups. Examples: 'how many blue cars passed in the first 5 seconds?', 'is any car visible between 7 and 10 seconds?', 'in the first 30 seconds, how many blue cars?', 'how many segments are in the video?', 'in which segment do cars appear the most?', 'give me the frame/time/position range of object 23'. It analyzes fixed-duration segments (e.g., 5s, 10s, 30s), uses per-segment object summaries, and can search segment descriptions by keywords.",
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
        },
        "object_type": {
          "type": ["string", "array"],
          "description": "Filter by object type(s) such as car, truck, person. Accepts a string or list of strings."
        },
        "min_confidence": {
          "type": ["number", "string"],
          "description": "Minimum detection confidence to include an object (e.g., 0.8)."
        },
        "segment_index": {
          "type": ["integer", "array"],
          "description": "Filter to specific segment index/indices."
        },
        "exists_only": {
          "type": "boolean",
          "description": "If true, return a boolean flag indicating whether any object matches the filters within the time window. Useful for queries like 'Is any car visible between 7 and 10 seconds?'."
        },
        "get_number_of_segments": {
          "type": "boolean",
          "description": "If true, return the number of segments in the video for the given segment_duration. return how many segment in the video for the given segment_duration."
        },
        "get_busiest_segment": {
          "type": "boolean",
          "description": "If true, return the segment with the highest object count. Can be combined with object_type to get busiest segment for a specific type."
        },
        "get_count_by_type": {
          "type": "boolean",
          "description": "If true, return counts and unique track counts by object type within the (optional) time window and filters."
        },
        "get_segment_descriptions": {
          "type": "boolean",
          "description": "If true, return segment descriptions with start/end times."
        },
        "get_segments_overview": {
          "type": "boolean",
          "description": "If true, return per-segment object_counts and totals as an overview."
        },
        "get_count_by_color_in_segment": {
          "type": "boolean",
          "description": "Time-windowed or segment-scoped color counts (de-duplicated by track_id). Use this when the user specifies a time window or segment constraint. For overall totals without time constraints (e.g., 'how many black cars detected?'), prefer get_all_object_details.\n\nFilters: \n- color: string or array (e.g., 'red' or ['red','blue'])\n- object_type: string or array (e.g., 'car', 'person')\n- min_confidence: float (optional)\n- segment_index: int or array of ints (optional)\n\nTime windows (pick one):\n- count_within_seconds: first N seconds (e.g., 30)\n- time_range_start/time_range_end: explicit range in seconds (e.g., 10..25)\n- last_n_seconds: last N seconds from video end (e.g., 20)\n\nReturns: unique counts by color (de-duplicated by track_id), list of unique track_ids per color, total_unique_objects, and the resolved window.\n\nExamples:\n1) 'How many red cars in the first 30 seconds?': set get_count_by_color=true, color='red', object_type='car', count_within_seconds=30\n2) 'Count red and blue vehicles between 10 and 25 seconds': get_count_by_color=true, color=['red','blue'], time_range_start=10, time_range_end=25\n3) 'How many red cars in the last 20 seconds?': get_count_by_color=true, color='red', object_type='car', last_n_seconds=20"
        },
        "search_description_keywords": {
          "type": ["string", "array"],
          "description": "Keyword(s) to search within segment descriptions (e.g., 'accident', 'crowd', 'blue car')."
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