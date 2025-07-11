from app.services.object_tracker import calculate_angle

def format_result(tracks, total_frames, fps, duration):
    formatted_tracks = []
    crack_count = 0
    
    # Count cracks and format tracks
    for track in tracks:
        if track.get("label") == "crack":
            crack_count += 1
            continue  # Skip adding cracks to regular tracks
            
        formatted_track = {
            "track_id": track["track_id"],
            "label": track["label"],
            "trajectory": track["trajectory"],
            "timestamps": track["timestamps"],
            "frames": track["frames"],
            "actions": describe_actions(track["trajectory"])
        }
        formatted_tracks.append(formatted_track)

    return {
        "summary": {
            "total_frames": total_frames,
            "fps": fps,
            "duration_seconds": round(duration, 2),
            "object_count": len(formatted_tracks),
            "crack_count": crack_count
        },
        "tracks": formatted_tracks,
        "crack_count": crack_count
    }

def describe_actions(trajectory):
    actions = []
    for i in range(1, len(trajectory)):
        angle = calculate_angle(trajectory[i-1], trajectory[i])
        actions.append(f"Moved at {angle:.2f} degrees")
    return actions


def build_summary_prompt(tracks, max_objects=100):
    """
    Builds a compact summary prompt for the LLM.

    Args:
        tracks (list): List of tracked objects.
        max_objects (int): Max number of objects to include.

    Returns:
        str: The prompt string for the LLM.
    """
    # Sort by movement length and take top N objects
    sorted_tracks = sorted(tracks, key=lambda t: len(t.get("trajectory", [])), reverse=True)
    top_tracks = sorted_tracks[:max_objects]

    object_lines = []
    for obj in top_tracks:
        trajectory = obj.get("trajectory", [])
        if len(trajectory) > 1:
            start = trajectory[0]
            end = trajectory[-1]
            dx, dy = end[0] - start[0], end[1] - start[1]
            distance = (dx**2 + dy**2) ** 0.5
            direction = get_main_direction(dx, dy)
            line = (
                f"- {obj['label']} (ID {obj['track_id']}): moved {distance:.1f} units towards {direction}."
            )
        else:
            line = f"- {obj['label']} (ID {obj['track_id']}): stationary."
        object_lines.append(line)

    prompt = (
        "You are a professional video movement analyst. Your task is to review object tracking data from a video and produce a concise, insightful summary in plain English.\n\n"
        "Here’s how to structure your summary:\n"
        "1. Begin with a all object overview: total objects tracked, types of objects, specific directions of movement of each object and distances covered, which direction dit the object move, etc..\n"
        "2. Group objects by type (e.g., cars, trucks, people, etc...) and provide counts for each type.\n"
        "3. Highlight significant movements (e.g., longest distances covered, unusual directions, or outliers , which direction dit the object move, etc..).\n"
        "4. Point out any observable patterns or interactions (e.g., multiple vehicles moving together, people crossing paths with vehicles , etc...).\n"
        "5. Keep the language concise, professional, and human-readable. Avoid technical jargon or raw data repetition.\n\n"
        
        "Example Output:\n"
        "\"The video shows 10 moving objects: 6 cars, 3 trucks, and 1 person. Most vehicles moved westward, while two cars traveled significant distances eastward. The longest movement was by Car ID 14, covering 801 units towards the east. No notable interactions were observed between objects.\"\n\n"
        "\"The video captures 12 moving objects: 7 cars, 3 pedestrians, and 2 trucks. Between 00:02 and 00:10, most vehicles traveled eastward, while Truck ID 5 moved slowly westward. At 00:05, Car ID 8 changed lanes abruptly to avoid a pedestrian crossing the road. The longest movement was by Car ID 2, covering 620 units towards the east. No collisions or unusual interactions occurred.\"\n\n"
        "Tracking Data (Top Moving Objects):\n"
        + "\n".join(object_lines) +
        "\n\nNow produce a similar 3-5 sentence summary for this data:"
    )

    return prompt


def get_main_direction(dx, dy):
    """
    Determines the primary direction of movement.

    Args:
        dx (float): Change in x.
        dy (float): Change in y.

    Returns:
        str: Direction (e.g., North, South, East, West)
    """
    if abs(dx) > abs(dy):
        return "East" if dx > 0 else "West"
    else:
        return "North" if dy > 0 else "South"

    # return (
    #     "You are a helpful assistant summarizing object movement in a video.\n"
    #     f"Only include insights about the top {max_objects} moving objects.\n"
    #     "Tracking Data:\n"
    #     + "\n".join(object_lines)
    #     + "\nProvide a concise summary of the objects' movements in the video."
    # )


