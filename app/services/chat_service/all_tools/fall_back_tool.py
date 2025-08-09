from pymongo import MongoClient
import os
import json

client = MongoClient(os.getenv("MONGO_URI"))
db = client["algo_compliance_db_2"]
video_details_segment = db["video_details_segment"]

import re
import json

class FallBackToolService:
    def __init__(self, request):
        self.request = request
        self.db = request.app.db if request and hasattr(request.app, 'db') and hasattr(request.app, 'db') else db

    def extract_filters(self, question: str):
        """Basic keyword extraction."""
        colors = ["red", "blue", "green", "white", "black", "gray", "yellow", "orange", "brown"]
        objects = ["car", "truck", "bus", "person", "bike", "motorcycle", "traffic light"]

        found_color = next((c for c in colors if c in question.lower()), None)
        found_object = next((o for o in objects if o in question.lower()), None)

        frame_match = re.search(r"frame\s+(\d+)", question.lower())
        frame_number = int(frame_match.group(1)) if frame_match else None

        return {"color": found_color, "object_type": found_object, "frame_number": frame_number}

    def is_generic_question(self, question: str) -> bool:
        """Detect generic overview questions that should return a segment-wise summary."""
        if not question:
            return True
        q = question.strip().lower()
        generic_phrases = [
            "what's happening",
            "what is happening",
            "what happens",
            "what is in the video",
            "what's in the video",
            "summary",
            "summarize",
            "overview",
            "describe",
            "describe the video",
            "tell me about the video",
            "what do you see",
        ]
        return any(p in q for p in generic_phrases)

    def combine_segment_descriptions(self, segments: list) -> dict:
        """Combine segment descriptions and aggregate object counts into a concise summary."""
        # Sort by segment_index to ensure correct order
        segments = sorted(segments, key=lambda s: s.get("segment_index", 0))
        per_segment = []
        total_counts = {}
        start_times = []
        end_times = []
        for seg in segments:
            summ = (seg.get("summary") or {})
            desc = summ.get("description") or "No description available."
            st = summ.get("start_time")
            et = summ.get("end_time")
            if st is not None:
                start_times.append(float(st))
            if et is not None:
                end_times.append(float(et))
            idx = seg.get("segment_index")
            if st is not None and et is not None:
                per_segment.append(f"Segment {idx} ({st:.2f}s–{et:.2f}s): {desc}")
            else:
                per_segment.append(f"Segment {idx}: {desc}")
            # aggregate counts
            oc = seg.get("object_counts") or {}
            for k, v in oc.items():
                try:
                    total_counts[k] = total_counts.get(k, 0) + int(v)
                except Exception:
                    pass
        overall_range = None
        if start_times and end_times:
            overall_range = [min(start_times), max(end_times)]
        # Build a compact overall summary sentence
        if total_counts:
            # top 5 by count
            items = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            counts_text = ", ".join([f"{k} ({v})" for k, v in items])
            overall = f"Overall, detected objects include: {counts_text}."
        else:
            overall = "Overall summary generated from segment descriptions."
        return {
            "overall_summary": overall,
            "per_segment_descriptions": per_segment,
            "total_segments": len(segments),
            "time_range": overall_range,
            "aggregated_object_counts": total_counts,
        }

    async def fall_back(self, args):
        """Generic fallback that searches DB using filters and segment summaries."""
        if args is None:
            args = {}
        elif isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                return {"result": "Invalid arguments: could not parse JSON."}

        video_id = args.get("video_id")
        question = args.get("question", "")
        if not video_id:
            return {"result": "video_id is required for fallback search."}

        filters = self.extract_filters(question)

        query = {"video_id": video_id}
        projection = {
            "_id": 0,
            "segment_index": 1,
            "summary": 1,
            "object_counts": 1,
            "objects": 1
        }

        segments = list(video_details_segment.find(query, projection).sort("segment_index", 1))

        # If the question is generic or contains no actionable filters, return combined summary as a concise string
        if self.is_generic_question(question) or (not filters.get("color") and not filters.get("object_type") and not filters.get("frame_number")):
            combined = self.combine_segment_descriptions(segments)
            per_seg = combined.get("per_segment_descriptions", [])
            # limit per-segment lines to avoid verbosity
            max_lines = 8
            if len(per_seg) > max_lines:
                shown = per_seg[:max_lines] + [f"... and {len(per_seg) - max_lines} more segment(s)"]
            else:
                shown = per_seg
            overall = combined.get("overall_summary", "")
            time_range = combined.get("time_range")
            if time_range and all(isinstance(x, (int, float)) for x in time_range):
                header = f"Overall summary ({time_range[0]:.2f}s–{time_range[1]:.2f}s): {overall}"
            else:
                header = f"Overall summary: {overall}"
            answer = header + "\n\n" + "\n".join(shown)
            return answer

        results = []

        # Step 1: Try filter-based search
        for seg in segments:
            for obj in seg.get("objects", []):
                props = obj.get("properties", {}) or {}
                # object_type is stored at top-level in each obj
                if filters["object_type"] and str(obj.get("object_type", "")).lower() != str(filters["object_type"]).lower():
                    continue
                col = props.get("color", obj.get("color"))
                if filters["color"] and str(col).lower() != str(filters["color"]).lower():
                    continue
                if filters["frame_number"] and not (
                    obj.get("start_frame") is not None and obj.get("end_frame") is not None and
                    int(obj.get("start_frame")) <= int(filters["frame_number"]) <= int(obj.get("end_frame"))
                ):
                    continue
                results.append({
                    "segment_index": seg["segment_index"],
                    "description": seg.get("summary", {}).get("description", ""),
                    "object_type": obj.get("object_type"),
                    "color": col,
                    "start_frame": obj.get("start_frame"),
                    "end_frame": obj.get("end_frame"),
                    "confidence": obj.get("confidence"),
                })

        # Step 2: If no object match, return summaries
        if not results:
            summaries = [
                f"Segment {seg['segment_index']}: {seg.get('summary', {}).get('description', 'No description available.')}"
                for seg in segments
            ]
            return "No exact object match found. Here's what the video contains:\n" + "\n".join(summaries)

        # Step 3: Format answer
        answer = f"Found {len(results)} matching object(s) for '{question}':\n"
        for r in results[:5]:  # limit
            conf = r.get('confidence')
            conf_text = f", confidence {conf:.2f}" if isinstance(conf, (int, float)) else ""
            answer += (
                f"- {r['color']} {r['object_type']} in segment {r['segment_index']} "
                f"(frames {r['start_frame']}–{r['end_frame']}{conf_text})\n"
                f"  Scene: {r['description']}\n"
            )

        return answer.strip()
