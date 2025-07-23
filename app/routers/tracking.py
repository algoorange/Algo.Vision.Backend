from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict
from app.utils.tracking_analytics import TrackingAnalyzer, generate_tracking_report
from app.services.object_tracker import get_track_statistics
from pymongo import MongoClient

router = APIRouter()

# MongoDB connection
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["algo_compliance_db_2"]
video_details_collection = db["video_details"]

@router.get("/analytics/{video_id}")
async def get_tracking_analytics(
    video_id: str,
    include_anomalies: bool = Query(True, description="Include anomaly detection"),
    include_interactions: bool = Query(True, description="Include object interactions"),
    include_zones: bool = Query(True, description="Include zone analysis")
):
    """
    Get comprehensive tracking analytics for a specific video
    """
    try:
        # Fetch tracking data from MongoDB
        tracking_data = list(video_details_collection.find({"video_id": video_id}))
        
        if not tracking_data:
            raise HTTPException(status_code=404, detail="No tracking data found for this video")
        
        # Group data by track_id
        tracks_by_id = {}
        for record in tracking_data:
            track_id = record.get("track_id")
            if track_id not in tracks_by_id:
                tracks_by_id[track_id] = {
                    "track_id": track_id,
                    "label": record.get("detected_object"),
                    "trajectory": [],
                    "timestamps": [],
                    "frames": [],
                    "movement_data": {
                        "total_distance": 0.0,
                        "max_speed": 0.0,
                        "avg_speed": 0.0,
                        "stationary_frames": 0,
                        "moving_frames": 0,
                        "directions": []
                    }
                }
            
            # Add position to trajectory
            pos = record.get("position", {})
            center_x = (pos.get("x", 0) + pos.get("x1", 0)) / 2
            center_y = (pos.get("y", 0) + pos.get("y1", 0)) / 2
            
            tracks_by_id[track_id]["trajectory"].append((center_x, center_y))
            tracks_by_id[track_id]["timestamps"].append(record.get("frame_time", 0))
            tracks_by_id[track_id]["frames"].append(record.get("frame", 0))
            
            # Extract movement data if available
            movement = record.get("movement", {})
            if movement:
                movement_data = tracks_by_id[track_id]["movement_data"]
                movement_data["directions"].append(movement.get("direction", 0))
                if movement.get("is_stationary", False):
                    movement_data["stationary_frames"] += 1
                else:
                    movement_data["moving_frames"] += 1
        
        # Convert to list and calculate additional metrics
        tracks_list = list(tracks_by_id.values())
        
        # Calculate movement metrics for each track
        for track in tracks_list:
            trajectory = track["trajectory"]
            if len(trajectory) > 1:
                total_distance = 0.0
                speeds = []
                
                for i in range(1, len(trajectory)):
                    p1, p2 = trajectory[i-1], trajectory[i]
                    distance = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
                    total_distance += distance
                    speeds.append(distance)
                
                track["movement_data"]["total_distance"] = total_distance
                track["movement_data"]["avg_speed"] = total_distance / len(trajectory) if len(trajectory) > 0 else 0
                track["movement_data"]["max_speed"] = max(speeds) if speeds else 0
        
        # Run analytics
        analyzer = TrackingAnalyzer()
        analysis = analyzer.analyze_track_movements(tracks_list)
        
        # Filter results based on query parameters
        if not include_anomalies:
            del analysis["anomalies"]
        if not include_interactions:
            del analysis["object_interactions"]
        if not include_zones:
            del analysis["zone_activity"]
        
        return {
            "video_id": video_id,
            "total_tracks": len(tracks_list),
            "analytics": analysis,
            "tracks_summary": [
                {
                    "track_id": track["track_id"],
                    "object_type": track["label"],
                    "duration_frames": len(track["trajectory"]),
                    "total_distance": track["movement_data"]["total_distance"],
                    "avg_speed": track["movement_data"]["avg_speed"]
                }
                for track in tracks_list
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing tracking data: {str(e)}")

@router.get("/report/{video_id}")
async def get_tracking_report(video_id: str):
    """
    Generate a comprehensive text report for tracking analysis
    """
    try:
        # Fetch and process data (same as analytics endpoint)
        tracking_data = list(video_details_collection.find({"video_id": video_id}))
        
        if not tracking_data:
            raise HTTPException(status_code=404, detail="No tracking data found for this video")
        
        # Group data by track_id (reuse logic from analytics endpoint)
        tracks_by_id = {}
        for record in tracking_data:
            track_id = record.get("track_id")
            if track_id not in tracks_by_id:
                tracks_by_id[track_id] = {
                    "track_id": track_id,
                    "label": record.get("detected_object"),
                    "trajectory": [],
                    "movement_data": {
                        "total_distance": 0.0,
                        "max_speed": 0.0,
                        "avg_speed": 0.0,
                        "stationary_frames": 0,
                        "moving_frames": 0,
                        "directions": []
                    }
                }
            
            pos = record.get("position", {})
            center_x = (pos.get("x", 0) + pos.get("x1", 0)) / 2
            center_y = (pos.get("y", 0) + pos.get("y1", 0)) / 2
            tracks_by_id[track_id]["trajectory"].append((center_x, center_y))
        
        tracks_list = list(tracks_by_id.values())
        
        # Generate comprehensive report
        report_text = generate_tracking_report(tracks_list)
        
        return {
            "video_id": video_id,
            "report": report_text,
            "generated_at": "now"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@router.get("/live-stats")
async def get_live_tracking_stats():
    """
    Get current live tracking statistics
    """
    try:
        stats = get_track_statistics()
        return {
            "live_tracking": {
                "active_tracks": stats.get("active_tracks", 0),
                "total_trajectory_points": stats.get("total_trajectories", 0),
                "memory_usage": {
                    "track_metadata_count": len(stats.get("track_metadata", {}))
                }
            },
            "status": "active" if stats.get("active_tracks", 0) > 0 else "idle"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting live stats: {str(e)}")

@router.get("/anomalies")
async def get_anomalies(
    video_id: Optional[str] = Query(None, description="Filter by video ID"),
    anomaly_type: Optional[str] = Query(None, description="Filter by anomaly type"),
    limit: int = Query(50, description="Maximum number of anomalies to return")
):
    """
    Get detected anomalies across all videos or filtered by video_id
    """
    try:
        # Build query
        query = {}
        if video_id:
            query["video_id"] = video_id
        
        # Fetch data
        tracking_data = list(video_details_collection.find(query).limit(limit * 10))  # Get more to analyze
        
        if not tracking_data:
            return {"anomalies": [], "total": 0}
        
        # Group by tracks and analyze
        tracks_by_id = {}
        for record in tracking_data:
            track_id = record.get("track_id")
            if track_id not in tracks_by_id:
                tracks_by_id[track_id] = {
                    "track_id": track_id,
                    "label": record.get("detected_object"),
                    "trajectory": [],
                    "movement_data": {"avg_speed": 0, "directions": []}
                }
            
            # Add basic trajectory and movement data
            pos = record.get("position", {})
            center_x = (pos.get("x", 0) + pos.get("x1", 0)) / 2
            center_y = (pos.get("y", 0) + pos.get("y1", 0)) / 2
            tracks_by_id[track_id]["trajectory"].append((center_x, center_y))
        
        tracks_list = list(tracks_by_id.values())
        
        # Detect anomalies
        analyzer = TrackingAnalyzer()
        anomalies = analyzer._detect_anomalies(tracks_list)
        
        # Filter by type if specified
        if anomaly_type:
            anomalies = [a for a in anomalies if a.get("type") == anomaly_type]
        
        # Limit results
        anomalies = anomalies[:limit]
        
        return {
            "anomalies": anomalies,
            "total": len(anomalies),
            "video_id": video_id,
            "anomaly_type": anomaly_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting anomalies: {str(e)}")

@router.get("/interactions/{video_id}")
async def get_object_interactions(video_id: str):
    """
    Get detected object interactions for a specific video
    """
    try:
        # Fetch tracking data
        tracking_data = list(video_details_collection.find({"video_id": video_id}))
        
        if not tracking_data:
            raise HTTPException(status_code=404, detail="No tracking data found for this video")
        
        # Process data (similar to other endpoints)
        tracks_by_id = {}
        for record in tracking_data:
            track_id = record.get("track_id")
            if track_id not in tracks_by_id:
                tracks_by_id[track_id] = {
                    "track_id": track_id,
                    "label": record.get("detected_object"),
                    "trajectory": [],
                    "frames": []
                }
            
            pos = record.get("position", {})
            center_x = (pos.get("x", 0) + pos.get("x1", 0)) / 2
            center_y = (pos.get("y", 0) + pos.get("y1", 0)) / 2
            tracks_by_id[track_id]["trajectory"].append((center_x, center_y))
            tracks_by_id[track_id]["frames"].append(record.get("frame", 0))
        
        tracks_list = list(tracks_by_id.values())
        
        # Detect interactions
        analyzer = TrackingAnalyzer()
        interactions = analyzer._detect_interactions(tracks_list)
        
        return {
            "video_id": video_id,
            "interactions": interactions,
            "total_interactions": len(interactions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting interactions: {str(e)}") 