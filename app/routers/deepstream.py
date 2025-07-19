from fastapi import APIRouter, HTTPException, UploadFile, File, Body
from typing import Dict, List, Optional
from pydantic import BaseModel
import logging
import uuid

from app.services.deepstream_service import (
    deepstream_service, 
    DeepStreamConfig, 
    StreamType,
    process_video_with_deepstream
)

router = APIRouter()

class DeepStreamPipelineRequest(BaseModel):
    source_type: str
    source_path: str
    width: int = 1920
    height: int = 1080
    fps: int = 30
    batch_size: int = 1
    enable_tracking: bool = True
    enable_display: bool = False
    enable_rtsp_out: bool = False
    rtsp_port: int = 8554

class PipelineResponse(BaseModel):
    pipeline_id: str
    status: str
    message: str

class PipelineStatusResponse(BaseModel):
    pipeline_id: str
    status: str
    frame_count: int
    detection_count: int

@router.post("/create-pipeline", response_model=PipelineResponse)
async def create_pipeline(request: DeepStreamPipelineRequest):
    """Create a new DeepStream processing pipeline"""
    try:
        # Generate unique pipeline ID
        pipeline_id = str(uuid.uuid4())
        
        # Map string to enum
        source_type_map = {
            "file": StreamType.FILE,
            "rtsp": StreamType.RTSP,
            "usb": StreamType.USB_CAMERA,
            "csi": StreamType.CSI_CAMERA,
            "live": StreamType.LIVE_STREAM
        }
        
        source_type = source_type_map.get(request.source_type.lower())
        if not source_type:
            raise HTTPException(status_code=400, detail=f"Invalid source type: {request.source_type}")
        
        # Create configuration
        config = DeepStreamConfig(
            source_type=source_type,
            source_path=request.source_path,
            width=request.width,
            height=request.height,
            fps=request.fps,
            batch_size=request.batch_size,
            enable_tracking=request.enable_tracking,
            enable_display=request.enable_display,
            enable_rtsp_out=request.enable_rtsp_out,
            rtsp_port=request.rtsp_port,
            primary_model_config="configs/deepstream/yolo_config.txt",
            tracker_config="configs/deepstream/tracker_config.txt"
        )
        
        # Create pipeline
        success = deepstream_service.create_pipeline(pipeline_id, config)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create pipeline")
        
        return PipelineResponse(
            pipeline_id=pipeline_id,
            status="created",
            message="Pipeline created successfully"
        )
        
    except Exception as e:
        logging.error(f"Error creating pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start-pipeline/{pipeline_id}", response_model=PipelineResponse)
async def start_pipeline(pipeline_id: str):
    """Start a DeepStream pipeline"""
    try:
        success = deepstream_service.start_pipeline(pipeline_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Pipeline not found or failed to start")
        
        return PipelineResponse(
            pipeline_id=pipeline_id,
            status="running",
            message="Pipeline started successfully"
        )
        
    except Exception as e:
        logging.error(f"Error starting pipeline {pipeline_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop-pipeline/{pipeline_id}", response_model=PipelineResponse)
async def stop_pipeline(pipeline_id: str):
    """Stop a DeepStream pipeline"""
    try:
        success = deepstream_service.stop_pipeline(pipeline_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        return PipelineResponse(
            pipeline_id=pipeline_id,
            status="stopped",
            message="Pipeline stopped successfully"
        )
        
    except Exception as e:
        logging.error(f"Error stopping pipeline {pipeline_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/remove-pipeline/{pipeline_id}", response_model=PipelineResponse)
async def remove_pipeline(pipeline_id: str):
    """Remove a DeepStream pipeline"""
    try:
        success = deepstream_service.remove_pipeline(pipeline_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        return PipelineResponse(
            pipeline_id=pipeline_id,
            status="removed",
            message="Pipeline removed successfully"
        )
        
    except Exception as e:
        logging.error(f"Error removing pipeline {pipeline_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipeline-status/{pipeline_id}", response_model=PipelineStatusResponse)
async def get_pipeline_status(pipeline_id: str):
    """Get status of a DeepStream pipeline"""
    try:
        status = deepstream_service.get_pipeline_status(pipeline_id)
        
        if status.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        return PipelineStatusResponse(
            pipeline_id=pipeline_id,
            status=status.get("status", "unknown"),
            frame_count=status.get("frame_count", 0),
            detection_count=status.get("detection_count", 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting pipeline status {pipeline_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipelines")
async def list_pipelines():
    """List all DeepStream pipelines"""
    try:
        pipelines = deepstream_service.list_pipelines()
        
        # Get status for each pipeline
        pipeline_info = []
        for pipeline_id in pipelines:
            status = deepstream_service.get_pipeline_status(pipeline_id)
            pipeline_info.append({
                "pipeline_id": pipeline_id,
                "status": status.get("status", "unknown"),
                "frame_count": status.get("frame_count", 0),
                "detection_count": status.get("detection_count", 0)
            })
        
        return {"pipelines": pipeline_info}
        
    except Exception as e:
        logging.error(f"Error listing pipelines: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipeline-results/{pipeline_id}")
async def get_pipeline_results(pipeline_id: str):
    """Get detection results from a pipeline"""
    try:
        results = deepstream_service.get_pipeline_results(pipeline_id)
        
        return {
            "pipeline_id": pipeline_id,
            "results": results,
            "total_detections": len(results)
        }
        
    except Exception as e:
        logging.error(f"Error getting pipeline results {pipeline_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-video")
async def process_video_deepstream(file: UploadFile = File(...)):
    """Process uploaded video using DeepStream pipeline"""
    try:
        # Generate unique video ID
        video_id = str(uuid.uuid4())
        
        # Save uploaded file
        video_filename = f"{video_id}_{file.filename}"
        video_path = f"uploads/{video_filename}"
        
        with open(video_path, "wb") as video_file:
            video_file.write(await file.read())
        
        # Process with DeepStream
        results = await process_video_with_deepstream(video_path, video_id)
        
        return {
            "message": "Video processed successfully with DeepStream",
            "video_id": video_id,
            "results": results
        }
        
    except Exception as e:
        logging.error(f"Error processing video with DeepStream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def deepstream_health():
    """Check DeepStream service health"""
    try:
        # Check if DeepStream bindings are available
        from app.services.deepstream_service import DEEPSTREAM_AVAILABLE
        
        return {
            "status": "healthy" if DEEPSTREAM_AVAILABLE else "degraded",
            "deepstream_available": DEEPSTREAM_AVAILABLE,
            "active_pipelines": len(deepstream_service.list_pipelines())
        }
        
    except Exception as e:
        logging.error(f"Error checking DeepStream health: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "deepstream_available": False,
            "active_pipelines": 0
        } 