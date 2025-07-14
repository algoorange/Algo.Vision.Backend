from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import torch
from app.services import object_detector

router = APIRouter()

class DetectionConfig(BaseModel):
    use_yolo: Optional[bool] = None
    use_rtdetr: Optional[bool] = None
    use_faster_rcnn: Optional[bool] = None
    yolo_threshold: Optional[float] = None
    rtdetr_threshold: Optional[float] = None
    faster_rcnn_threshold: Optional[float] = None
    prioritize_faster_rcnn: Optional[bool] = None

@router.get("/config")
async def get_detection_config():
    """Get current detection model configuration"""
    try:
        config = object_detector.get_detection_config()
        available_models = object_detector.get_available_models()
        return {
            "success": True,
            "current_config": config,
            "available_models": available_models,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")

@router.post("/config")
async def update_detection_config(config: DetectionConfig):
    """Update detection model configuration"""
    try:
        # Convert to dict and filter out None values
        config_dict = {k: v for k, v in config.dict().items() if v is not None}
        
        if not config_dict:
            raise HTTPException(status_code=400, detail="No configuration parameters provided")
        
        object_detector.set_detection_config(**config_dict)
        new_config = object_detector.get_detection_config()
        
        return {
            "success": True,
            "message": "Detection configuration updated successfully",
            "updated_fields": config_dict,
            "new_config": new_config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")

@router.get("/models")
async def get_available_models():
    """Get list of available detection models with current status"""
    try:
        config = object_detector.get_detection_config()
        models = object_detector.get_available_models()
        
        model_status = {}
        for model in models:
            if model == "YOLOv8":
                model_status[model] = {
                    "enabled": config.get("use_yolo", False),
                    "threshold": config.get("yolo_threshold", 0.7)
                }
            elif model == "RTDETR":
                model_status[model] = {
                    "enabled": config.get("use_rtdetr", False),
                    "threshold": config.get("rtdetr_threshold", 0.6)
                }
            elif model == "Faster R-CNN":
                model_status[model] = {
                    "enabled": config.get("use_faster_rcnn", False),
                    "threshold": config.get("faster_rcnn_threshold", 0.7)
                }
        
        return {
            "success": True,
            "models": models,
            "model_status": model_status,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "prioritize_faster_rcnn": config.get("prioritize_faster_rcnn", False)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")

@router.post("/test")
async def test_detection(config: Optional[DetectionConfig] = None):
    """Test detection with current or provided configuration"""
    try:
        original_config = object_detector.get_detection_config()
        
        # Temporarily apply test configuration if provided
        if config:
            config_dict = {k: v for k, v in config.dict().items() if v is not None}
            if config_dict:
                object_detector.set_detection_config(**config_dict)
        
        # Get the current configuration for testing
        current_config = object_detector.get_detection_config()
        
        # Reset to original configuration
        object_detector.set_detection_config(**original_config)
        
        return {
            "success": True,
            "message": "Detection configuration test completed",
            "test_config": current_config,
            "device_available": torch.cuda.is_available(),
            "device_type": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
    except Exception as e:
        # Ensure we reset configuration even if error occurs
        try:
            object_detector.set_detection_config(**original_config)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to test configuration: {str(e)}")

@router.get("/performance")
async def get_performance_info():
    """Get performance information about detection models"""
    try:
        device_info = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "cuda_available": torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            device_info.update({
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(),
                "cuda_memory_allocated": torch.cuda.memory_allocated(),
                "cuda_memory_reserved": torch.cuda.memory_reserved()
            })
        
        model_info = {
            "YOLOv8": {
                "description": "Ultra-fast object detection, good for real-time applications",
                "typical_inference_time": "10-20ms",
                "accuracy": "High",
                "recommended_threshold": 0.7
            },
            "RTDETR": {
                "description": "Real-time detection transformer, excellent accuracy",
                "typical_inference_time": "15-30ms", 
                "accuracy": "Very High",
                "recommended_threshold": 0.6
            },
            "Faster R-CNN": {
                "description": "Two-stage detector, highest accuracy but slower",
                "typical_inference_time": "50-100ms",
                "accuracy": "Highest",
                "recommended_threshold": 0.7
            }
        }
        
        return {
            "success": True,
            "device_info": device_info,
            "model_info": model_info,
            "recommendations": {
                "real_time": ["YOLOv8"],
                "balanced": ["RTDETR"],
                "highest_accuracy": ["Faster R-CNN"],
                "multiple_models": ["Faster R-CNN", "RTDETR"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance info: {str(e)}") 