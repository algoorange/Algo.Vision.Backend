import os
import sys
import json
import asyncio
import logging
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import cv2
import numpy as np

# DeepStream Python bindings (to be installed)
try:
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GstRtspServer', '1.0')
    from gi.repository import GObject, Gst, GstRtspServer
    import pyds
    DEEPSTREAM_AVAILABLE = True
except ImportError:
    DEEPSTREAM_AVAILABLE = False
    logging.warning("DeepStream Python bindings not available. Install pyds for full functionality.")

from app.utils.helpers import format_result
from app.services.summary_generate_by_llm import generate_summary

class StreamType(Enum):
    FILE = "file"
    RTSP = "rtsp"
    USB_CAMERA = "usb"
    CSI_CAMERA = "csi"
    LIVE_STREAM = "live"

@dataclass
class DeepStreamConfig:
    """DeepStream pipeline configuration"""
    # Input configuration
    source_type: StreamType = StreamType.FILE
    source_path: str = ""
    width: int = 1920
    height: int = 1080
    fps: int = 30
    
    # Processing configuration
    batch_size: int = 1
    gpu_id: int = 0
    enable_tracking: bool = True
    enable_analytics: bool = True
    
    # Model configuration
    primary_model_config: str = ""
    secondary_model_configs: Optional[List[str]] = None
    tracker_config: str = ""
    
    # Output configuration
    enable_display: bool = True
    enable_rtsp_out: bool = False
    rtsp_port: int = 8554
    enable_file_out: bool = False
    output_file: str = ""
    
    # Performance configuration
    enable_tensorrt: bool = True
    inference_interval: int = 1
    tracking_interval: int = 1
    
    def __post_init__(self):
        if self.secondary_model_configs is None:
            self.secondary_model_configs = []

class DeepStreamPipeline:
    """DeepStream pipeline manager for AlgoVision"""
    
    def __init__(self, config: DeepStreamConfig, callback: Optional[Callable] = None):
        self.config = config
        self.callback = callback
        self.pipeline = None
        self.bus = None
        self.loop = None
        self.is_running = False
        self.frame_count = 0
        self.detection_results = []
        
        # Initialize GStreamer
        if DEEPSTREAM_AVAILABLE:
            Gst.init(None)
            self._create_pipeline()
        else:
            raise RuntimeError("DeepStream not available. Please install DeepStream SDK and Python bindings.")
    
    def _create_pipeline(self):
        """Create GStreamer pipeline with DeepStream elements"""
        self.pipeline = Gst.Pipeline()
        
        # Create source element based on type
        source = self._create_source()
        
        # Create preprocessing elements
        parser = self._create_parser()
        decoder = self._create_decoder()
        streammux = self._create_streammux()
        
        # Create inference elements
        pgie = self._create_primary_inference()
        tracker = self._create_tracker() if self.config.enable_tracking else None
        sgies = self._create_secondary_inferences()
        
        # Create processing elements
        nvvidconv = self._create_video_converter()
        nvosd = self._create_osd()
        
        # Create output elements
        sinks = self._create_sinks()
        
        # Add elements to pipeline
        elements = [source, parser, decoder, streammux, pgie]
        if tracker:
            elements.append(tracker)
        elements.extend(sgies)
        elements.extend([nvvidconv, nvosd])
        elements.extend(sinks)
        
        for element in elements:
            if element:
                self.pipeline.add(element)
        
        # Link elements
        self._link_elements(source, parser, decoder, streammux, pgie, tracker, sgies, nvvidconv, nvosd, sinks)
        
        # Set up bus
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self._bus_call)
        
        # Add probe to capture inference results
        if pgie:
            pgie_src_pad = pgie.get_static_pad("src")
            if pgie_src_pad:
                pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, self._inference_probe_callback, 0)
    
    def _create_source(self):
        """Create source element based on configuration"""
        if self.config.source_type == StreamType.FILE:
            source = Gst.ElementFactory.make("filesrc", "source")
            source.set_property("location", self.config.source_path)
        elif self.config.source_type == StreamType.RTSP:
            source = Gst.ElementFactory.make("rtspsrc", "source")
            source.set_property("location", self.config.source_path)
        elif self.config.source_type == StreamType.USB_CAMERA:
            source = Gst.ElementFactory.make("v4l2src", "source")
            source.set_property("device", self.config.source_path)
        else:
            raise ValueError(f"Unsupported source type: {self.config.source_type}")
        
        return source
    
    def _create_parser(self):
        """Create parser element"""
        if self.config.source_type == StreamType.FILE:
            # Auto-detect format
            return Gst.ElementFactory.make("h264parse", "parser")
        return None
    
    def _create_decoder(self):
        """Create decoder element"""
        return Gst.ElementFactory.make("nvv4l2decoder", "decoder")
    
    def _create_streammux(self):
        """Create stream multiplexer"""
        streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
        streammux.set_property("width", self.config.width)
        streammux.set_property("height", self.config.height)
        streammux.set_property("batch-size", self.config.batch_size)
        streammux.set_property("batched-push-timeout", 4000000)
        return streammux
    
    def _create_primary_inference(self):
        """Create primary inference element"""
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        if self.config.primary_model_config:
            pgie.set_property("config-file-path", self.config.primary_model_config)
        pgie.set_property("batch-size", self.config.batch_size)
        return pgie
    
    def _create_tracker(self):
        """Create tracker element"""
        tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        if self.config.tracker_config:
            tracker.set_property("ll-config-file", self.config.tracker_config)
        tracker.set_property("tracker-width", 640)
        tracker.set_property("tracker-height", 384)
        return tracker
    
    def _create_secondary_inferences(self):
        """Create secondary inference elements"""
        sgies = []
        if self.config.secondary_model_configs:
            for i, config_path in enumerate(self.config.secondary_model_configs):
                sgie = Gst.ElementFactory.make("nvinfer", f"secondary-inference-{i}")
                sgie.set_property("config-file-path", config_path)
                sgie.set_property("process-mode", 2)  # Process on tracked objects
                sgies.append(sgie)
        return sgies
    
    def _create_video_converter(self):
        """Create video converter"""
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "converter")
        return nvvidconv
    
    def _create_osd(self):
        """Create on-screen display element"""
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        nvosd.set_property("process-mode", 0)
        nvosd.set_property("display-text", 1)
        return nvosd
    
    def _create_sinks(self):
        """Create output sink elements"""
        sinks = []
        
        if self.config.enable_display:
            # Display sink
            display_sink = Gst.ElementFactory.make("nveglglessink", "display-sink")
            display_sink.set_property("sync", False)
            sinks.append(display_sink)
        
        if self.config.enable_rtsp_out:
            # RTSP sink
            rtsp_sink = self._create_rtsp_sink()
            sinks.append(rtsp_sink)
        
        if self.config.enable_file_out:
            # File sink
            file_sink = self._create_file_sink()
            sinks.append(file_sink)
        
        return sinks
    
    def _create_rtsp_sink(self):
        """Create RTSP streaming sink"""
        # This is a simplified version - full RTSP implementation requires more elements
        rtsp_sink = Gst.ElementFactory.make("udpsink", "rtsp-sink")
        rtsp_sink.set_property("host", "127.0.0.1")
        rtsp_sink.set_property("port", self.config.rtsp_port)
        return rtsp_sink
    
    def _create_file_sink(self):
        """Create file output sink"""
        file_sink = Gst.ElementFactory.make("filesink", "file-sink")
        file_sink.set_property("location", self.config.output_file)
        return file_sink
    
    def _link_elements(self, source, parser, decoder, streammux, pgie, tracker, sgies, nvvidconv, nvosd, sinks):
        """Link pipeline elements"""
        # Link source chain
        if parser:
            source.link(parser)
            parser.link(decoder)
        else:
            source.link(decoder)
        
        # Link decoder to mux
        decoder.link(streammux)
        
        # Link inference chain
        current = streammux
        current.link(pgie)
        current = pgie
        
        if tracker:
            current.link(tracker)
            current = tracker
        
        for sgie in sgies:
            current.link(sgie)
            current = sgie
        
        # Link processing chain
        current.link(nvvidconv)
        nvvidconv.link(nvosd)
        
        # Link to sinks
        if len(sinks) == 1:
            nvosd.link(sinks[0])
        elif len(sinks) > 1:
            # Use tee for multiple outputs
            tee = Gst.ElementFactory.make("tee", "tee")
            if self.pipeline:
                self.pipeline.add(tee)
            nvosd.link(tee)
            
            for i, sink in enumerate(sinks):
                queue = Gst.ElementFactory.make("queue", f"queue-{i}")
                if self.pipeline:
                    self.pipeline.add(queue)
                tee.link(queue)
                queue.link(sink)
    
    def _inference_probe_callback(self, pad, info, user_data):
        """Callback to process inference results"""
        if not DEEPSTREAM_AVAILABLE:
            return Gst.PadProbeReturn.OK
            
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        
        # Get batch metadata
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK
        
        # Process each frame in batch
        frame_meta_list = batch_meta.frame_meta_list
        while frame_meta_list:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_meta_list.data)
            
            # Extract detections from frame
            detections = self._extract_detections(frame_meta)
            
            if detections:
                result = {
                    "frame_number": self.frame_count,
                    "timestamp": frame_meta.buf_pts,
                    "detections": detections
                }
                
                self.detection_results.append(result)
                
                # Call callback if provided
                if self.callback:
                    self.callback(result)
            
            self.frame_count += 1
            frame_meta_list = frame_meta_list.next
        
        return Gst.PadProbeReturn.OK
    
    def _extract_detections(self, frame_meta):
        """Extract detection results from frame metadata"""
        detections = []
        
        object_meta_list = frame_meta.obj_meta_list
        while object_meta_list:
            obj_meta = pyds.NvDsObjectMeta.cast(object_meta_list.data)
            
            detection = {
                "class_id": obj_meta.class_id,
                "confidence": obj_meta.confidence,
                "bbox": {
                    "left": obj_meta.rect_params.left,
                    "top": obj_meta.rect_params.top,
                    "width": obj_meta.rect_params.width,
                    "height": obj_meta.rect_params.height
                }
            }
            
            if hasattr(obj_meta, 'object_id'):
                detection["track_id"] = obj_meta.object_id
            
            detections.append(detection)
            object_meta_list = object_meta_list.next
        
        return detections
    
    def _bus_call(self, bus, message):
        """Handle bus messages"""
        t = message.type
        if t == Gst.MessageType.EOS:
            logging.info("End-of-stream reached")
            self.stop()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logging.error(f"Pipeline error: {err}, Debug: {debug}")
            self.stop()
        return True
    
    def start(self):
        """Start the pipeline"""
        if not self.pipeline:
            raise RuntimeError("Pipeline not created")
        
        logging.info("Starting DeepStream pipeline...")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to start pipeline")
        
        self.is_running = True
        
        # Start GLib main loop in separate thread
        self.loop = GObject.MainLoop()
        self.loop_thread = threading.Thread(target=self.loop.run)
        self.loop_thread.daemon = True
        self.loop_thread.start()
        
        logging.info("DeepStream pipeline started successfully")
    
    def stop(self):
        """Stop the pipeline"""
        if self.pipeline and self.is_running:
            logging.info("Stopping DeepStream pipeline...")
            self.pipeline.set_state(Gst.State.NULL)
            self.is_running = False
            
            if self.loop:
                self.loop.quit()
        
        logging.info("DeepStream pipeline stopped")
    
    def get_results(self):
        """Get detection results"""
        return self.detection_results
    
    def clear_results(self):
        """Clear detection results"""
        self.detection_results.clear()
        self.frame_count = 0

class DeepStreamService:
    """High-level DeepStream service for AlgoVision integration"""
    
    def __init__(self):
        self.pipelines: Dict[str, DeepStreamPipeline] = {}
        self.configs: Dict[str, DeepStreamConfig] = {}
        
    def create_pipeline(self, pipeline_id: str, config: DeepStreamConfig, callback: Optional[Callable] = None) -> bool:
        """Create a new DeepStream pipeline"""
        try:
            if pipeline_id in self.pipelines:
                logging.warning(f"Pipeline {pipeline_id} already exists")
                return False
            
            pipeline = DeepStreamPipeline(config, callback)
            self.pipelines[pipeline_id] = pipeline
            self.configs[pipeline_id] = config
            
            logging.info(f"Created DeepStream pipeline: {pipeline_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to create pipeline {pipeline_id}: {e}")
            return False
    
    def start_pipeline(self, pipeline_id: str) -> bool:
        """Start a pipeline"""
        if pipeline_id not in self.pipelines:
            logging.error(f"Pipeline {pipeline_id} not found")
            return False
        
        try:
            self.pipelines[pipeline_id].start()
            return True
        except Exception as e:
            logging.error(f"Failed to start pipeline {pipeline_id}: {e}")
            return False
    
    def stop_pipeline(self, pipeline_id: str) -> bool:
        """Stop a pipeline"""
        if pipeline_id not in self.pipelines:
            logging.error(f"Pipeline {pipeline_id} not found")
            return False
        
        try:
            self.pipelines[pipeline_id].stop()
            return True
        except Exception as e:
            logging.error(f"Failed to stop pipeline {pipeline_id}: {e}")
            return False
    
    def remove_pipeline(self, pipeline_id: str) -> bool:
        """Remove a pipeline"""
        if pipeline_id not in self.pipelines:
            logging.error(f"Pipeline {pipeline_id} not found")
            return False
        
        try:
            self.pipelines[pipeline_id].stop()
            del self.pipelines[pipeline_id]
            del self.configs[pipeline_id]
            logging.info(f"Removed pipeline: {pipeline_id}")
            return True
        except Exception as e:
            logging.error(f"Failed to remove pipeline {pipeline_id}: {e}")
            return False
    
    def get_pipeline_results(self, pipeline_id: str) -> List[Dict]:
        """Get results from a pipeline"""
        if pipeline_id not in self.pipelines:
            return []
        
        return self.pipelines[pipeline_id].get_results()
    
    def list_pipelines(self) -> List[str]:
        """List all pipeline IDs"""
        return list(self.pipelines.keys())
    
    def get_pipeline_status(self, pipeline_id: str) -> Dict:
        """Get pipeline status"""
        if pipeline_id not in self.pipelines:
            return {"status": "not_found"}
        
        pipeline = self.pipelines[pipeline_id]
        return {
            "status": "running" if pipeline.is_running else "stopped",
            "frame_count": pipeline.frame_count,
            "detection_count": len(pipeline.detection_results)
        }

# Global DeepStream service instance
deepstream_service = DeepStreamService()

async def process_video_with_deepstream(video_path: str, video_id: str, callback: Optional[Callable] = None) -> Dict:
    """Process video using DeepStream pipeline"""
    
    # Create configuration
    config = DeepStreamConfig(
        source_type=StreamType.FILE,
        source_path=video_path,
        primary_model_config="configs/yolo_config.txt",  # Configure based on your models
        enable_tracking=True,
        enable_display=False,  # Disable for server processing
        enable_file_out=False
    )
    
    # Create and start pipeline
    pipeline_id = f"video_{video_id}"
    
    if not deepstream_service.create_pipeline(pipeline_id, config, callback):
        raise RuntimeError(f"Failed to create DeepStream pipeline for video {video_id}")
    
    try:
        # Start processing
        if not deepstream_service.start_pipeline(pipeline_id):
            raise RuntimeError(f"Failed to start DeepStream pipeline for video {video_id}")
        
        # Wait for completion (implement proper monitoring)
        await asyncio.sleep(1)  # Placeholder - implement proper completion detection
        
        # Get results
        results = deepstream_service.get_pipeline_results(pipeline_id)
        
        # Generate summary
        summary = await generate_summary_from_deepstream_results(results)
        
        return {
            "video_id": video_id,
            "total_frames": len(results),
            "detections": results,
            "summary": summary
        }
        
    finally:
        # Cleanup
        deepstream_service.remove_pipeline(pipeline_id)

async def generate_summary_from_deepstream_results(results: List[Dict]) -> str:
    """Generate summary from DeepStream results"""
    if not results:
        return "No detections found in video"
    
    # Extract summary statistics
    total_frames = len(results)
    total_detections = sum(len(result.get("detections", [])) for result in results)
    
    # Count objects by class
    class_counts = {}
    for result in results:
        for detection in result.get("detections", []):
            class_id = detection.get("class_id", "unknown")
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    summary_text = f"Video analysis completed with {total_detections} detections across {total_frames} frames. "
    summary_text += f"Detected objects: {dict(class_counts)}"
    
    # Use LLM for natural language summary
    try:
        return await generate_summary(summary_text)
    except Exception as e:
        logging.error(f"Failed to generate LLM summary: {e}")
        return summary_text 