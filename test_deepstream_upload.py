#!/usr/bin/env python3
"""
Test script to demonstrate DeepStream video analysis
"""
import requests
import time
import os

# API base URL
BASE_URL = "http://localhost:8000"

def test_deepstream_health():
    """Check if DeepStream is available"""
    response = requests.get(f"{BASE_URL}/deepstream/health")
    print("🔍 DeepStream Health Check:")
    print(f"Status: {response.json()}")
    return response.json()

def process_video_with_deepstream(video_file_path):
    """Upload and process video with DeepStream"""
    print(f"\n🎬 Processing video: {video_file_path}")
    
    # Upload and process video
    with open(video_file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/deepstream/process-video", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Video processed successfully!")
        print(f"Video ID: {result['video_id']}")
        print(f"Results: {result.get('results', 'No results')}")
        return result
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None

def create_pipeline_for_video(video_path):
    """Create a DeepStream pipeline for detailed video analysis"""
    print(f"\n🔧 Creating DeepStream pipeline for: {video_path}")
    
    # Create pipeline
    pipeline_config = {
        "source_type": "file",
        "source_path": video_path,
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "batch_size": 1,
        "enable_tracking": True,
        "enable_display": False,
        "enable_rtsp_out": False
    }
    
    response = requests.post(f"{BASE_URL}/deepstream/create-pipeline", json=pipeline_config)
    
    if response.status_code == 200:
        pipeline_info = response.json()
        pipeline_id = pipeline_info['pipeline_id']
        print(f"✅ Pipeline created: {pipeline_id}")
        
        # Start pipeline
        start_response = requests.post(f"{BASE_URL}/deepstream/start-pipeline/{pipeline_id}")
        if start_response.status_code == 200:
            print("🚀 Pipeline started!")
            
            # Monitor pipeline
            monitor_pipeline(pipeline_id)
            
            return pipeline_id
        else:
            print(f"❌ Failed to start pipeline: {start_response.text}")
    else:
        print(f"❌ Failed to create pipeline: {response.text}")
    
    return None

def monitor_pipeline(pipeline_id):
    """Monitor pipeline progress"""
    print(f"\n📊 Monitoring pipeline: {pipeline_id}")
    
    for i in range(10):  # Monitor for 10 iterations
        response = requests.get(f"{BASE_URL}/deepstream/pipeline-status/{pipeline_id}")
        if response.status_code == 200:
            status = response.json()
            print(f"Status: {status['status']}, Frames: {status['frame_count']}, Detections: {status['detection_count']}")
            
            if status['status'] == 'completed':
                print("🎉 Pipeline completed!")
                break
        
        time.sleep(2)  # Wait 2 seconds between checks
    
    # Get final results
    results_response = requests.get(f"{BASE_URL}/deepstream/pipeline-results/{pipeline_id}")
    if results_response.status_code == 200:
        results = results_response.json()
        print(f"📋 Final results: {results['total_detections']} detections")
        return results
    
    return None

def list_all_pipelines():
    """List all active pipelines"""
    response = requests.get(f"{BASE_URL}/deepstream/pipelines")
    if response.status_code == 200:
        pipelines = response.json()
        print(f"\n📋 Active pipelines: {len(pipelines['pipelines'])}")
        for pipeline in pipelines['pipelines']:
            print(f"  - {pipeline['pipeline_id']}: {pipeline['status']}")
        return pipelines
    return None

if __name__ == "__main__":
    print("🎯 DeepStream Video Analysis Test")
    print("=" * 40)
    
    # 1. Check DeepStream health
    health = test_deepstream_health()
    
    if not health.get('deepstream_available', False):
        print("⚠️ DeepStream not available. Using fallback processing.")
        print("For GPU acceleration, run with DeepStream container:")
        print("  docker-compose -f docker-compose.yml up --build")
        exit(1)
    
    # 2. Test simple video processing
    video_path = input("\n📁 Enter path to video file (or press Enter for test): ").strip()
    if not video_path:
        print("💡 Use the API documentation at http://localhost:8000/docs")
        print("💡 Upload a video file using the /deepstream/process-video endpoint")
    else:
        if os.path.exists(video_path):
            # Quick processing
            result = process_video_with_deepstream(video_path)
            
            # Advanced pipeline processing
            pipeline_id = create_pipeline_for_video(video_path)
            
            # List all pipelines
            list_all_pipelines()
        else:
            print(f"❌ File not found: {video_path}") 