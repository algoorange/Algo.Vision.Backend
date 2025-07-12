#!/usr/bin/env python3
"""
Test script for NVIDIA DeepStream integration
"""

import cv2
import time
import numpy as np
from app.services.object_detector import detect_objects_with_fast_rcnn, DEEPSTREAM_AVAILABLE, deepstream_detector

def test_deepstream_integration():
    """Test DeepStream integration with sample frames"""
    print("🧪 Testing NVIDIA DeepStream Integration")
    print("=" * 50)
    
    # Check DeepStream availability
    print(f"DeepStream Available: {DEEPSTREAM_AVAILABLE}")
    if DEEPSTREAM_AVAILABLE:
        print(f"DeepStream Initialized: {deepstream_detector.initialized}")
    
    # Create a test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some test objects (colored rectangles)
    cv2.rectangle(test_frame, (50, 50), (200, 150), (255, 0, 0), -1)  # Blue rectangle
    cv2.rectangle(test_frame, (300, 200), (500, 350), (0, 255, 0), -1)  # Green rectangle
    cv2.rectangle(test_frame, (100, 300), (250, 400), (0, 0, 255), -1)  # Red rectangle
    
    # Add some text
    cv2.putText(test_frame, "DeepStream Test Frame", (200, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    print("\n🎯 Running Object Detection Test")
    print("-" * 30)
    
    # Test detection performance
    num_tests = 10
    total_time = 0
    
    for i in range(num_tests):
        start_time = time.time()
        
        try:
            detections, annotated_frame = detect_objects_with_fast_rcnn(test_frame.copy())
            
            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time
            
            print(f"Test {i+1}: {inference_time:.3f}s - {len(detections)} objects detected")
            
        except Exception as e:
            print(f"Test {i+1}: Error - {e}")
    
    # Calculate average performance
    avg_time = total_time / num_tests
    fps = 1.0 / avg_time
    
    print(f"\n📊 Performance Results:")
    print(f"Average inference time: {avg_time:.3f}s")
    print(f"Average FPS: {fps:.1f}")
    
    # Save test results
    cv2.imwrite("test_frame_original.jpg", test_frame)
    if 'annotated_frame' in locals():
        cv2.imwrite("test_frame_annotated.jpg", annotated_frame)
        print(f"✅ Test frames saved: test_frame_original.jpg, test_frame_annotated.jpg")
    
    return avg_time, fps

def benchmark_comparison():
    """Compare DeepStream vs PyTorch performance"""
    print("\n🏁 Benchmarking DeepStream vs PyTorch")
    print("=" * 50)
    
    # Create test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test PyTorch Faster R-CNN
    print("Testing PyTorch Faster R-CNN...")
    from app.services.crack_detector2 import detect_objects_faster_rcnn
    
    pytorch_times = []
    for i in range(5):
        start_time = time.time()
        detections, _ = detect_objects_faster_rcnn(test_frame.copy())
        end_time = time.time()
        pytorch_times.append(end_time - start_time)
    
    avg_pytorch_time = np.mean(pytorch_times)
    pytorch_fps = 1.0 / avg_pytorch_time
    
    print(f"PyTorch Faster R-CNN: {avg_pytorch_time:.3f}s ({pytorch_fps:.1f} FPS)")
    
    # Test DeepStream (if available)
    if DEEPSTREAM_AVAILABLE and deepstream_detector.initialized:
        print("Testing DeepStream...")
        
        deepstream_times = []
        for i in range(5):
            start_time = time.time()
            detections, _ = detect_objects_with_fast_rcnn(test_frame.copy())
            end_time = time.time()
            deepstream_times.append(end_time - start_time)
        
        avg_deepstream_time = np.mean(deepstream_times)
        deepstream_fps = 1.0 / avg_deepstream_time
        
        print(f"DeepStream: {avg_deepstream_time:.3f}s ({deepstream_fps:.1f} FPS)")
        
        # Calculate speedup
        speedup = avg_pytorch_time / avg_deepstream_time
        print(f"\n🚀 DeepStream Speedup: {speedup:.1f}x faster")
    else:
        print("⚠️  DeepStream not available for comparison")

def main():
    """Main test function"""
    print("🔬 NVIDIA DeepStream Integration Test")
    print("=" * 60)
    
    try:
        # Test basic integration
        avg_time, fps = test_deepstream_integration()
        
        # Run benchmark comparison
        benchmark_comparison()
        
        print(f"\n✅ Test completed successfully!")
        print(f"Your system achieved {fps:.1f} FPS with the current configuration")
        
        # Recommendations
        if fps < 10:
            print("\n💡 Performance Tips:")
            print("- Install NVIDIA DeepStream for better performance")
            print("- Use a more powerful GPU (RTX 3060 or better)")
            print("- Reduce input resolution for faster processing")
        elif fps < 30:
            print("\n💡 Good performance! Consider:")
            print("- Using INT8 precision for even faster inference")
            print("- Increasing batch size for multiple streams")
        else:
            print("\n🎉 Excellent performance! Your system is ready for real-time processing")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Please check the DEEPSTREAM_SETUP.md guide for troubleshooting")

if __name__ == "__main__":
    main() 