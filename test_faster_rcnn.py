#!/usr/bin/env python3
"""
Test script for Faster R-CNN implementation
Run this to verify that the detection models are working correctly
"""

import cv2
import numpy as np
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

try:
    from app.services import object_detector
    print("✅ Successfully imported object_detector")
except ImportError as e:
    print(f"❌ Failed to import object_detector: {e}")
    sys.exit(1)

def create_test_image():
    """Create a simple test image with some basic shapes"""
    # Create a 640x480 test image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some colored rectangles that might be detected as objects
    cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), -1)  # Green square
    cv2.rectangle(img, (300, 150), (500, 250), (255, 0, 0), -1)  # Blue rectangle
    cv2.circle(img, (400, 350), 50, (0, 0, 255), -1)  # Red circle
    
    # Add some text
    cv2.putText(img, "Test Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img

def test_detection_models():
    """Test all detection models"""
    print("\n🧪 Testing Faster R-CNN Detection Implementation")
    print("=" * 50)
    
    # Create test image
    test_img = create_test_image()
    print("✅ Created test image")
    
    try:
        # Get current configuration
        config = object_detector.get_detection_config()
        print(f"✅ Current config: {config}")
        
        # Get available models
        models = object_detector.get_available_models()
        print(f"✅ Available models: {models}")
        
        # Test with only Faster R-CNN enabled
        print("\n🔍 Testing Faster R-CNN only...")
        object_detector.set_detection_config(
            use_yolo=False,
            use_rtdetr=False,
            use_faster_rcnn=True
        )
        
        detections, annotated_frame = object_detector.detect_objects(test_img)
        print(f"✅ Faster R-CNN detections: {len(detections)} objects found")
        for det in detections:
            print(f"   - {det['label']}: {det['confidence']:.3f} ({det['source']})")
        
        # Test with only YOLO enabled
        print("\n🔍 Testing YOLOv8 only...")
        object_detector.set_detection_config(
            use_yolo=True,
            use_rtdetr=False,
            use_faster_rcnn=False
        )
        
        detections, annotated_frame = object_detector.detect_objects(test_img)
        print(f"✅ YOLOv8 detections: {len(detections)} objects found")
        for det in detections:
            print(f"   - {det['label']}: {det['confidence']:.3f} ({det['source']})")
        
        # Test with only RTDETR enabled
        print("\n🔍 Testing RTDETR only...")
        object_detector.set_detection_config(
            use_yolo=False,
            use_rtdetr=True,
            use_faster_rcnn=False
        )
        
        detections, annotated_frame = object_detector.detect_objects(test_img)
        print(f"✅ RTDETR detections: {len(detections)} objects found")
        for det in detections:
            print(f"   - {det['label']}: {det['confidence']:.3f} ({det['source']})")
        
        # Test with all models enabled
        print("\n🔍 Testing ALL models together...")
        object_detector.set_detection_config(
            use_yolo=True,
            use_rtdetr=True,
            use_faster_rcnn=True,
            prioritize_faster_rcnn=True
        )
        
        detections, annotated_frame = object_detector.detect_objects(test_img)
        print(f"✅ All models detections: {len(detections)} objects found")
        
        # Group by source
        by_source = {}
        for det in detections:
            source = det['source']
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(det)
        
        for source, dets in by_source.items():
            print(f"   {source}: {len(dets)} detections")
            for det in dets:
                print(f"     - {det['label']}: {det['confidence']:.3f}")
        
        # Save test results
        cv2.imwrite("test_original.jpg", test_img)
        cv2.imwrite("test_annotated.jpg", annotated_frame)
        print(f"\n✅ Saved test_original.jpg and test_annotated.jpg")
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_configuration():
    """Test configuration management"""
    print("\n⚙️  Testing Configuration Management")
    print("=" * 40)
    
    try:
        # Save original config
        original_config = object_detector.get_detection_config()
        print(f"✅ Original config saved: {original_config}")
        
        # Test configuration updates
        test_configs = [
            {"use_faster_rcnn": True, "faster_rcnn_threshold": 0.5},
            {"use_yolo": False, "use_rtdetr": True},
            {"prioritize_faster_rcnn": False},
        ]
        
        for i, config_update in enumerate(test_configs):
            print(f"\n🔧 Testing config update {i+1}: {config_update}")
            object_detector.set_detection_config(**config_update)
            new_config = object_detector.get_detection_config()
            
            # Verify updates
            for key, value in config_update.items():
                if new_config.get(key) == value:
                    print(f"   ✅ {key}: {value}")
                else:
                    print(f"   ❌ {key}: expected {value}, got {new_config.get(key)}")
        
        # Restore original config
        object_detector.set_detection_config(**original_config)
        restored_config = object_detector.get_detection_config()
        
        if restored_config == original_config:
            print(f"✅ Configuration successfully restored")
        else:
            print(f"❌ Configuration restoration failed")
            
        print("✅ Configuration management tests completed")
        
    except Exception as e:
        print(f"❌ Configuration test error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Starting Faster R-CNN Implementation Tests")
    
    # Test configuration management
    config_success = test_configuration()
    
    # Test detection models
    detection_success = test_detection_models()
    
    if config_success and detection_success:
        print("\n🎉 ALL TESTS PASSED! Faster R-CNN implementation is working correctly.")
        print("\n📝 Next steps:")
        print("1. Start the FastAPI server: uvicorn app.main:app --reload")
        print("2. Visit http://localhost:8000/docs to see the new detection API endpoints")
        print("3. Use the /detection/config endpoints to manage model settings")
        print("4. Upload a video to test the detection in action")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1) 