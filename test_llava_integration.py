#!/usr/bin/env python3
"""
Test script for LLaVA integration with Groq API
"""

import os
import asyncio
import cv2
import numpy as np
from app.services.llava_groq_service import (
    analyze_frame_with_llava, 
    encode_image_to_base64,
    create_default_prompts,
    generate_video_summary_with_llava
)

async def test_llava_integration():
    """Test the LLaVA integration with a simple test image."""
    
    # Check if API key is set
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY environment variable not set!")
        print("Please set it using:")
        print("1. For Windows (PowerShell): $env:GROQ_API_KEY='your_api_key_here'")
        print("2. For Linux/MacOS: export GROQ_API_KEY=your_api_key_here")
        return False
    
    print("🧪 Testing LLaVA integration with Groq API...")
    
    # Create a simple test image (colored rectangles)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some colored shapes
    cv2.rectangle(test_image, (50, 50), (200, 150), (0, 255, 0), -1)  # Green rectangle
    cv2.rectangle(test_image, (250, 100), (400, 200), (255, 0, 0), -1)  # Blue rectangle
    cv2.circle(test_image, (500, 300), 50, (0, 0, 255), -1)  # Red circle
    cv2.putText(test_image, "TEST IMAGE", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Test image encoding
    print("📸 Testing image encoding...")
    try:
        image_base64 = encode_image_to_base64(test_image)
        print(f"✅ Image encoded successfully (length: {len(image_base64)} chars)")
    except Exception as e:
        print(f"❌ Image encoding failed: {e}")
        return False
    
    # Test frame analysis
    print("🔍 Testing frame analysis with LLaVA...")
    test_prompt = "Describe what you see in this test image. What shapes and colors are visible?"
    
    try:
        analysis_result = await analyze_frame_with_llava(test_image, test_prompt)
        print(f"✅ Frame analysis completed!")
        print(f"📝 Analysis result: {analysis_result[:200]}...")
        
        if "error" in analysis_result.lower():
            print(f"⚠️ Analysis contains error: {analysis_result}")
            return False
            
    except Exception as e:
        print(f"❌ Frame analysis failed: {e}")
        return False
    
    # Test prompt creation
    print("📝 Testing prompt creation...")
    try:
        prompts = create_default_prompts(3)
        print(f"✅ Created {len(prompts)} prompts")
        for i, prompt in enumerate(prompts):
            print(f"   {i+1}: {prompt[:50]}...")
    except Exception as e:
        print(f"❌ Prompt creation failed: {e}")
        return False
    
    # Test video summary generation
    print("📊 Testing video summary generation...")
    try:
        mock_analyses = [
            "The image shows a green rectangle on the left side and a blue rectangle in the center.",
            "There is a red circle on the right side of the image with white text below.",
            "The overall composition includes geometric shapes in primary colors on a black background."
        ]
        
        mock_metadata = {
            "duration": 10.0,
            "fps": 30.0,
            "total_frames": 300,
            "analyzed_frames": 3,
            "has_zone_restriction": False
        }
        
        summary = await generate_video_summary_with_llava(mock_analyses, mock_metadata)
        print(f"✅ Video summary generated!")
        print(f"📄 Summary: {summary[:200]}...")
        
        if "error" in summary.lower():
            print(f"⚠️ Summary contains error: {summary}")
            return False
            
    except Exception as e:
        print(f"❌ Video summary generation failed: {e}")
        return False
    
    print("\n🎉 All LLaVA integration tests passed!")
    print("✅ Ready to process videos with LLaVA via Groq API")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_llava_integration())
    exit(0 if success else 1) 