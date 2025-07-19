import sys
sys.path.insert(0, '/workspace')

print('Testing core dependencies...')

# Test system packages
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    Gst.init(None)
    print('✅ GStreamer: OK')
except Exception as e:
    print(f'❌ GStreamer: {e}')

# Test web framework
try:
    import fastapi, uvicorn
    print('✅ FastAPI/Uvicorn: OK')
except Exception as e:
    print(f'❌ FastAPI/Uvicorn: {e}')

# Test computer vision
try:
    import cv2, numpy as np, torch
    print('✅ Computer Vision: OK')
except Exception as e:
    print(f'❌ Computer Vision: {e}')

# Test tracking
try:
    from deep_sort_realtime import DeepSort
    print('✅ Object Tracking: OK')
except Exception as e:
    print(f'❌ Object Tracking: {e}')

# Test your application
try:
    from app.services.deepstream_service import DEEPSTREAM_AVAILABLE
    print(f'✅ DeepStream Service: Available = {DEEPSTREAM_AVAILABLE}')
except Exception as e:
    print(f'❌ DeepStream Service: {e}')

try:
    from app.main import app
    print('✅ AlgoVision App: OK')
except Exception as e:
    print(f'❌ AlgoVision App: {e}')

print('Import tests complete!') 