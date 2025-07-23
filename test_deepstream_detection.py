#!/usr/bin/env python3
"""
Test script to debug DeepStream detection
Run this inside your DeepStream container to see why detection is failing
"""

import os
import sys
import subprocess

print("🔍 DeepStream Detection Test")
print("=" * 50)

# Test 1: Environment variables
print("\n1. Environment Variables:")
deepstream_env = os.getenv('NVIDIA_DEEPSTREAM_VERSION')
print(f"   NVIDIA_DEEPSTREAM_VERSION: {deepstream_env}")

# Test 2: DeepStream paths
print("\n2. DeepStream Installation Paths:")
deepstream_paths = [
    '/opt/nvidia/deepstream',
    '/usr/lib/x86_64-linux-gnu/gstreamer-1.0/deepstream',
    '/opt/nvidia/deepstream/deepstream',
    '/opt/nvidia/deepstream/deepstream-7.1'
]

for path in deepstream_paths:
    exists = os.path.exists(path)
    print(f"   {path}: {'✅ EXISTS' if exists else '❌ NOT FOUND'}")

# Test 3: Python bindings
print("\n3. Python Bindings:")
try:
    import gi
    print("   gi module: ✅ Available")
    
    try:
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
        print("   GStreamer: ✅ Available")
        
        # Initialize GStreamer
        Gst.init(None)
        print("   GStreamer init: ✅ Success")
        
    except Exception as e:
        print(f"   GStreamer: ❌ Error - {e}")
        
    try:
        import pyds
        print("   pyds (DeepStream): ✅ Available")
    except ImportError as e:
        print(f"   pyds (DeepStream): ❌ Not available - {e}")
        
except ImportError as e:
    print(f"   gi module: ❌ Not available - {e}")

# Test 4: Docker detection
print("\n4. Docker Detection:")
try:
    result = subprocess.run(['docker', '--version'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print(f"   Docker: ✅ Available - {result.stdout.strip()}")
    else:
        print("   Docker: ❌ Command failed")
except Exception as e:
    print(f"   Docker: ❌ Not available - {e}")

# Test 5: Check if we're in a container
print("\n5. Container Detection:")
container_indicators = [
    ('/.dockerenv', 'Docker environment file'),
    ('/proc/1/cgroup', 'Process cgroup (check for docker)'),
]

for path, desc in container_indicators:
    if os.path.exists(path):
        print(f"   {desc}: ✅ Found at {path}")
        if path == '/proc/1/cgroup':
            try:
                with open(path, 'r') as f:
                    content = f.read()
                    if 'docker' in content:
                        print("     → Docker container detected")
                    else:
                        print("     → Not a Docker container")
            except:
                pass
    else:
        print(f"   {desc}: ❌ Not found")

# Test 6: Check working directory and expected files
print("\n6. Working Directory Check:")
print(f"   Current dir: {os.getcwd()}")
expected_files = ['app', 'requirements.txt', 'docker-compose.yml']
for file in expected_files:
    exists = os.path.exists(file)
    print(f"   {file}: {'✅ EXISTS' if exists else '❌ NOT FOUND'}")

print("\n" + "=" * 50)
print("🎯 Run this script inside your DeepStream container to debug!")
print("Usage in container:")
print("   python3 test_deepstream_detection.py") 