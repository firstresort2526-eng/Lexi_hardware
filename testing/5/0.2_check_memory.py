import psutil
import os
import paddle
import numpy as np
import sys

print("=" * 60)
print("PADDLEOCR MEMORY & VERSION DIAGNOSTIC")
print("=" * 60)

print("\n=== SYSTEM CHECK ===")
print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")
print(f"RAM Usage: {psutil.virtual_memory().percent}%")
print(f"CPU Count: {psutil.cpu_count()}")

print("\n=== PADDLE VERSION CHECK ===")
print(f"Paddle version: {paddle.__version__}")
print(f"CUDA available: {paddle.is_compiled_with_cuda()}")
print(f"CUDA devices: {paddle.device.cuda.device_count() if paddle.is_compiled_with_cuda() else 0}")

# For Paddle 3.x, get compile info differently
try:
    if paddle.is_compiled_with_cuda():
        print(f"CUDA version: {paddle.version.cuda()}")
    print(f"Commit: {paddle.version.commit() if hasattr(paddle.version, 'commit') else 'N/A'}")
except Exception as e:
    print(f"Could not get detailed version info: {e}")

print("\n=== BASIC PADDLE TEST ===")
try:
    # Simple tensor operation
    test_tensor = paddle.ones([100, 100])
    result = test_tensor + test_tensor
    print("✓ Basic paddle operation: SUCCESS")
    
    # Check if GPU is being used
    if paddle.is_compiled_with_cuda():
        print(f"✓ GPU is available and compiled")
        print(f"  Default device: {paddle.get_device()}")
    else:
        print("ℹ Running on CPU only")
except Exception as e:
    print(f"✗ Paddle basic test FAILED: {e}")

print("\n=== MEMORY STRESS TEST ===")
print("Gradually increasing tensor size to test memory limits:")

sizes = [100, 500, 1000, 2000, 3000, 4000, 5000]
for size in sizes:
    try:
        # Calculate approximate memory needed (float32 = 4 bytes)
        mem_needed = size * size * 4 / (1024**2)  # MB
        print(f"  Testing {size}x{size} tensor ({mem_needed:.1f} MB)...", end=" ", flush=True)
        
        tensor = paddle.ones([size, size])
        # Do some operation to ensure it's actually allocated
        tensor = tensor * 2
        print(f"✓ SUCCESS")
        
        # Clean up
        del tensor
    except Exception as e:
        print(f"✗ FAILED at {size}x{size}: {str(e)[:50]}")
        break

print("\n=== PADDLEOCR LOAD TEST ===")
try:
    import paddleocr
    print(f"PaddleOCR version: {paddleocr.__version__}")
    
    # Test loading without actually running OCR
    print("Loading PaddleOCR (this may take a moment)...")
    ocr = paddleocr.PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
    print("✓ PaddleOCR loaded successfully")
    
    # Test with tiny dummy image
    print("Testing with tiny dummy image...")
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    result = ocr.ocr(dummy_img, cls=False)
    print("✓ Tiny image test SUCCESSFUL")
    
except ImportError:
    print("✗ PaddleOCR not installed")
except Exception as e:
    print(f"✗ PaddleOCR test FAILED: {e}")

print("\n=== ENVIRONMENT INFO ===")
print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")
print(f"Executable: {sys.executable}")

# Check for common issues
print("\n=== POTENTIAL ISSUES CHECK ===")
issues_found = False

if psutil.virtual_memory().available < 2 * 1024**3:  # Less than 2GB free
    print("⚠ WARNING: Low memory available (< 2GB)")
    issues_found = True

if not paddle.is_compiled_with_cuda():
    print("ℹ Note: Running on CPU only (this is fine unless you need GPU)")

# Check OpenCV version
try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
    if cv2.__version__ > '4.8.0':
        print("⚠ WARNING: OpenCV version > 4.8.0 may cause issues")
        issues_found = True
except:
    print("ℹ OpenCV not imported")

if not issues_found:
    print("✓ No obvious issues detected")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)