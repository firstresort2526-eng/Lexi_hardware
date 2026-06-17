import cv2
import time

# Open the default USB camera (0 = first camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# --- Optional: Set resolution (try 1920x1080 or 2560x1440) ---
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)

# --- Optional: Set exposure manually ---
# 0.25 = manual mode (varies by driver), 0.75 = auto mode
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
# cap.set(cv2.CAP_PROP_EXPOSURE, -4)  # Try values like -6, -4, -2, 0

cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)      # Set to manual mode
cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 3800)    # 5500K is "daylight" – neutral

# Give the camera a moment to adjust settings
time.sleep(0.5)

# Capture a single frame
ret, frame = cap.read()

if ret:
    # Save the image
    cv2.imwrite("capture.jpg", frame)
    print("Image saved as capture.jpg")
    
    # Print the resolution of the captured frame
    height, width = frame.shape[:2]
    print(f"Resolution: {width}x{height}")
else:
    print("Error: Failed to capture image.")

# Release the camera
cap.release()