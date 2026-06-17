# server_url is "127.0.0.1/3141"

import cv2
import time
from flask import Flask
import base64
import requests

app = Flask(__name__)
image_processing_server_url = "http://127.0.0.1:3000/detect"

def init():
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Set resolution 2048*1536 (This is max resolution given by taobaoguy)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75) # Set 曝光 to auto
    cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 3800) # Set the 色温

    # Give the camera a moment to adjust settings
    time.sleep(0.5)
    return cap

@app.route("/capture",methods=['GET'])
def capture(): # Also will send to the server alr
    cap = init()
    ret, frame = cap.read()

    if ret:
        # Save the image
        success, encoded_image = cv2.imencode('.jpg', frame)
        if success:
            # Convert to base64 bytes
            base64_string = base64.b64encode(encoded_image).decode('utf-8')
            payload = {'image':base64_string, 'plot':True}
            result = requests.post(image_processing_server_url, json=payload)
            return result.json()
        else:
            print("Error: Fail to encode image")
    else:
        print("Error: Failed to capture image.")

# Release the camera
@app.route("/close",methods=['GET'])
def release_buffer():
    cap.release()

if __name__ == "__main__":
    app.run(debug=True, port=3141)