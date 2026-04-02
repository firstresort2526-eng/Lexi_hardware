import base64
from picamera2 import Picamera2
from flask import Flask
import io
import requests

app = Flask(__name__)
picam2 = Picamera2()
tf_url = 'http://127.0.0.1:3000'

def init():
    config = picam2.create_preview_configuration(main={"size": (3280, 2464)})  # Use preview config for speed
    picam2.configure(config)
    picam2.start()
    picam2.options["quality"] = 100

init()

# Capture to in-memory bytes buffer
@app.route("/capture",methods=['GET'])
def capture():
    buffer = io.BytesIO()
    picam2.capture_file(buffer, format='jpeg')
    base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    payload={'image':base64_str,'plot':True}
    results = requests.post(url=tf_url, json=payload)
    print(results.json()) # print it just for debug

@app.route("/close",methods=['GET'])
def terminate():
    picam2.stop()
    print("Done.")

@app.route("/init",methods=['GET'])
def initiate():
    init()

if __name__ == "__main__":
    app.run(debug=True, port=3141) # Pi as port number just for fun, cuz 3000 5000 8000 all used