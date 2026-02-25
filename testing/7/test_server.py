'''
    Expects JSON with base64 image:
    {
        "image": "base64_encoded_string"
    }"
'''
import requests
import base64

with open("resize_8261.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

url = "http://127.0.0.1:5000/ocr"
payload = {"image":encoded_string}

try:
    result = requests.post(url, json=payload)
    
    if result.status_code == 200:
        print("Success!")
        print(result.json())  # This will show the words
    else:
        print(f"Error {result.status_code}: {result.text}")
        
except requests.exceptions.ConnectionError:
    print("Could not connect to server. Is it running?")