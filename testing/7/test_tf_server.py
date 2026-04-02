from PIL import Image
import requests
import base64

IMAGE_PATH = "test_img/IMG_8289.jpg"
with open(IMAGE_PATH, "rb") as f:
    base64_string = base64.b64encode(f.read()).decode('utf-8')
payload = {'image':base64_string, 'plot':True}

url = "http://127.0.0.1:3000/detect"
result = requests.post(url, json=payload)
print(result.json())