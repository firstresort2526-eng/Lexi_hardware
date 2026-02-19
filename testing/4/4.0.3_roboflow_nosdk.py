import requests
import base64
import json, os
from pprint import pprint
Api_key = os.getenv('ROBOFLOW_API_KEY')

def run_sam_workflow(image_path, api_key):
    # Upload image to temporary hosting or use base64
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()
    
    payload = {
        "api_key": api_key,
        "inputs": {
            "image": {
                "type": "base64",
                "value": img_base64
            },
            "prompts": ["pencil"]
        }
    }
    
    response = requests.post(
        "https://serverless.roboflow.com/caspar9872/workflows/sam3-with-prompts",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    return response.json()['outputs']

result = run_sam_workflow("IMG_8261.jpg", 'OJfl7aWHfksWqfvdvE4U').outputs
sam_prediction = result  # Adjust path based on actual response structure
print(json.dumps(sam_prediction, indent=2)[:1000])  # First 1000 chars only