# 1. Import the library
from inference_sdk import InferenceHTTPClient
import time,json,os
import base64
from PIL import Image
from io import BytesIO

startTime = time.perf_counter()
# 2. Connect to your workflow
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="OJfl7aWHfksWqfvdvE4U"
)

prediction_img = Image.open("./images/IMG_8261.jpg")
new_size = (300,400)
new_image = prediction_img.resize(new_size)
new_image.save("./temp/prediction_img.jpg")

# 3. Run your workflow on an image
result = client.run_workflow(
    workspace_name="caspar9872",
    workflow_id="sam3-with-prompts",
    images={
        "image": "./temp/prediction_img.jpg"
    },
    parameters={
        "prompts": ["pencil"]
    },
    use_cache=True # Speeds up repeated requests
)
endTime = time.perf_counter()

# 4. Get your results

print(result[0]["sam"]["predictions"])
print('\n\n\n\n\n\n\n\n')
print(startTime-endTime)

image_data = base64.b64decode(result[0]["bounding_boxes"])
file_data = BytesIO(image_data)
img = Image.open(file_data)

with open('result1.json','a') as f:
    json.dump(result[0]["sam"]["predictions"],f,indent=4)

try:
    os.remove("./temp/prediction_img.jpg")
except Exception as e:
    print(f"Exception: {e}")