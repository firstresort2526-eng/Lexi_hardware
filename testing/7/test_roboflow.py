# 1. Import the library
from inference_sdk import InferenceHTTPClient

# 2. Connect to your workflow
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="OJfl7aWHfksWqfvdvE4U"
)

# 3. Run your workflow on an image
result = client.run_workflow(
    workspace_name="caspar9872",
    workflow_id="sam3-with-prompts",
    images={
        "image": "test_img/IMG_8289.jpg" # Path to your image file
    },
    parameters={
        "prompts": ["pencil"]
    },
    use_cache=True # Speeds up repeated requests
)

# 4. Get your results
print(result)
