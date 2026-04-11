from flask import Flask, request, jsonify
import os
import dotenv
import base64
import tensorflow as tf
import requests, json
from io import BytesIO
from tensorflow.keras import layers
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from inference_sdk import InferenceHTTPClient

dotenv.load_dotenv()
api_key = os.getenv('ROBOFLOW_API_KEY') # Get Roboflow API key

print(api_key)
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

def run_sam_workflow(image_path):    
    result = client.run_workflow(
        workspace_name="caspar9872",
        workflow_id="sam3-with-prompts",
        images={
            "image": image_path # Path to your image file
        },
        parameters={
            "prompts": ["pencil"]
        },
        use_cache=True # Speeds up repeated requests
    )
    return result

# Define the Tensorflow custom layer HeatmapToCoords
@tf.keras.utils.register_keras_serializable()
class HeatmapToCoords(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        size = input_shape[-2]
        x = tf.linspace(0.0, 1.0, size)
        y = tf.linspace(0.0, 1.0, size)
        x_grid, y_grid = tf.meshgrid(x, y, indexing='xy')
        self.x_grid = tf.constant(x_grid, dtype=tf.float32)
        self.y_grid = tf.constant(y_grid, dtype=tf.float32)

    def call(self, inputs):
        x_heatmap = inputs[..., 0]
        y_heatmap = inputs[..., 1]
        batch = tf.shape(inputs)[0]
        x_flat = tf.reshape(x_heatmap, (batch, -1))
        y_flat = tf.reshape(y_heatmap, (batch, -1))
        x_prob = tf.nn.softmax(x_flat, axis=-1)
        x_prob = tf.reshape(x_prob, tf.shape(x_heatmap))
        y_prob = tf.nn.softmax(y_flat, axis=-1)
        y_prob = tf.reshape(y_prob, tf.shape(y_heatmap))
        x_pred = tf.reduce_sum(x_prob * self.x_grid, axis=[1,2])
        y_pred = tf.reduce_sum(y_prob * self.y_grid, axis=[1,2])
        return tf.stack([x_pred, y_pred], axis=-1)

    def get_config(self):
        config = super().get_config()
        return config
    
def decode_coords(num,axis,init_size):
    if axis=='x':
        return num/300.0*init_size[0]
    elif axis=='y':
        return num/400.0*init_size[1]
    
def plot_image(image,coords):
    print("   44.1 Inside plot_image")
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    plt.scatter(coords[0],coords[1])
    plt.show()
    plt.savefig("./temp/output.jpg")
    print("   44.2 Plot saved")
    
# Load the tensorflow model
model = tf.keras.models.load_model("best_model4.keras",custom_objects={'HeatmapToCoords': HeatmapToCoords})

app = Flask(__name__)

@app.route('/detect',methods=['POST'])
def process():
    param = request.get_json()

    base64_string = param.get('image')
    plot = param.get('plot')
    init_img = Image.open(BytesIO(base64.b64decode(base64_string)))
    init_size = init_img.size
    new_size = (300,400)
    new_image = init_img.resize(new_size)
    new_image.save("./temp/new_img.jpg")

    # Detect the bounding box for a pencil
    result = run_sam_workflow('./temp/new_img.jpg')
    sam_prediction = result[0]["sam"]["predictions"][0]

    # Grab all those coordinates from the sam prediction (the original format is so ma fan)
    crop_width = sam_prediction['width']
    crop_height = sam_prediction['height']
    xmin = decode_coords(sam_prediction['x'] - crop_width/2,'x',init_size)
    ymin = decode_coords(sam_prediction['y'] - crop_height/2,'y',init_size)
    xmax = decode_coords(sam_prediction['x'] + crop_width/2, 'x',init_size)
    ymax = decode_coords(sam_prediction['y'] + crop_height/2, 'y',init_size)
    init_crop_width = xmax-xmin
    init_crop_height = ymax-ymin

    cropped_pencil = init_img.crop((xmin,ymin,xmax,ymax))
    cropped_pencil = ImageOps.pad(cropped_pencil, (480,480), color='white') # Add padding to the image for TF prediction
    cropped_pencil.save("temp/cropped.jpg")

    cropped_pencil_np = np.asarray(cropped_pencil)
    cropped_pencil_np = cropped_pencil_np/255.0 # Normalize the colors, very important!
    img_tensor = np.expand_dims(cropped_pencil_np, axis=0).astype(np.float32)

    prediction = model.predict(img_tensor)[0]*480 # Run the TF model, becuz the result is normalized, we need to *480

    pad_ratio = max(init_crop_height,init_crop_width)/480 # Calculate the ratio between padded image and original one

    if init_crop_width>init_crop_height:
        x_padding = 0
        y_padding = (480 - init_crop_height/init_crop_width*480)/2
        dim = (0, y_padding, 480, 480-y_padding)
    else:
        x_padding = (480 - init_crop_width/init_crop_height*480)/2
        y_padding = 0
        dim = (x_padding, 0, 480-x_padding, 480)
    print(f"41. Padding: x={x_padding}, y={y_padding}")

    print("42. Calculating final coordinates...")
    x = min(dim[2],max(prediction[0],dim[0]))
    y = min(dim[3],max(prediction[1],dim[1]))
    x = x-x_padding
    y = y-y_padding
    init_x = xmin + x*pad_ratio # Prediction of x coordinate with the coordinate system of the original img
    init_y = ymin + y*pad_ratio

    if plot:
        print("plotting cropped image:")
        plot_image(cropped_pencil_np,prediction) # Cropped image
        print("plotting original image:")
        plot_image(init_img,(init_x,init_y)) # Original image

    # Crop the target area for the OCR
    target_area = (max(init_x-1000, 0), max(init_y-1000),min(init_x+1000,init_size[0]),min(init_y+1000,init_size[1]))
    cropped_img = init_img.crop(target_area)
    cropped_img.save('temp/cropped_88888.jpg')

    server_url = "http://127.0.0.1:5000/ocr"

    with open('temp/cropped_88888.jpg','rb') as f:
        encoded_string = base64.b64encode(f.read()).decode('utf-8')

    payload = {"image":encoded_string}
    audie_url = "http://localhost:8000/speak"

    # Let's handle the respond (Rare time when i use try except)
    try:
        results = requests.post(url=server_url,json=payload)
        if results.status_code == 200:
            print("yay")
            result_json = results.json()
            print(result_json)
            # Now I need to send a post request to audie's FastAPI server
            # Output from her server is {"status": "processing", "text": text}
            aud_result = requests.post(url=audie_url,json=result_json)
            print(aud_result.json())
            return jsonify({"status": "success", "coordinates": {"x": init_x, "y": init_y}})
        else:
            print(f"Error code: {results.status_code}")
            return jsonify({"status": "error", "message": f"OCR failed with code {results.status_code}"}), results.status_code
    except Exception as e:
        print(f"Exception caught: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
        

if __name__ == "__main__":
    app.run(debug=True, port=3000)