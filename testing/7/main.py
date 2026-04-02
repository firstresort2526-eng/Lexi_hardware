print("1. Starting imports...")

from picamera2 import Picamera2
print("2. Imported picamera2")

import time, os, base64, json
print("3. Imported time, os, base64, json")

from PIL import Image, ImageOps
print("4. Imported PIL")

import numpy as np
print("5. Imported numpy")

import matplotlib.pyplot as plt
print("6. Imported matplotlib")

import tensorflow as tf
print("8. Imported tensorflow")

from tensorflow.keras import layers
print("9. Imported keras layers")

import requests

from dotenv import load_dotenv

load_dotenv()

print("10. Creating Picamera2 instance...")
picam2 = Picamera2()
print("11. Picamera2 instance created")

config = picam2.create_preview_configuration(main={"size": (3024, 4032)})
print("12. Config created")

picam2.configure(config)
print("13. Camera configured")

picam2.start()
print("14. Camera started")

api_key = os.getenv('ROBOFLOW_API_KEY')
print("15. Got API key")

print("16. Defining run_sam_workflow function...")
def run_sam_workflow(image_path, api_key):
    print("  16.1 Inside run_sam_workflow")
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()
    print("  16.2 Image encoded")
    
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
    print("  16.3 Payload created")
    
    response = requests.post(
        "https://serverless.roboflow.com/caspar9872/workflows/sam3-with-prompts",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    print("  16.5 Request sent")
    print(json.dumps(response.json())[0:1000])
    return response.json()['outputs']

print("17. Defining HeatmapToCoords layer...")
@tf.keras.utils.register_keras_serializable()
class HeatmapToCoords(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("    17.1 HeatmapToCoords init")

    def build(self, input_shape):
        print(f"    17.2 Building with input_shape: {input_shape}")
        size = input_shape[-2]
        x = tf.linspace(0.0, 1.0, size)
        y = tf.linspace(0.0, 1.0, size)
        x_grid, y_grid = tf.meshgrid(x, y, indexing='xy')
        self.x_grid = tf.constant(x_grid, dtype=tf.float32)
        self.y_grid = tf.constant(y_grid, dtype=tf.float32)
        print("    17.3 Grids created")

    def call(self, inputs):
        print("    17.4 Call method")
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

print("18. Loading model...")
model = tf.keras.models.load_model("best_model4.keras",custom_objects={'HeatmapToCoords': HeatmapToCoords})
print("19. Model loaded")

print("22. Starting photo capture...")
startTime = time.perf_counter()
picam2.options["quality"] = 100
picam2.capture_file(f"temp/IMG_88888.jpg")
print("23. Photo captured")

print("24. Processing image for SAM...")
IMAGE_PATH = "temp/IMG_88888.jpg" #"temp/IMG_88888.jpg"
init_img = Image.open(IMAGE_PATH)
init_size = init_img.size
new_size = (300,400)
new_image = init_img.resize(new_size)
new_image.save("./temp/new_img.jpg")
print("25. Image resized and saved")

print("26. Running SAM workflow...")
result = run_sam_workflow('./temp/new_img.jpg',api_key)
print("27. SAM workflow completed")

sam_prediction = result[0]["sam"]["predictions"][0]
print(f"28. SAM prediction: {sam_prediction}")

print("29. Defining decode_coords function...")
def decode_coords(num,axis):
    if axis=='x':
        return num/300.0*init_size[0]
    elif axis=='y':
        return num/400.0*init_size[1]

print("30. Calculating crop coordinates...")
crop_width = sam_prediction['width']
crop_height = sam_prediction['height']
xmin = decode_coords(sam_prediction['x'] - crop_width/2,'x')
ymin = decode_coords(sam_prediction['y'] - crop_height/2,'y')
xmax = decode_coords(sam_prediction['x'] + crop_width/2, 'x')
ymax = decode_coords(sam_prediction['y'] + crop_height/2, 'y')
init_crop_width = xmax-xmin
init_crop_height = ymax-ymin
print(f"31. Crop area: ({xmin}, {ymin}, {xmax}, {ymax})")

print("32. Cropping pencil...")
cropped_pencil = init_img.crop((xmin,ymin,xmax,ymax))
cropped_pencil = ImageOps.pad(cropped_pencil, (480,480), color='white')
cropped_pencil.save("temp/cropped.jpg")
print("33. Pencil cropped and saved")

print("34. Preparing for model prediction...")
cropped_pencil_np = np.asarray(cropped_pencil)
cropped_pencil_np = cropped_pencil_np/255.0
img_tensor = np.expand_dims(cropped_pencil_np, axis=0).astype(np.float32)
print("35. Tensor prepared")

print("36. Running model prediction...")
prediction = model.predict(img_tensor)[0]*480
print(f"37. Model prediction: {prediction}")

print("38. Calculating pad ratio...")
pad_ratio = max(init_crop_height,init_crop_width)/480
init_cropped_tip_predictions = prediction*pad_ratio
print(f"39. Pad ratio: {pad_ratio}")

print("40. Calculating padding...")
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
init_x = xmin + x*pad_ratio
init_y = ymin + y*pad_ratio
print(f"43. Final coordinates: ({init_x}, {init_y})")

print("44. Defining plot_image function...")
def plot_image(image,coords):
    print("   44.1 Inside plot_image")
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    plt.scatter(coords[0],coords[1])
    plt.show()
    plt.savefig("./temp/output.jpg")
    print("   44.2 Plot saved")

print("45. Plotting cropped image...")
plot_image(cropped_pencil_np,prediction)
print("46. Cropped image plotted")

print("47. Plotting original image...")
plot_image(init_img,(init_x,init_y))
print("48. Original image plotted")

print("49. Cropping target area for OCR...")
target_area = (max(init_x-350, 0), max(0,init_y-350), min(3024,init_x+350), min(4032,init_y))
cropped_img = init_img.crop(target_area)
cropped_img.save('temp/cropped_88888.jpg')
print("50. Target area cropped and saved")

# Now beacuse we used flask for the stupid paddleocr, we can finally just call a http requests and sleep happily forever afeter
server_url = "http://127.0.0.1:5000/ocr"

with open('temp/cropped_88888.jpg','rb') as f:
    encoded_string = base64.b64encode(f.read()).decode('utf-8')

payload = {"image":encoded_string}

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
    else:
        print(f"Error code: {results.status_code}")
except Exception as e:
    print(f"Exception caught: {e}")

print("59. Calculating end time...")
endTime = time.perf_counter()
print(f"60. Total time: {endTime - startTime:.2f} seconds")

print("61. Stopping camera...")
picam2.stop()
print("62. Camera stopped")