# 1. Import the library
from inference_sdk import InferenceHTTPClient
import time,os
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# Create the custom layer
@tf.keras.utils.register_keras_serializable()
class HeatmapToCoords(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        size = input_shape[-2]

        # Create the grids
        x = tf.linspace(0.0, 1.0, size)
        y = tf.linspace(0.0, 1.0, size)
        x_grid, y_grid = tf.meshgrid(x, y, indexing='xy')

        # use tf.constant so that it's untrainable for sure
        self.x_grid = tf.constant(x_grid, dtype=tf.float32)
        self.y_grid = tf.constant(y_grid, dtype=tf.float32)

    def call(self, inputs):
        # inputs shape is (batch, h, w, 2)
        x_heatmap = inputs[..., 0]
        y_heatmap = inputs[..., 1]

        batch = tf.shape(inputs)[0]
        x_flat = tf.reshape(x_heatmap, (batch, -1)) # Flatten to make the softmax easier (by default do it at the last axis)
        y_flat = tf.reshape(y_heatmap, (batch, -1))

        x_prob = tf.nn.softmax(x_flat, axis=-1)
        x_prob = tf.reshape(x_prob, tf.shape(x_heatmap))
        y_prob = tf.nn.softmax(y_flat, axis=-1)
        y_prob = tf.reshape(y_prob, tf.shape(y_heatmap))

        # Do elementwise matmul, and sum them
        x_pred = tf.reduce_sum(x_prob * self.x_grid, axis=[1,2])
        y_pred = tf.reduce_sum(y_prob * self.y_grid, axis=[1,2])

        # Return stacked coordinates
        return tf.stack([x_pred, y_pred], axis=-1)  # (batch, 2)

    def get_config(self):
        config = super().get_config()
        return config


startTime = time.perf_counter()

IMAGE_PATH = "./images/IMG_8289.jpg"

sam_client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="OJfl7aWHfksWqfvdvE4U"
) # Create the HTTP Client

init_img = Image.open(IMAGE_PATH)
init_size = init_img.size # (width, height)
new_size = (300,400) # Keep in mind that we resized the img
new_image = init_img.resize(new_size)
new_image.save("./temp/new_img.jpg")

# Run the workflow!
result = sam_client.run_workflow(
    workspace_name="caspar9872",
    workflow_id="sam3-with-prompts",
    images={
        "image": "./temp/new_img.jpg"
    },
    parameters={
        "prompts": ["pencil"]
    },
    use_cache=True # Speeds up repeated requests
)

endTime = time.perf_counter() # Calc the time used for sam
sam_prediction = result[0]["sam"]["predictions"][0]
print(sam_prediction)

def decode_coords(num,axis):
    if axis=='x':
        return num/300.0*init_size[0]
    elif axis=='y':
        return num/400.0*init_size[1]

# Crop the photos
crop_width = sam_prediction['width']
crop_height = sam_prediction['height']
xmin = decode_coords(sam_prediction['x'] - crop_width/2,'x') # These are initsized coords alr
ymin = decode_coords(sam_prediction['y'] - crop_height/2,'y')
xmax = decode_coords(sam_prediction['x'] + crop_width/2, 'x')
ymax = decode_coords(sam_prediction['y'] + crop_height/2, 'y')
init_crop_width = xmax-xmin
init_crop_height = ymax-ymin

cropped_pencil = init_img.crop((xmin,ymin,xmax,ymax))
cropped_pencil = ImageOps.pad(cropped_pencil, (480,480), color='white')
cropped_pencil.save("temp/cropped.jpg")

cropped_pencil_np = np.asarray(cropped_pencil)
cropped_pencil_np = cropped_pencil_np/255.0

img_tensor = tf.convert_to_tensor(cropped_pencil_np)
expanded_dim = tf.expand_dims(img_tensor,axis=0)

loaded_model = tf.keras.models.load_model("best_model4.keras")
prediction = loaded_model.predict(expanded_dim)[0]*480
print(prediction)

# now we need to convert the predictions (which is 480*480) back to the original size
pad_ratio = max(init_crop_height,init_crop_width)/480
init_cropped_tip_predictions = prediction*pad_ratio # The non-normalized coordinates of the pencil tip with padding.

# Make sure that the prediction isn't at the white padding region
if init_crop_width>init_crop_height:
    x_padding = 0
    y_padding = (480 - init_crop_height/init_crop_width*480)/2
    dim = (0, # If width fills the screen, ofc xmin is 0
           y_padding, # 480/init_crop_width is the ratio. Ex. (480,960) becomes (240,480). 0.5 is the ratio. So, 480/960*480 is indeed 240. Then, do 480-height, so u get the padding size, then /2, gives u the upper padding
           480, # Width fills the screen so it's 480
           480-y_padding, # Just 480 (bottom) minus the lower padding area.
           )
else:
    x_padding = (480 - init_crop_width/init_crop_height*480)/2
    y_padding = 0
    dim = (x_padding,
           0,
           480-x_padding,
           480,
           )
    
# X and Y now is just the paddlized normalized coords (0-480)

x = min(dim[2],max(prediction[0],dim[0]))
y = min(dim[3],max(prediction[1],dim[1]))

x = x-x_padding
y = y-y_padding
# Now x and y is the coordinates in the normalized cropped image but not counting the padding

init_x = xmin + x*pad_ratio
init_y = ymin + y*pad_ratio


def plot_image(image,coords):
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    plt.scatter(coords[0],coords[1])
    plt.show()

plot_image(cropped_pencil_np,prediction)
plot_image(init_img,(init_x,init_y))

try:
    os.remove("./temp/cropped.jpg")
except Exception as e:
    print(f"Exception: {e}")