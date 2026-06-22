from PIL import Image

IMAGE_PATH = "temp/cropped_88888.jpg"
image = Image.open(IMAGE_PATH)
resolution = (int(image.size[0]/5*2), int(image.size[1]/5*2))
SAVE_PATH = f"temp/{resolution[0]}x{resolution[1]}.jpg"

# Resize the image
resized_image = image.resize(resolution)

# Save the resized image
resized_image.save(SAVE_PATH)