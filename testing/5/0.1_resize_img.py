from PIL import Image

image = Image.open("IMG_8261.jpg")
target_size = (500,666)
resized = image.resize(target_size)
resized.save('resize_8261.jpg')