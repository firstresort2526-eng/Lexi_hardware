from PIL import Image

image = Image.open("IMG_8261.jpg")
target_size = (900,2900)
resized = image.crop(box=(400,2800,target_size[0],target_size[1]))
resized.save('resize_8261.jpg')