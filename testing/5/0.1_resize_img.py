from PIL import Image

image = Image.open("001.png")
target_size = (400,400)
resized = image.resize(target_size)
resized.save('002.png')