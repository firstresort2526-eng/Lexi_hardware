import easyocr
import cv2
import time

# Initialize reader (只 load 一次)
reader = easyocr.Reader(['ch_tra', 'en'], gpu=False)

# Test
start = time.perf_counter()
img = cv2.imread('temp/cropped_88888.jpg')
result = reader.readtext(img)
end = time.perf_counter()

print(f"Time: {end - start:.2f}s")
for bbox, text, conf in result:
    print(f"'{text}' (conf: {conf:.2f})")
    print(f"  BBox: {bbox}")