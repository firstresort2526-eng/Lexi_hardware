import pytesseract
import cv2
import matplotlib.pyplot as plt

# Configure Tesseract path if needed (Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Read image
img = cv2.imread('image.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape

# Get character boxes (NO recognition)
boxes = pytesseract.image_to_boxes(
    img_rgb,
    lang='chi_sim',
    config='--psm 6'
)

# Draw boxes on image
for box in boxes.splitlines():
    box = box.split(' ')
    if len(box) >= 6:
        x1 = int(box[1])
        y1 = int(box[2]) 
        x2 = int(box[3])
        y2 = int(box[4])
        
        # Convert coordinates (Tesseract uses bottom-left origin)
        y1 = h - y1
        y2 = h - y2
        
        # Draw rectangle
        cv2.rectangle(img_rgb, (x1, y2), (x2, y1), (0, 255, 0), 2)

# Display
plt.figure(figsize=(12, 8))
plt.imshow(img_rgb)
plt.axis('off')
plt.title(f'Detected {len(boxes.splitlines())} Chinese characters')
plt.show()