import cv2
import numpy as np
from paddleocr import PaddleOCR

# Load image
img = cv2.imread("temp/cropped_88888.jpg")
height, width = img.shape[:2]

# Run PaddleOCR
ocr = PaddleOCR(lang='ch', use_textline_orientation=True, cpu_threads=4)
result = ocr.predict(img)[0]

# Draw center (where pencil tip is)
center_x, center_y = width // 2, height // 2
cv2.circle(img, (center_x, center_y), 8, (0, 0, 255), -1)
cv2.line(img, (center_x - 40, center_y), (center_x + 40, center_y), (0, 0, 255), 2)
cv2.line(img, (center_x, center_y - 40), (center_x, center_y + 40), (0, 0, 255), 2)

# Draw bounding boxes and centers
for poly, text in zip(result['rec_polys'], result['rec_texts']):
    pts = np.array(poly, dtype=np.int32)
    cv2.polylines(img, [pts], True, (0, 255, 0), 2)
    
    # Character center
    cx = int((poly[0][0] + poly[2][0]) / 2)
    cy = int((poly[0][1] + poly[2][1]) / 2)
    cv2.circle(img, (cx, cy), 4, (255, 0, 0), -1)
    
    # Label with text and distance
    dist = (cx - center_x)**2 + (cy - center_y)**2
    cv2.putText(img, f"{text} {dist:.0f}", (cx - 10, cy - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

# Save output
cv2.imwrite("ocr_boxes.jpg", img)
print("Saved to ocr_boxes.jpg")