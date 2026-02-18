from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
import cv2
import numpy as np

def mainCode():
    # 1. Initialize OCR
    ocr = PaddleOCR(lang='ch')

    # 2. Run OCR
    result = ocr.predict('001.png')

    # 3. Load image and draw bounding boxes
    img = cv2.imread('001.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Draw boxes
    for line in result[0]:
        box = np.array(line[0]).astype(int)
        print(line)
        # Draw polygon
        plt.plot([box[0,0], box[1,0]], [box[0,1], box[1,1]], 'r-', linewidth=2)
        plt.plot([box[1,0], box[2,0]], [box[1,1], box[2,1]], 'r-', linewidth=2)
        plt.plot([box[2,0], box[3,0]], [box[2,1], box[3,1]], 'r-', linewidth=2)
        plt.plot([box[3,0], box[0,0]], [box[3,1], box[0,1]], 'r-', linewidth=2)

    # Show/save
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('output/output.jpg', bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == '__main__':
    print("hello!")
    mainCode()