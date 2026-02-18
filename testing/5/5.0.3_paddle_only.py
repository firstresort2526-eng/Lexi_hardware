print("Entire code start.")
from paddleocr import PaddleOCR
print("Finished importing.")

def mainCode():
    print("started running.")
    # Initialize OCR
    ocr = PaddleOCR(lang='ch', use_angle_cls=True,cpu_threads=1)
    print("Model init.")
    # Run OCR
    result = ocr.predict('001.png')
    print("Predicted.")
    print(result)
    
    print("OCR completed successfully!")

if __name__ == '__main__':
    print("Starting...")
    mainCode()
    print("Done!")