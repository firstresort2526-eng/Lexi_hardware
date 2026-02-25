print("Entire code start.")
from paddleocr import PaddleOCR
print("Finished importing.")

def mainCode():
    print("started running.")
    # Initialize OCR
    ocr = PaddleOCR(lang='ch', use_angle_cls=True,cpu_threads=4)
    print("Model init.")
    # Run OCR
    result = ocr.predict('001.png')
    print("Predicted.")
    polys = result[0]['rec_polys'][0]
    print('\n\n\n\n\n\n\n\n\n\n\n\nPolys:\n')
    for i in polys:
        print(i)
    print('\n\n\n\n Length:')
    print(len(polys))
    
    words = result[0]['rec_texts']
    print('\n\n\n\n\n\n\n\n\n\n\n\n\Words:\n')
    print(words)
    print('\n\n\n\nLength:')
    print(len(words))
    print(result)

    print("OCR completed successfully!")

if __name__ == '__main__':
    print("Starting...")
    mainCode()
    print("Done!")