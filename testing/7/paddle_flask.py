# paddle_ocr_server.py
from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
import base64
import numpy as np
import cv2
from PIL import Image
import io
import time

# Load model ONCE at startup (not per request)
print("Loading PaddleOCR model...")
start_time = time.time()
ocr = PaddleOCR(lang='ch', use_textline_orientation=True, cpu_threads=4)
print(f"Model loaded in {time.time() - start_time:.2f} seconds")

class Line():
    def __init__(self,point1=None,point2=None,slope=None,b=None):
        if isinstance(slope,(float,int)) and isinstance(b,(float,int)):
            self.slope=slope
            self.b = b
            return
        if hasattr(point1, '__len__') and hasattr(point2, '__len__'):
            print(f"  Case 2: point1 type={type(point1)}, point2 type={type(point2)}")
            self.slope = (point2[1]-point1[1]) / (point2[0]-point1[0])
            self.b = point1[1] - self.slope*point1[0]
            print(f"  Calculated: slope={self.slope}, type={type(self.slope)}, b={self.b}")
            return
        self.slope=None
        self.b=None
    def find_y(self,x):
        print(self.slope, type(self.slope))
        if isinstance(self.slope,float):
            print("find_y")
            return self.slope*x + self.b

def calc_posits(poly, chars):
    length = len(chars) + 1
    topleft = poly[0]
    topright = poly[1]
    bottomright = poly[2]
    bottomleft = poly[3]

    top_char_width = (topright[0] - topleft[0]) / length
    bottom_char_width = (bottomright[0] - bottomleft[0]) / length

    print(topleft,topright,bottomleft,bottomright, sep=",")
    top_horizontal_line = Line(point1=topleft, point2=topright)
    bottom_horizontal_line = Line(point1=bottomleft, point2=bottomright)
    print(top_horizontal_line.find_y(344))

    bottom_points = []
    top_points = []
    for i in range(length):
        bottom_x = bottomleft[0] + bottom_char_width * i
        bottom_points.append((bottom_x, bottom_horizontal_line.find_y(bottom_x)))

        top_x = topleft[0] + top_char_width * i
        top_points.append((top_x, top_horizontal_line.find_y(top_x)))
    
    chars_posits = {}
    for i, char in enumerate(chars):
        chars_posits[char] = [
            top_points[i],      # top-left
            top_points[i+1],    # top-right
            bottom_points[i+1], # bottom-right
            bottom_points[i]    # bottom-left
        ]
    return chars_posits

def calc_distance(point1,point2=(350,350)):
    return abs(point1[1]-point2[1])**2 + abs(point1[0]-point2[0])**2

def find_nearest_char(chars_posits):
    print(chars_posits)
    distances = []
    for key,value in chars_posits.items():
        bottom_line = Line(point1=value[3],point2=value[2])
        middle_x = value[3][0] + (value[2][0] - value[3][0]) / 2
        middle_point = (middle_x,bottom_line.find_y(middle_x))

        distance = (calc_distance(point1 = middle_point),key)
        distances.append(distance)

    distances.sort()
    return distances

def process_image(image_data):
    """Process image and return just the words with their distances"""
    
    # Save temporarily (or you can work with PIL/CV2 directly)
    temp_path = "/tmp/ocr_image.jpg"
    image_data.save(temp_path)
    
    # Run OCR
    prediction = ocr.predict(temp_path)[0]
    
    # Extract just the words with their nearest char info
    results = []
    for i, (poly, text) in enumerate(zip(prediction['rec_polys'], prediction['rec_texts'])):
        distances = find_nearest_char(calc_posits(poly, text))
        
        # Format: just the words with their closest character info
        words_info = {
            'line': i,
            'text': text,
            'closest_char': distances[0][1] if distances else None,
            'second_closest': distances[1][1] if len(distances) > 1 else None
        }
        results.append(words_info)
    
    return results

# Create Flask app
app = Flask(__name__)

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    """
    Expects JSON with base64 image:
    {
        "image": "base64_encoded_string"
    }
    Returns:
    {
        "words": [
            {
                "line": 0,
                "text": "recognised_text",
                "closest_char": "nearest_text",
                "second_closest": "2nd_nearest_text"
            },
            ...
        ]
    }
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_base64 = data['image']
        
        # Remove header if present (e.g., "data:image/jpeg;base64,")
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Process the image
        results = process_image(image)
        
        # Return just the words (you can customize what you need)
        return jsonify({
            'words': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    # Run the server
    app.run(host='127.0.0.1', port=5000, debug=False)