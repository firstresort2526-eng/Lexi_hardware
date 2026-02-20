from paddleocr import PaddleOCR
import matplotlib.pyplot as plt

# A bunch of functions! Yay
class Line():
    def __init__(self,point1=None,point2=None,slope=None,b=None):
        if isinstance(slope,(float,int)) and isinstance(b,(float,int)):
            self.slope=slope
            self.b = b
            return
        if hasattr(point1, '__len__') and hasattr(point2, '__len__'):
            print(f"  Case 2: point1 type={type(point1)}, point2 type={type(point2)}")
            self.slope = (point2[1]-point1[1]) / (point2[0]-point1[0])
            self.b = point1[1] - self.slope*point1[0] # y-mx
            print(f"  Calculated: slope={self.slope}, type={type(self.slope)}, b={self.b}")
            return
        self.slope=None
        self.b=None
    def find_y(self,x):
        print(self.slope, type(self.slope))
        if isinstance(self.slope,float):
            print("find_y")
            return self.slope*x + self.b # y=mx+b
    


def calc_posits(poly, chars):
    length = len(chars) + 1  # number of points = number of chars + 1
    topleft = poly[0]
    topright = poly[1]
    bottomright = poly[2]
    bottomleft = poly[3]

    top_char_width = (topright[0] - topleft[0]) / length
    bottom_char_width = (bottomright[0] - bottomleft[0]) / length
    #left_char_height = (topleft[1] - bottomleft[1]) / length
    #right_char_height = (topright[1] - bottomright[1]) / length

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

def calc_distance(point1,point2=(250,0)):
    return abs(point1[1]-point2[1])**2 + abs(point1[0]-point2[0])**2

def find_nearest_char(chars_posits):
    #min_d = float('inf')
    #best_char = None
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

# Init the model
ocr = PaddleOCR(lang='ch', use_textline_orientation=True,cpu_threads=4)

prediction = ocr.predict('resize_8261.jpg')[0]

for i, (poly, text) in enumerate(zip(prediction['rec_polys'], prediction['rec_texts'])):
    print(f"\nLine {i}: '{text}'")
    print(poly)
    distances = find_nearest_char(calc_posits(poly, text))
    print(f"  Closest: {distances[0][1]} (dist: {distances[0][0]:.2f})")
    print(f"  2nd closest: {distances[1][1]} (dist: {distances[1][0]:.2f})")

#line = Line(point1 = [344,97],point2 = [832,90])
#print(line.find_y(344))