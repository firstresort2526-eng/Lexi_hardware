import requests
import dotenv,os
from PIL import Image
from io import BytesIO
import time

dotenv.load_dotenv()
API_key = os.getenv('ACCESS_KEY')
url = "https://api.unsplash.com/search/photos?query=貓&per_page=1"
header = {'Authorization': f"Client-ID {API_key}"}

starttime = time.perf_counter()
result = requests.get(url,headers=header).json()
image_url = result['results'][0]['urls']['small']  # 400px width

image_response = requests.get(image_url)
img = Image.open(BytesIO(image_response.content))
img_320 = img.resize((320, 320), Image.Resampling.LANCZOS)
endtime = time.perf_counter()
img_320.show()
print(endtime-starttime)