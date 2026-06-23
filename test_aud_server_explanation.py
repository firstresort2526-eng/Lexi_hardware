import requests

payload = {"words": [{"closest_char": "家", "line": 0, "second_closest": "回", "text": "回家的路上"}]}

result = requests.post(url="http://127.0.0.1:8000/button_press",json={})
result = requests.post(url="http://127.0.0.1:8000/camera_data",json=payload)