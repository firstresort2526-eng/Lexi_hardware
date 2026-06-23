import requests

url = "http://0.0.0.0:8000/button_press"
payload = {}
respond = requests.post(url,json=payload)
print(respond.json())