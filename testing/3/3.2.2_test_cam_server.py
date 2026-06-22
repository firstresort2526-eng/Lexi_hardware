import requests
result = requests.get("http://127.0.0.1:3141/capture")
print(result.status_code)
print(result.json)