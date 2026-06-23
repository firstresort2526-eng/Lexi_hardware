import requests
print(requests.get(
    "https://ipv4.webshare.io/",
    proxies={
        "http": "38.154.203.95:5863/",
        "https": "38.154.203.95:5863/"
    }
).text)