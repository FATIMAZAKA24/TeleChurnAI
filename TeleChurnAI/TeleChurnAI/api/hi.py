import requests

url = "http://127.0.0.1:8000/api/single-predict/"
response = requests.post(url, json={})
print(response.json())
