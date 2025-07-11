import requests
url = "http://127.0.0.1:8001/shiny_dashboard/"  # Adjust if needed
# Use GET instead of POST
response = requests.get(url)
# Print response content (HTML page)
print(response.text)
