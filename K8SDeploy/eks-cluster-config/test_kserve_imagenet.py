import requests
import json
# cat-classifier
url = "http://af4a6e56f32e24ae6a9bb54c05c92ba4-808383676.ap-south-1.elb.amazonaws.com/v1/models/cat-classifier:predict"

with open("input.json") as f:
    payload = json.load(f)
headers = {"Host": "imagenet-classifier.default.emlo.tsai", "Content-Type": "application/json"}

response = requests.request("POST", url, headers=headers, json=payload)

print(response.headers)
print(response.status_code)
print(response.json())

# http://food-classifier.default.emlo.tsai