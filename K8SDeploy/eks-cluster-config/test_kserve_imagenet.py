import requests
import json
# cat-classifier
url_1 = "http://a8aa96e3f69824ea4b8edc06479f50ca-1028354565.ap-south-1.elb.amazonaws.com/v1/models/cat-classifier:predict"

# url_1 = "http://a8aa96e3f69824ea4b8edc06479f50ca-1028354565.ap-south-1.elb.amazonaws.com/v1/models/sports-classifier:predict"

# url_1 = "http://a8aa96e3f69824ea4b8edc06479f50ca-1028354565.ap-south-1.elb.amazonaws.com/v1/models/cat-classifier:predict"
# a8aa96e3f69824ea4b8edc06479f50ca-1028354565.ap-south-1.elb.amazonaws.com
# url_2 = "http://a86a7b22b8c09441b8ce16f1f94e6d45-2006895197.ap-south-1.elb.amazonaws.com/v1/models/dog-classifier:predict"
# kubectl exec -it pods/sports-classifier-predictor-00001-deployment-75579c7d84-trj95  -- /bin/bash
# url_3 = "http://a6c04fc26da384b0298877ccc6573dd0-1917169771.ap-south-1.elb.amazonaws.com/v1/models/dog-classifier:predict"
# with open("input.json") as f:
#     payload = json.load(f)

with open("input.json") as f:
    payload = json.load(f)

# headers_1 = {"Host": "imagenet-classifier-1.default.emlo.tsai", "Content-Type": "application/json"}

headers_1 = {"Host": "imagenet-classifier.default.emlo.tsai", "Content-Type": "application/json"}

response_1 = requests.request("POST", url_1, headers=headers_1, json=payload)

# headers_2 = {"Host": "imagenet-classifier-2.default.emlo.tsai", "Content-Type": "application/json"}

# response_2 = requests.request("POST", url_2, headers=headers_2, json=payload)

# headers_3 = {"Host": "sports-classifier-2.default.emlo.tsai", "Content-Type": "application/json"}

# response_3 = requests.request("POST", url_3, headers=headers_3, json=payload)

print(response_1.headers)
print(response_1.status_code)
print(response_1.json())

# print("##############")

# print(response_2.headers)
# print(response_2.status_code)
# print(response_2.json())

# print(response_3.headers)
# print(response_3.status_code)
# print(response_3.json())

# http://food-classifier.default.emlo.tsai