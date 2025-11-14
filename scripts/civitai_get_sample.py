import requests, json
from pprint import pprint

url = "https://civitai.com/api/v1/models?types=LORA&limit=1"
response = requests.get(url, timeout=10)
data = response.json()

# 1개 모델의 JSON 구조 확인
pprint(data["items"][0])

with open("sample_civitai.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)