# src/classify_new_lora.py
import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
from src.config import Config

"""
새로운 LoRA 모델을 다운로드하지 않고, 대표 이미지를 CivitAI에서 직접 가져와
CLIP 임베딩 후 기존 클러스터의 centroid와 비교하여
어느 스타일 클러스터인지 자동 분류하기 위한 모듈입니다.
"""

CENTROID_PATH = Config.OUTPUT_DIR / "clusters" / "cluster_centroids.npy"
centroids = np.load(CENTROID_PATH)   # (k, 512)

# CLIP 로드
device = Config.DEVICE
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(model_name)

def get_lora_preview_image(model_id: int):
    """
    CivitAI 모델 ID를 받아 대표 이미지 URL을 반환.
    """
    url = f"https://civitai.com/api/v1/models/{model_id}"
    res = requests.get(url, timeout=10)
    res.raise_for_status()
    data = res.json()
    
    try:
        img_url = data["modelVersions"][0]["images"][0]["url"]
        return img_url
    except:
        return None

def get_clip_embedding(image: Image.Image):
    """
    PIL 이미지를 CLIP 임베딩 (512) 로 변환
    """
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.cpu().numpy().flatten()

def classify_lora(model_id: int):
    """
    LoRA ID 입력 → 스타일 클러스터 자동 분류
    """
    img_url = get_lora_preview_image(model_id)
    if img_url is None:
        return None, "이미지를 찾을 수 없음"

    img_bytes = requests.get(img_url).content
    image = Image.open(BytesIO(img_bytes)).convert("RGB")

    emb = get_clip_embedding(image)  # (512,)

    # cosine similarity
    sims = centroids @ emb / (
        np.linalg.norm(centroids, axis=1) * np.linalg.norm(emb)
    )
    cluster_id = int(np.argmax(sims))
    return cluster_id, img_url

if __name__ == "__main__":
    cid, url = classify_lora(12345)
    print("Cluster:", cid)
    print("Image URL:", url)
