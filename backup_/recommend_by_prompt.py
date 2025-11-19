# src/recommend_by_prompt.py
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from src.config import Config

"""
텍스트 프롬프트를 입력하면
CLIP text encoder로 임베딩을 만들고,
각 클러스터 centroid와 cosine similarity를 계산하여
가장 적합한 스타일 클러스터를 추천한다.
"""

CENTROID_PATH = Config.OUTPUT_DIR / "clusters" / "cluster_centroids.npy"
centroids = np.load(CENTROID_PATH)

device = Config.DEVICE
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(model_name)

def recommend_cluster(prompt: str):
    """
    프롬프트 -> 클러스터 추천
    """
    # clip_processor로 프롬프트 텍스트를 토큰화, 패딩, attention mask 생성, 텐서 변환
    tokenized = clip_processor(text=prompt, return_tensors="pt").to(device)
    
    # CLIP 모델로 텍스트 임베딩 생성
    inputs = clip_model.get_text_features(
        **tokenized)
    
    with torch.no_grad():
        # 임베딩이 종합 1이 되도록 L2 정규화
        text_emb = inputs / inputs.norm(p=2, dim=-1, keepdim=True)

    text_emb = text_emb.cpu().numpy().flatten()

    sims = centroids @ text_emb / (
        np.linalg.norm(centroids, axis=1) * np.linalg.norm(text_emb)
    )
    cluster_id = int(np.argmax(sims))
    return cluster_id, sims

if __name__ == "__main__":
    cid, sims = recommend_cluster("a dreamy soft anime portrait")
    print("추천 클러스터:", cid)
