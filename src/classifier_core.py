# src/classifier_core.py
"""
CLIP 기반 LoRA 스타일 군집 핵심 로직

이 모듈은 다음 기능을 제공한다.
1) CLIP 모델 로드
2) 이미지 -> 임베딩
3) 텍스트 -> 임베딩
4) 클러스터 centroid 로드
5) 임베딩과 centroid 유사도 계산 하여 cluster id 예측

다른 스크립트(Streamlit UI, 실시간 LoRA 분류, 단발성 신규 LoRA 분류)는
이 모듈만 import해서 사용하면 된다.

참고 자료:
- Hugging Face CLIP 모델: https://huggingface.co/openai/clip-vit-base-patch32
- CLIP 모델 자료: https://johnowhitaker.github.io/tglcourse/clip.html
- CLIP은 Transformer를 돌리지만, 다른 Transformer 모델과 달리 토큰별 임베딩을 반환하는 것이 아니라 512 임베딩 차원 하나만을 반환한다.
- CLIP의 목적은 텍스트와 이미지를 동일한 임베딩 공간에서 비교하는 것.
- 이미 정규화된 벡터를 다시 정규화해도 값은 달라지지 않는다.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor

# CLIP 모델 로드
def load_clip(model_name="openai/clip-vit-base-patch32"):
    """
    인자:
        model_name (str, optional): "openai/clip-vit-base-patch32"

    반환값:
        모델, 전처리기, 디바이스
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device


# 이미지 임베딩 생성 (PIL 이미지 모듈의 이미지 클래스)
def embed_image(img: Image.Image, model, processor, device):
    """
    인자:
        img (Image.Image): 이미지 객체
        model: 모델
        processor: 전처리기
        device: 디바이스 변수

    반환값:
        emb: L2 정규화된 이미지 임베딩 (512,)
    """
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        # 임베딩 출력값이 총합 1이 되도록 L2 정규화 
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    # CLIP 모델은 배치(batch) 차원을 포함해서 (1, 512) 형태로 반환하므로 [0]으로 첫 번째 축 잘라내기
    return emb.cpu().numpy()[0]  # (512,)


# 텍스트 임베딩 생성
def embed_text(text: str, model, processor, device):
    """
    인자:
        text (str): 프롬프트 문자열
        model: 모델
        processor: 전처리기
        device: 디바이스 변수

    반환값:
        emb: L2 정규화된 텍스트 임베딩 (512,)
    """
    inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]  # (512,)


# centroid 로드
def load_centroids(path: Path):
    return np.load(path)  # shape: (k, 512)


# 클러스터 예측
def predict_cluster(embedding: np.ndarray, centroids: np.ndarray):
    """
    embedding: (512,)
    centroids: (k, 512)

    가장 유사도 높은 cluster index 반환
    
    참고사항
    - 원래 코사인 유사도는 두 벡터 간의 내적을 각 벡터의 L2 노름을 곱한 값으로 나눈 값
    - 그렇지만 embedding과 centroids가 모두 L2 정규화된 상태라면, 코사인 유사도는 L2 정규화된 벡터 간의 내적(dot product)으로 계산 가능
    """
    # np.isclose로 embedding의 L2 노름을 구하고 1에 가깝지 않으면 L2 정규화
    if not np.isclose(np.linalg.norm(embedding), 1.0):
        embedding = embedding / np.linalg.norm(embedding)

    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / centroid_norms

    sims = centroids @ embedding
    return int(np.argmax(sims))


# 텍스트 프롬프트 기반 추천기
def recommend_cluster_by_prompt(prompt: str, model, processor, device, centroids):
    text_emb = embed_text(prompt, model, processor, device)
    return predict_cluster(text_emb, centroids)