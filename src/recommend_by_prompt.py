# src/recommend_by_prompt.py
"""
프롬프트 기반 스타일 클러스터 추천 모듈

기능
1. CLIP 텍스트 인코더로 프롬프트를 임베딩으로 변환
2. 미리 계산된 클러스터 중심(centroids.npy)과 유사도 비교
3. 가장 가까운 클러스터 ID를 반환

사용처
- Streamlit / 웹 UI에서 "프롬프트 -> 스타일 클러스터" 추천
- 콘솔에서 간단히 프롬프트를 넣고 어떤 클러스터인지 확인
"""

from pathlib import Path

import numpy as np

from src.config import Config
from src.classifier_core import (
    load_clip,
    load_centroids,
    recommend_cluster_by_prompt,
)

# 전역 캐시: 한 번만 로드하고 재사용
_clip_model = None
_clip_processor = None
_device = None
_centroids = None


def _ensure_loaded():
    """
    CLIP 모델과 centroids를 한 번만 로드하고,
    이후 호출에서는 캐시된 전역 변수를 재사용한다.
    """
    global _clip_model, _clip_processor, _device, _centroids

    if _clip_model is None:
        # CLIP 모델 / 프로세서 / 디바이스 로드
        _clip_model, _clip_processor, _device = load_clip()

    if _centroids is None:
        centroids_path = Config.OUTPUT_DIR / "clusters" / "cluster_centroids.npy"
        _centroids = load_centroids(centroids_path)


def recommend_cluster(prompt: str) -> int:
    """
    텍스트 프롬프트를 받아 가장 유사한 스타일 클러스터 ID를 반환한다.

    인자
    prompt : str
        사용자가 입력한 자연어 프롬프트

    반환값
    int
        가장 유사도가 높은 클러스터의 ID (0 ~ k-1)
    """
    _ensure_loaded()
    cluster_id = recommend_cluster_by_prompt(
        prompt=prompt,
        model=_clip_model,
        processor=_clip_processor,
        device=_device,
        centroids=_centroids,
    )
    return cluster_id


if __name__ == "__main__":
    # 콘솔 테스트용 진입점
    while True:
        text = input("\n프롬프트를 입력하세요 (종료: exit): ").strip()
        if text.lower() in {"exit"}:
            break
        cid = recommend_cluster(text)
        print(f"추천 클러스터 ID: {cid}")
