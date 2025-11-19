# src/auto_classify_civitai.py
"""
실시간 LoRA 자동 분류 스크립트

기능:
1. CivitAI API에서 LoRA 모델 가져오기
2. 대표 이미지 다운로드
3. classifier_core.py를 이용해 CLIP 임베딩 -> cluster 자동 예측
4. 결과를 CSV로 저장

사용 사례:
- 새로 올라오는 LoRA를 실시간으로 스타일 클러스터에 자동 태깅
- 분류 서버 or Streamlit 대시보드에서 활용
"""

import os
import csv
import time
import requests
from io import BytesIO
from pathlib import Path
from PIL import Image

from config import Config
from classifier_core import (
    load_clip,
    embed_image,
    load_centroids,
    predict_cluster
)

# 설정값

# API 호출 간 대기 시간
REQUEST_INTERVAL = 3.0

# 한 번의 API 요청당 가져올 모델 수
LIMIT = 50

# 정렬 기준: "Newest"로 두면 항상 최신 순으로 온다.
SORT = "Newest"

# 이미지 저장 위치
SAVE_DIR = Config.OUTPUT_DIR / "auto_classified"
CSV_PATH = Config.OUTPUT_DIR / "auto_classified" / "auto_results.csv"

SAVE_DIR.mkdir(parents=True, exist_ok=True)


# CivitAI LoRA 모델 가져오기
def fetch_lora_models(limit=LIMIT):
    """
    CivitAI API에서 최신 LoRA 목록을 가져옵니다.

    인자:
        limit (int): 한 번에 가져올 모델 수 (최대 100)

    반환값:
        list[dict]: CivitAI 모델 JSON의 "items" 리스트
    """
    # 베이스 URL
    url = "https://civitai.com/api/v1/models"
    # 쿼리스트링
    params = {
        "types": "LORA",
        "sort": SORT,
        "limit": limit,
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        # 응답을 JSON 딕셔너리로 변환하고 get 메서드로 item 값 리스트로 가져오기
        return r.json().get("items", [])
    except Exception as e:
        print(f"[ERROR] API 요청 실패: {e}")
        return []


# 이미지 다운로드
def download_image_from_url(url: str):
    """
    URL -> PIL.Image 로 다운로드.
    실패 시 None 반환.
    """
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        # url은 그 자체로는 이미지가 아니기에 바이너리 데이터로 다운로드
        img = Image.open(BytesIO(r.content)).convert("RGB")
        return img
    except Exception as e:
        print(f"[WARN] 이미지 다운로드 실패: {e}")
        return None


# 메인 로직: 자동 분류
def auto_classify():
    print("\n[STEP] CLIP 모델 로드")
    model, processor, device = load_clip()

    # 클러스터 centroid 로드
    centroids_path = Config.OUTPUT_DIR / "clusters" / "centroids.npy"
    centroids = load_centroids(centroids_path)

    # CSV 초기화
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_name", "model_id", "cluster_id", "image_url", "filename"])

    print("[INFO] 자동 LoRA 분류 시작")

    while True:
        # 최신 LoRA 모델 목록 가져오기
        models = fetch_lora_models()
        if not models:
            print("[INFO] 모델을 가져오지 못함. 재시도")
            time.sleep(REQUEST_INTERVAL)
            # continue로 이번 반복의 남은 코드를 건너뛰고 다시 반복문의 처음으로 이동
            continue

        for m in models:
            model_id = m.get("id")
            model_name = m.get("name", "unknown_model")

            # 대표 이미지 추출
            try:
                img_url = m["modelVersions"][0]["images"][0]["url"]
            except (KeyError, IndexError):
                continue


            # 이미지 다운로드
            img = download_image_from_url(img_url)
            if img is None:
                continue


            # CLIP 임베딩과 클러스터 예측
            emb = embed_image(img, model, processor, device)
            cluster_id = predict_cluster(emb, centroids)


            # 결과 저장
            filename = f"{model_id}.jpg"
            img.save(SAVE_DIR / filename)

            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([model_name, model_id, cluster_id, img_url, filename])

            print(f"[OK] {model_name} -> 클러스터 {cluster_id}")

        print(f"[WAIT] {REQUEST_INTERVAL}초 대기 후 재시작...")
        time.sleep(REQUEST_INTERVAL)


# 실행 진입점
if __name__ == "__main__":
    auto_classify()
