# src/classify_new_lora.py
"""
단일 LoRA 자동 분류 스크립트

입력: LoRA 모델 ID (예: 123456)
출력: 클러스터 ID, 로컬 이미지 저장, CSV 기록

사용 사례:
- Streamlit에서 사용자가 LoRA ID를 입력하면 자동 스타일 분류
- auto_classify_civitai.py는 배치/실시간용
- 본 파일은 단일 처리용 엔드포인트
"""
import csv
import requests
from io import BytesIO
from pathlib import Path
from PIL import Image

from src.config import Config
from src.classifier_core import (
    load_clip,
    embed_image,
    load_centroids,
    predict_cluster
)

# 저장 경로
SAVE_DIR = Config.OUTPUT_DIR / "single_classify"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = SAVE_DIR / "results.csv"


# CivitAI에서 LoRA 단일 모델 정보 가져오기
def fetch_lora_info(lora_id: int):
    """
    특정 LoRA 모델의 상세정보를 가져옵니다.
    """
    url = f"https://civitai.com/api/v1/models/{lora_id}"

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ERROR] LoRA 정보 요청 실패: {e}")
        return None


# 이미지 다운로드
def download_image(url: str):
    """
    이미지 URL을 PIL.Image로 다운로드합니다.
    """
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as e:
        print(f"[WARN] 이미지 다운로드 실패: {e}")
        return None


# 메인 로직
def classify_lora(lora_id: int):
    """
    LoRA ID를 입력받아 자동 분류 실행.
    """
    print(f"\n[INFO] LoRA ID {lora_id} 분류 시작")

    # LoRA 정보 가져오기
    info = fetch_lora_info(lora_id)
    if info is None:
        return None

    # 모델 이름 지정
    model_name = info.get("name", f"id_{lora_id}")

    # 이미지 URL 찾기
    try:
        url = info["modelVersions"][0]["images"][0]["url"]
    except (KeyError, IndexError):
        print("[ERROR] 이미지 URL을 찾을 수 없습니다.")
        return None

    # 이미지 다운로드
    img = download_image(url)
    if img is None:
        return None

    # CLIP 불러오기
    model, processor, device = load_clip()

    # 임베딩 후 cluster 예측
    emb = embed_image(img, model, processor, device)
    centroids = load_centroids(Config.OUTPUT_DIR / "clusters" / "centroids.npy")
    cluster_id = predict_cluster(emb, centroids)

    # 이미지 저장
    filename = f"{lora_id}.jpg"
    img.save(SAVE_DIR / filename)

    # CSV 저장
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([lora_id, model_name, cluster_id, url, filename])

    print(f"[DONE] LoRA '{model_name}' -> 클러스터 {cluster_id}")
    return cluster_id


# 실행 진입점
if __name__ == "__main__":
    test_id = int(input("분류할 LoRA ID를 입력하세요: "))
    classify_lora(test_id)
