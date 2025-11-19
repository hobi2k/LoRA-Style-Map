# scripts/auto_classify_civitai.py
"""
CivitAI LoRA 실시간 자동 수집 및 클러스터 분류

목표
1. CivitAI REST API에서 최신 LoRA 모델 목록을 주기적으로 가져온다.
2. 각 LoRA의 대표 이미지를 다운로드하지 않고 URL에서 바로 가져온다.
3. CLIP 이미지 인코더로 512차원 임베딩을 계산한다.
4. 미리 계산해 둔 cluster_centroids.npy와 코사인 유사도를 비교하여
   "이 LoRA는 몇 번 클러스터 스타일에 가깝다"를 자동으로 분류한다.
5. 결과를 CSV 로그 파일로 남긴다. (나중에 분석/추천 시스템에 활용 가능)

설계
1. 전역 설정 및 상수 정의
- CivitAI API URL, 요청 간 대기 시간, 한 번에 가져올 LoRA 수, 폴링 간격 등

2. 유틸리티 함수
- load_centroids: 클러스터 중심 로드
- load_clip_model: CLIP 모델/프로세서 로드
- fetch_latest_loras: 최신 LoRA 모델 목록 가져오기
- get_preview_image_url: 모델 JSON에서 대표 이미지 URL 추출
- get_clip_embedding: PIL 이미지를 CLIP 임베딩으로 변환
- cosine_similarities: 임베딩과 클러스터 중심 간 코사인 유사도 계산

3. 핵심 로직
- load_processed_ids: 이미 처리한 LoRA ID 집합 로드
- append_log_row: CSV 로그에 한 줄 추가
- classify_and_log_model: 새 LoRA 한 개를 임베딩 -> 클러스터 분류 -> 로그 기록
- poll_once: 한 번 API를 호출하고 아직 처리하지 않은 LoRA들만 분류
- poll_loop: 일정 주기로 poll_once를 반복 호출하여 "준 실시간" 시스템 구성

사용법
1) 먼저 C단계, cluster_embeddings.py 실행 후 compute_centroids.py로
   cluster_centroids.npy가 생성되어 있어야 한다.

2) 이 스크립트를 다음과 같이 실행:
   python -m scripts.auto_classify_civitai

3) 기본 설정은 "한 번만" 최신 LoRA 목록을 가져와 분류한다.
   - 실시간 데몬처럼 돌리고 싶으면, main에서 poll_loop(max_runs=None)로 변경.
"""

import csv
import time
# HTTP로 받은 바이너리 데이터를 파일처럼 다루기 위한 버퍼
from io import BytesIO
from pathlib import Path

import requests
import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

# 기본 설정 및 경로 불러오기
from config import Config

# 전역 상수 및 경로 설정
# CivitAI 모델 목록 API 엔드포인트
CIVITAI_MODELS_URL = "https://civitai.com/api/v1/models"

# 한 번에 가져올 LoRA 모델 수 (CivitAI 최대 100)
LIMIT_PER_REQUEST = 50

# 정렬 기준: "Newest"로 두면 항상 최신 순으로 온다.
SORT = "Newest"

# API 요청 간 대기 시간(초) - rate limit 회피용 (필요 시 조절)
REQUEST_INTERVAL = 2.0

# 폴링 간격(초) - poll_loop에서 다음 라운드까지 대기 시간
POLL_INTERVAL = 60.0

# 로그 CSV 경로: 새로운 LoRA 자동 분류 결과가 누적될 파일
LOG_CSV_PATH = Config.OUTPUT_DIR / "civitai_auto_classified.csv"

# 클러스터 중심점 벡터(n_clusters, 512)가 저장된 파일
CENTROID_PATH = Config.OUTPUT_DIR / "clusters" / "cluster_centroids.npy"


# 전역 상태: CLIP, centroids, processed IDs
# 캐시란 한 번 계산하거나 가져온 값을 빠르게 다시 쓰기 위해 저장해두는 임시 저장소.
# CLIP 모델/프로세서는 한 번만 로드하고 재사용한다.
_clip_model: CLIPModel | None
_clip_processor: CLIPProcessor | None

# 클러스터 중심점 (n_clusters, 512)
_centroids: np.ndarray | None

# 코사인 유사도 계산용으로 미리 계산해 둔 중심점 벡터들의 L2 노름
_centroid_norms: np.ndarray | None  # 각 centroid 벡터의 L2 노름

# 이미 처리한 LoRA ID 집합 (중복 처리 방지)
_processed_ids: set[int] = set()


# 초기화 관련 함수
def load_centroids():
    """
    cluster_centroids.npy를 로드하여 전역 변수에 저장한다.

    - _centroids: (n_clusters, 512)
    - _centroid_norms: (n_clusters,)
    """
    # 전역 변수 선언
    global _centroids, _centroid_norms

    if not CENTROID_PATH.exists():
        raise FileNotFoundError(f"클러스터 중심점 파일이 없습니다: {CENTROID_PATH}")

    _centroids = np.load(CENTROID_PATH)  # (k, 512)
    _centroid_norms = np.linalg.norm(_centroids, axis=1)
    print(f"[INIT] 클러스터 중심점 로드 완료: {_centroids.shape}")


def load_clip_model():
    """
    CLIP 비전 모델과 프로세서를 로드하여 전역 변수에 저장한다.
    """
    global _clip_model, _clip_processor

    if _clip_model is not None and _clip_processor is not None:
        # 이미 로드된 경우 다시 로드하지 않는다.
        return

    model_name = "openai/clip-vit-base-patch32"
    device = Config.DEVICE

    print(f"[INIT] CLIP 모델 로드: {model_name} (device={device})")
    _clip_model = CLIPModel.from_pretrained(model_name).to(device)
    _clip_processor = CLIPProcessor.from_pretrained(model_name)
    # 평가 모드로 전환
    _clip_model.eval()
    print("[INIT] CLIP 모델 로드 완료")


def ensure_log_file():
    """
    로그 CSV 파일이 없으면 헤더를 생성한다.
    """
    if LOG_CSV_PATH.exists():
        return

    LOG_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 로그 파일 헤더 작성 및 newline='' 옵션으로 줄바꿈 문제 방지
    with open(LOG_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_id",
            "model_name",
            "image_url",
            "cluster_id",
            "similarity",
            "timestamp",
        ])
    print(f"[INIT] 로그 파일 생성: {LOG_CSV_PATH}")


def load_processed_ids_from_log():
    """
    기존 로그 CSV에서 이미 분류한 model_id를 읽어와
    _processed_ids 집합에 저장한다.
    """
    global _processed_ids

    if not LOG_CSV_PATH.exists():
        _processed_ids = set()
        print("[INIT] 기존 로그 파일이 없어 processed_ids를 빈 집합으로 초기화합니다.")
        return

    ids: set[int] = set()
    with open(LOG_CSV_PATH, "r", newline="", encoding="utf-8") as f:
        # CSV 파일의 첫 줄을 헤더로 보고 각 줄을 딕셔너리로 읽음
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mid = int(row["model_id"])
                ids.add(mid)
            except Exception:
                continue

    _processed_ids = ids
    print(f"[INIT] 기존에 처리된 모델 ID {len(_processed_ids)}개 로드 완료")

# CivitAI API 호출 관련 함수
def fetch_latest_loras(limit: int = LIMIT_PER_REQUEST) -> list[dict]:
    """
    CivitAI API에서 최신 LoRA 모델 목록을 가져온다.

    인자:
        limit (int): 한 번에 가져올 모델 수 (최대 100)

    반환값:
        list[dict]: CivitAI 모델 JSON의 "items" 리스트
    """
    params = {
        "types": "LORA",
        "sort": SORT,
        "limit": limit,
        # 페이지네이션은 여기서는 사용하지 않고, 대신 폴링을 여러 번 돌려서 전체적으로 커버한다.
    }

    try:
        resp = requests.get(CIVITAI_MODELS_URL, params=params, timeout=10)
        if resp.status_code == 429:
            # 요청 제한에 걸린 경우: 잠시 대기 후 빈 리스트 반환
            print("[WARN] 429 Too Many Requests: 잠시 대기 후 재시도 권장")
            time.sleep(30)
            return []

        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        print(f"[INFO] 최신 LoRA {len(items)}개 수신 (sort={SORT}, limit={limit})")
        return items

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] CivitAI API 요청 실패: {e}")
        return []

# 대표 이미지 URL 추출 함수
def extract_preview_image_url(model_json: dict) -> str | None:
    """
    CivitAI 모델 JSON에서 대표 이미지 URL을 추출한다.

    반환값:
        str 또는 None: 이미지 URL (없으면 None)
    """
    try:
        url = model_json["modelVersions"][0]["images"][0]["url"]
        return url
    except (KeyError, IndexError, TypeError):
        return None


# CLIP 임베딩 및 유사도 계산 함수: extract_preview_image_url에서 이어짐
def get_clip_embedding_from_url(image_url: str) -> np.ndarray | None:
    """
    이미지 URL을 다운로드하여 PIL 이미지로 변환한 뒤,
    CLIP 이미지 임베딩(512차원)을 계산한다.

    반환값:
        np.ndarray shape (512,) or None (실패 시)
    """
    global _clip_model, _clip_processor

    if _clip_model is None or _clip_processor is None:
        load_clip_model()

    try:
        # url은 그 자체로는 이미지가 아니기에 바이너리 데이터로 다운로드
        r = requests.get(image_url, timeout=15)
        r.raise_for_status()
        image = Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as e:
        print(f"[WARN] 이미지 다운로드/로딩 실패: {image_url} ({e})")
        return None

    device = Config.DEVICE

    # CLIPProcessor가 리사이즈/크롭/정규화를 자동으로 처리한다.
    inputs = _clip_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        # clip processing 후의 inputs 딕셔너리를 언패킹하여 전달
        feats = _clip_model.get_image_features(**inputs)
        # L2 정규화: 임베딩 벡터의 길이를 1로 맞춤
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)

    # shape: (1, 512) -> (512,)
    return feats.cpu().numpy().flatten()

# 코사인 유사도 계산 함수
def cosine_similarities(vec: np.ndarray) -> np.ndarray:
    """
    주어진 임베딩 vec(512,)과 모든 클러스터 중심점(_centroids) 간의
    코사인 유사도 벡터를 반환한다.

    반환값:
        np.ndarray: (n_clusters,)
    """
    global _centroids, _centroid_norms

    if _centroids is None or _centroid_norms is None:
        load_centroids()

    # 내적 / (각 노름의 곱)
    # dot: (k, 512) · (512,) -> (k,)
    dots = _centroids @ vec            # 각 중심점과의 내적 계산 (분자)
    vec_norm = np.linalg.norm(vec)      # 입력 벡터의 L2 노름 계산 (분모 일부)

    if vec_norm == 0:
        # 이론상 거의 발생하지 않지만, 안전하게 처리
        return np.zeros_like(_centroid_norms)

    sims = dots / (_centroid_norms * vec_norm)
    return sims


# 로그 및 단일 모델 분류 함수
def append_log_row(model_id: int, model_name: str, image_url: str, cluster_id: int, similarity: float):
    """
    분류 결과를 로그 CSV 파일에 한 줄 추가한다.
    """
    # string format time으로 시간 객체를 문자열로 변환
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    with open(LOG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            model_id,
            model_name,
            image_url,
            cluster_id,
            f"{similarity:.6f}",
            timestamp,
        ])

# 단일 모델 분류 함수
def classify_and_log_model(model_json: dict):
    """
    단일 LoRA 모델 JSON에 대해:
    1. model_id / name / preview image URL 추출
    2. CLIP 임베딩 계산
    3. 클러스터 중심점과 코사인 유사도 계산
    4. 가장 유사한 cluster_id 선택
    5. 로그 CSV에 기록
    """
    global _processed_ids

    model_id = model_json.get("id", None)
    if model_id is None:
        return

    # int로 캐스팅 가능한지 확인
    try:
        model_id = int(model_id)
    except Exception:
        return

    # 이미 처리한 모델이면 스킵
    if model_id in _processed_ids:
        return

    model_name = model_json.get("name", "unknown_model")
    img_url = extract_preview_image_url(model_json)
    if img_url is None:
        print(f"[SKIP] 이미지가 없는 모델: id={model_id}, name={model_name}")
        _processed_ids.add(model_id)
        return

    emb = get_clip_embedding_from_url(img_url)
    if emb is None:
        print(f"[SKIP] 임베딩 계산 실패: id={model_id}, url={img_url}")
        _processed_ids.add(model_id)
        return

    sims = cosine_similarities(emb)
    # 코사인 유사도가 가장 높은 클러스터 선택
    cluster_id = int(np.argmax(sims))
    # 코사인 유사도 값
    sim_value = float(sims[cluster_id])

    append_log_row(model_id, model_name, img_url, cluster_id, sim_value)
    _processed_ids.add(model_id)

    print(f"[OK] 모델 {model_id} ({model_name}) -> 클러스터 {cluster_id} (sim={sim_value:.3f})")


# 폴링 단위 동작과 전체 루프
def poll_once():
    """
    최신 LoRA 목록을 한 번 가져와서,
    아직 처리하지 않은 모델만 자동 분류하고 로그에 기록한다.
    """
    models = fetch_latest_loras()
    if not models:
        print("[INFO] 가져올 모델이 없거나 요청 실패")
        return

    new_count = 0
    for m in models:
        before = len(_processed_ids)
        classify_and_log_model(m)
        after = len(_processed_ids)
        if after > before:
            new_count += 1
            # rate limit 완화용 대기
            time.sleep(REQUEST_INTERVAL)

    print(f"[INFO] 이번 라운드에서 새로 분류된 모델 수: {new_count}")


def poll_loop(max_runs: int):
    """
    CivitAI를 주기적으로 폴링하는 루프.

    인자:
        max_runs:
            - 정수인 경우: 해당 횟수만큼 poll_once 실행 후 종료
            - None인 경우: 무한 루프 (Ctrl+C로 종료)

    예:
        poll_loop(max_runs=1)   -> 테스트용 단발 실행
        poll_loop(max_runs=None) -> 데몬처럼 계속 실행
    """
    run = 0
    while True:
        run += 1
        print(f"\n[LOOP] 폴링 라운드 {run} 시작")
        poll_once()
        if max_runs is not None and run >= max_runs:
            print("[LOOP] 지정된 횟수만큼 실행 완료. 종료합니다.")
            break

        print(f"[LOOP] 다음 라운드까지 {POLL_INTERVAL}초 대기")
        time.sleep(POLL_INTERVAL)


# 실행 진입점
if __name__ == "__main__":
    # 초기화: 로그 파일, centroids, processed_ids, CLIP
    ensure_log_file()
    load_centroids()
    load_clip_model()
    load_processed_ids_from_log()

    # 우선은 테스트 겸 한 번만 실행
    # 실서비스처럼 돌릴 때는 max_runs=None 으로 변경 가능
    poll_loop(max_runs=1)
