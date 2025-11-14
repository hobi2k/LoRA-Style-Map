"""
LoRA 전체 이미지 데이터 수집 스크립트 (스타일 구분 없음)

기능
1. CivitAI의 공개 API를 통해 전체 LoRA 모델 목록을 요청합니다.
2. 각 모델의 대표 이미지를 다운로드합니다.
3. 다운로드된 이미지를 'data/raw/all/' 폴더에 저장합니다.
4. 동시에 각 이미지의 메타정보(filename, lora_name, image_url)을
   'data/labels.csv' 파일로 기록합니다.

목표
- 스타일 태그에 구애받지 않고 가능한 많은 LoRA 이미지 확보
- 이후 단계에서 CLIP 임베딩 + 군집화를 통해 자동 스타일 분류 수행

참고 사항
- CivitAI API 문서 https://github.com/civitai/civitai/wiki/REST-API-Reference

주의 사항
- API 요청 횟수가 많을 경우 rate limit(요청 제한)에 걸릴 수 있으므로,
  `REQUEST_INTERVAL` 값을 적절히 조정합니다.

설계
1. import -> 도구 준비
2. 상수 정의 -> 기본 파라미터 설정
3. 함수 정의 -> 코드 블록 구성 (fetch, download, collect)
4. if __name__ == "__main__" -> 실행 진입점
"""

# 외부 라이브러리 및 모듈 불러오기
import os
import csv
import time
import requests
from pathlib import Path
from src.config import Config

# 상수 정의 — 프로그램 전역 설정값

# 요청 간 대기 시간 (초)
# 서버 과부하 방지를 위해 적절히 설정 필요
REQUEST_INTERVAL = 3.0

# 한 번의 요청당 가져올 모델 수 (API 상한: 100)
LIMIT_PER_REQUEST = 50

# 전체 다운로드 목표 이미지 수 (예: 300장)
TOTAL_IMAGES = 300

# CSV 메타데이터 파일 경로
CSV_PATH = Config.LABEL_CSV

def fetch_lora_models(limit=LIMIT_PER_REQUEST) -> list:
    """
    CivitAI API에서 LoRA 모델 목록을 요청합니다.
    query 파라미터 없이 전체 LoRA 중 일부를 정렬 기준에 따라 가져옵니다.

    인자:
        limit: 한 번의 요청당 가져올 모델 수 (기본값 50)

    반환값:
        list: 모델 정보 딕셔너리들의 리스트
    """
    # CivitAI API 고정 엔드포인트
    base_url = "https://civitai.com/api/v1/models"
    
    # 정렬 기준을 다양화하여 데이터 다양성 확보
    sort_options = ["Newest", "Most Downloaded", "Highest Rated"]

    all_models = []
    for sort in sort_options:
        params = {
            "limit": limit,
            "types": "LORA",
            "sort": sort
        }

        try:
            # 고정 엔드포인트에 GET 요청. parms로 쿼리 스트링 전달
            response = requests.get(base_url, params=params, timeout=10)
            # HTTPError 발생 시 except로 이동.
            response.raise_for_status()
            #응답 JSON 파싱 후 최상위 키 items 가져옴. 없으면 기본값 [].
            data = response.json().get("items", [])
            print(f"[INFO] {sort} 기준으로 {len(data)}개 모델 수집 완료")
            # data가 리스트이므로 확장하여 추가
            all_models.extend(data)

            # rate-limit 방지를 위한 대기
            time.sleep(REQUEST_INTERVAL)

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] '{sort}' 요청 실패: {e}")
            continue

    # 모델 ID 기준으로 딕셔너리를 만들고 vals()로 값만 추출
    unique_models = {m["id"]: m for m in all_models}.values()
    print(f"[INFO] 총 {len(unique_models)}개 고유 모델 확보")
    return list(unique_models)


def download_image(image_url: str, save_path: Path) -> bool:
    """
    이미지 URL을 로컬 경로로 다운로드합니다.

    인자:
        image_url: 다운로드할 이미지의 URL
        save_path: 저장할 로컬 파일 경로

    반환값:
        bool: 다운로드 성공 여부
    """
    try:
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()

        # 이미지 파일을 바이너리 모드(wb)로 저장
        with open(save_path, "wb") as f:
            # 응답 데이터를 디코딩하지 않고 그대로 기록(이미지 저장)
            f.write(response.content)
        return True

    except Exception as e:
        print(f"[WARN] {image_url} 다운로드 실패: {e}")
        return False

# LoRA 모델 목록을 가져오기 -> 대표 이미지 URL을 뽑기 -> 중복을 피해서 저장하기 -> CSV에 메타데이터를 기록하기 -> 진행 로그 출력
def collect_images():
    """
    LoRA 모델 목록을 기반으로 이미지를 다운로드하고,
    각 파일의 메타데이터를 CSV로 기록합니다.

    동작 순서:
    1. CSV 초기화 및 폴더 생성
    2. 모델 목록 요청
    3. 각 모델의 대표 이미지를 다운로드
    4. 파일 메타정보를 labels.csv에 기록
    """
    # CSV 파일 초기화
    os.makedirs(Config.RAW_DIR, exist_ok=True)
    save_dir = Config.RAW_DIR / "all"  # 모든 이미지를 한 폴더에 저장
    save_dir.mkdir(parents=True, exist_ok=True)

    # CSV 파일 생성 및 헤더 작성
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "lora_name", "image_url"])

    # LoRA 모델 목록 수집
    models = fetch_lora_models()
    if not models:
        print("[ERROR] 모델을 가져오지 못했습니다. 종료합니다.")
        return

    # 이미지 다운로드 루프
    collected = 0
    # 집합 함수로 중복 제거
    seen_urls = set()

    for model in models:
        # 목표 이미지 수 도달 시 종료
        if collected >= TOTAL_IMAGES:
            break

        model_name = model.get("name", "unknown_model")

        # 대표 이미지 URL 추출: civitai API 응답 구조에 따라 조정 필요
        try:
            image_url = model["modelVersions"][0]["images"][0]["url"]
        except (KeyError, IndexError):
            continue

        # 중복된 URL은 스킵
        if image_url in seen_urls:
            continue
        seen_urls.add(image_url)

        filename = f"lora_{collected+1:03d}.jpg"
        save_path = save_dir / filename

        # 이미지 다운로드 수행
        # download_image 함수가 True 반환 시에만 CSV 기록
        if download_image(image_url, save_path):
            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([filename, model_name, image_url])

            collected += 1
            print(f"[OK] {filename} 저장 완료 ({collected}/{TOTAL_IMAGES})")

        # 요청 간 대기 시간 (서버 보호)
        time.sleep(REQUEST_INTERVAL)

    print(f"\n[COMPLETE] 총 {collected}장의 이미지가 {save_dir}에 저장됨.")
    print(f"[INFO] 메타데이터 CSV: {CSV_PATH}")


# 실행 진입점: 스크립트 직접 실행 시 동작
if __name__ == "__main__":
    collect_images()