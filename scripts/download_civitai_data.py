"""
CivitAI LoRA 이미지 누적 수집 스크립트 (개선판)

목적
- 공식 REST API의 페이지네이션 한계를 감안하여, 같은 요청을 반복 실행하며
  새로운 이미지를 누적 저장합니다. (페이지네이션을 시도했으나 작동을 안 함)
- 실행을 중간에 멈추었다가 다시 시작해도 이어받기가 가능하도록 합니다.
- 'LoRA 분류기' 목적에 맞게 types=LORA를 유지합니다.

참고 사항
- CivitAI API 문서 https://github.com/civitai/civitai/wiki/REST-API-Reference
"""

import os
import csv
import time
import random
import requests
from pathlib import Path
from src.config import Config


# 전역 상수 설정

# API 요청 간 최소/최대 대기 (랜덤 슬립으로 rate-limit/캐시 완화)
REQUEST_INTERVAL_RANGE = (1.0, 2.5)

# 반복 횟수: 한 번의 실행에서 몇 라운드 돌릴지
REPEAT_RUNS = 10

# 요청당 최대 모델 수 (CivitAI 상한 100)
LIMIT_PER_REQUEST = 100

# 정렬 기준: 400 에러가 적은 값들 위주로 사용
# 'Highest Rated'는 400이 나는 경우가 있어 제외
SORT_OPTIONS = ["Newest", "Most Downloaded", "Most Buzzed"]

# 검색 키워드(= 스타일 키워드): Config.STYLE_CLASSES 사용
# getattr(객체, 속성이름, 기본값)으로 안전하게 접근, 기본값 설정으로 속성이 없을 때 기본값 반환
STYLE_KEYWORDS = list(getattr(Config, "STYLE_CLASSES", ["anime", "realistic", "3d", "sketch", "cinematic"]))

# CSV 저장 경로 / 출력 폴더
CSV_PATH = Config.LABEL_CSV
SAVE_DIR = Config.RAW_DIR / "all"

# User-Agent 풀: 캐시/봇차단 무력화에 도움
# 요청 헤더(header)에 들어가는 User-Agent 값은 웹 서버가 요청의 출처를 식별하는 정보로, 이 요청을 보낸 클라이언트가 어떤 프로그램인지 알려주는 문자열
# 다양한 User-Agent를 사용하여 랜덤으로 요청 시도
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:118.0) Gecko/20100101 Firefox/118.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_2) AppleWebKit/605.1.15 (KHTML, like Gecko)",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
]


# 유틸리티 함수
def sleep_interval():
    """
    요청 간 랜덤 대기
    
    REQUEST_INTERVAL_RANGE 값을 사용하여 랜덤 범위로 대기 시간을 설정
    time.sleep() 함수로 실행 정지 수행
    """
    time.sleep(random.uniform(*REQUEST_INTERVAL_RANGE))


def backoff_sleep(attempt: int):
    """
    지수 백오프 대기.
    - API 호출 시 오류가 났을 때 재시도하기 전에 점진적으로 길게 기다리게 하는 함수
    - attempt: 1,2,3... 시도 횟수
    인자:
        attempt (int): 재시도 시도 횟수
    반환값: 
        대기
    기능:
        2^(attempt-1) 초 만큼 대기 (최대 30초)
    """
    wait = min(30.0, 2.0 * (2 ** (attempt - 1)))  # 최대 30초
    time.sleep(wait)


def init_resume_state():
    """
    이어받기(resume)를 위한 초기 상태 구성.
    - 기존 CSV가 있으면 seen_urls를 CSV에서 복원
    - 기존 SAVE_DIR에 저장된 파일 수로 collected 시작 위치 산정
    반환:
        seen_urls (set[str]), collected (int), csv_exists (bool)
        
    기능:
        기존 CSV에서 image_url 컬럼을 읽어와 seen_urls 집합 생성
        SAVE_DIR 폴더 내 이미지 파일 수를 세어 collected 값 산정
    """
    # 저장 폴더 생성(없으면 상위 폴더까지 포함해 생성)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    seen_urls = set()
    collected = 0
    # csv 파일 존재 여부 확인(bool값 반환)
    csv_exists = Path(CSV_PATH).exists()

    # CSV에서 기존 URL 복원
    if csv_exists:
        try:
            with open(CSV_PATH, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                # 첫 줄을 헤더로 읽어서 변수에 저장
                header = next(reader, None)
                # image_url 컬럼 인덱스 탐색 (없으면 2번 컬럼이라고 가정)
                url_idx = header.index("image_url") if header and "image_url" in header else 2
                for row in reader:
                    if len(row) > url_idx:
                        seen_urls.add(row[url_idx])
        except Exception:
            # 문제가 있어도 이어서 진행
            pass

    # 폴더 내 기존 파일 수로 collected 산정
    if SAVE_DIR.exists():
        # iterdir()로 폴더 내 모든 항목 순회하고 is_file()로 파일만 필터링 -> suffix로 이미지 파일 확장자 검사
        existing = [p for p in SAVE_DIR.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".png", ".jpeg")]
        collected = len(existing)

    return seen_urls, collected, csv_exists


def open_csv_for_append(csv_exists: bool):
    """
    CSV 파일을 append 모드로 열고, 없으면 헤더를 씁니다.
    반환: csv.writer
    """
    f = open(CSV_PATH, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if not csv_exists:
        writer.writerow(["filename", "lora_name", "image_url", "keyword", "sort", "run_index"])
    # f가 열린 파일 객체, writer가 csv.writer 객체
    # 파일 객체도 반환해야 나중에 닫을 수 있음
    return f, writer


def fetch_lora_models(keyword: str, sort: str, limit: int = LIMIT_PER_REQUEST) -> list[dict]:
    """
    CivitAI API에서 LoRA 모델 목록을 가져옵니다.

    인자:
        keyword: 검색 키워드 (예: "anime")
        sort: 정렬 기준 (예: "Newest")
        limit: 요청당 모델 수 (100)
    반환값:
        list[dict]: 모델 정보 리스트
    """
    base_url = "https://civitai.com/api/v1/models"
    params = {
        "query": keyword,     # 인자로 받은 keyword를 그대로 사용
        "sort": sort,
        "limit": limit,
        "types": "LORA",      # LoRA 분류기 목적에 맞게 LORA만 유지
    }

    # 헤더: User-Agent 랜덤으로 교체하여 요청 (봇차단/캐시 우회 목적)
    headers = {"User-Agent": random.choice(USER_AGENTS)}

    # 429/일시적 오류 대비 재시도
    for attempt in range(1, 5):  # 최대 4회 시도
        try:
            resp = requests.get(base_url, params=params, headers=headers, timeout=12)
            if resp.status_code == 429:
                print(f"[WARN] 429 Too Many Requests — 백오프 대기 후 재시도 (attempt={attempt})")
                # 지수 백오프 대기
                backoff_sleep(attempt)
                continue
            # HTTPError 발생 시 except로 이동.
            resp.raise_for_status()
            # 응답 JSON 파싱 후 최상위 키 items 가져옴. 없으면 기본값 [].
            data = resp.json().get("items", [])
            print(f"[INFO] keyword='{keyword}' sort='{sort}' → {len(data)}개")
            return data
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] 요청 실패: keyword='{keyword}' sort='{sort}' attempt={attempt}: {e}")
            backoff_sleep(attempt)

    # 모든 재시도 실패
    return []


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
        # 저장 경로의 상위 폴더가 없으면 생성
        # 헷갈리지 않도록 부가 설명하자면, 파일이 들어 있는 폴더의 경로만 가져와서 생성
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # 이미지 파일을 바이너리 모드(wb)로 저장
        with open(save_path, "wb") as f:
            # 응답 데이터를 디코딩하지 않고 그대로 기록(이미지 저장)
            f.write(response.content)
        return True

    except Exception as e:
        print(f"[WARN] {image_url} 다운로드 실패: {e}")
        return False

# 메인 파이프라인
def collect_images():
    """
    LoRA 이미지를 여러 번 반복 호출하며 누적 다운로드합니다.
    - 이어받기 지원: 기존 CSV/폴더를 참조하여 중복 방지 및 파일명 번호 이어쓰기
    - 각 라운드(run)마다 keyword × sort 조합을 순회
    """
    # 이어받기 상태 구성
    seen_urls, collected, csv_exists = init_resume_state()
    print(f"[RESUME] 기존 수집 이미지: {collected}장, 기존 URL: {len(seen_urls)}개, CSV 존재: {csv_exists}")

    # CSV append 모드로 열기
    csv_file, writer = open_csv_for_append(csv_exists)

    try:
        for run in range(REPEAT_RUNS):
            print(f"\n[RUN {run+1}/{REPEAT_RUNS}] 수집 시작")

            # 키워드 순서 랜덤 섞기: 매 실행마다 노출 다양화
            random.shuffle(STYLE_KEYWORDS)

            for keyword in STYLE_KEYWORDS:
                for sort in SORT_OPTIONS:
                    models = fetch_lora_models(keyword, sort)
                    if not models:
                        sleep_interval()
                        continue

                    for model in models:
                        # 이미지 URL/이름 추출
                        try:
                            image_url = model["modelVersions"][0]["images"][0]["url"]
                            name = model.get("name", "unknown_model")
                        except (KeyError, IndexError):
                            continue

                        # 중복 URL 필터
                        if image_url in seen_urls:
                            continue
                        seen_urls.add(image_url)

                        # 파일명 생성 (이어쓰기)
                        collected += 1
                        filename = f"lora_{collected:04d}.jpg"
                        save_path = SAVE_DIR / filename

                        # 다운로드 및 CSV 기록
                        if download_image(image_url, save_path):
                            writer.writerow([filename, name, image_url, keyword, sort, run])
                            csv_file.flush()  # 비정상 종료 대비하여 buffer에 쌓인 데이터 즉시 기록
                            print(f"[OK] {filename} 저장 (총 {collected}장)")
                        else:
                            # continue 대신 pass를 사용해서 흐름에 영향 안 주도록 함
                            pass

                        # 요청 간 간격
                        sleep_interval()

            print(f"[INFO] RUN {run+1} 완료 — 누적 {collected}장")
            # 라운드 사이 간격
            time.sleep(random.uniform(4.0, 7.0))

    finally:
        csv_file.close()

    print(f"\n[COMPLETE] 총 {collected}장 이미지가 {SAVE_DIR}에 저장됨.")
    print(f"[INFO] CSV 저장 위치: {CSV_PATH}")


# 실행 진입점
if __name__ == "__main__":
    collect_images()
