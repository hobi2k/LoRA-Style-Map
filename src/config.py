# src/config.py
import torch
# 파일 경로를 문자열이 아닌 객체(Path)로 다루기 위한 표준 라이브러리
from pathlib import Path

"""
프로젝트 전역 설정을 모아두는 클래스.

- 경로: 데이터/라벨/출력/모델 가중치의 표준 위치
- 하이퍼파라미터: 이미지 크기, 배치 크기, 학습률 등
- 디바이스: GPU(cuda) 또는 CPU 자동 선택
"""

class Config:
    # 경로 설정
    # 루트 폴더를 LoRA_Style_Classifier로 설정
    # path(__file__)는 현재 이 파일(config.py)의 경로 문자열
    # __file__.resolve().parents[1]는 현재 파일(src/config.py)의 부모 폴더의 부모 폴더를 의미
    ROOT = Path(__file__).resolve().parents[1] 
    # 출력 결과물 저장 폴더(모델, 로그, 예측 결과 등)
    OUTPUT_DIR = ROOT / "outputs"
    # 전처리 완료본
    DATA_DIR = ROOT / "data" / "processed"
    # 원본 데이터
    RAW_DIR = ROOT / "data" / "raw"
    # 라벨 정보가 담긴 CSV 파일
    LABEL_CSV = ROOT / "data" / "labels.csv"

    # 하이퍼파라미터
    # 이미지 크기를 256x256으로 리사이즈 설정용
    IMG_SIZE = 256
    # 미니배치 크기
    BATCH_SIZE = 32
    # 학습 에폭 수
    EPOCHS = 10
    # 학습률
    LR = 0.0001
    
    # 분류 클래스 정의
    # 스타일 카테고리를 세분화하여 확장 (추후 실시간 분류 구현 예비용)
    # anime, realistic, 3d, sketch, cinematic 외에 주요 LoRA 스타일을 추가
    STYLE_CLASSES = [
    "anime",              # 2D 일본풍 스타일
    "western-cartoon",    # 디즈니/픽사풍 만화
    "realistic",          # 실사형 렌더 (사람, 풍경 등)
    "photographic",       # 실제 사진풍
    "3d-render",          # 3D 모델 기반 렌더
    "lowpoly",            # 단순 폴리곤 렌더 (게임풍)
    "pixel-art",          # 도트 그래픽
    "sketch",             # 선화/연필 드로잉
    "line-art",           # 윤곽선 중심의 흑백선 스타일
    "oil-painting",       # 유화풍 텍스처
    "watercolor",         # 수채화풍 질감
    "comic-style",        # 만화책(잉크, 스크린톤) 풍
    "cyberpunk",          # SF 네온풍 스타일
    "fantasy-art",        # 판타지 일러스트 (드래곤, 마법 등)
    "concept-art"         # 게임/영화용 디자인 콘셉트화
]
    # 클래스 개수 자동 계산
    NUM_CLASSES = len(STYLE_CLASSES)

    # 디바이스 변수
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    