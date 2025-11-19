# src/config.py
import torch
# 파일 경로를 문자열이 아닌 객체(Path)로 다루기 위한 표준 라이브러리
from pathlib import Path

"""
프로젝트 전역 설정을 모아두는 클래스.

- 경로: 데이터/라벨/출력의 표준 위치
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

    # 디바이스 변수
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    