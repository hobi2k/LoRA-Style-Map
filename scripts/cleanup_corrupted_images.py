"""
손상된 LoRA 이미지 정리 스크립트


CivitAI에서 수집한 이미지 중 일부는
- 다운로드 도중 끊김
- JPG 헤더 손상
- 잘못된 확장자 (PNG인데 JPG로 저장됨)
등으로 인해 열리지 손상될 수 있습니다.

이 스크립트는 PIL로 모든 이미지를 열어보고,
열 수 없는 파일을 삭제하며 labels.csv에서도 제거합니다.
"""

import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import pandas as pd
from src.config import Config

# 경로 설정
IMAGE_DIR = Config.RAW_DIR / "all"
CSV_PATH = Config.LABEL_CSV
# 추후 logs 폴더로 이동 예정
LOG_PATH = Path(Config.OUTPUT_DIR / "broken_images.txt")

def cleanup_corrupted_images():
    """
    손상된 이미지 파일을 탐지하고 삭제한 뒤,
    labels.csv에서도 해당 행을 제거합니다.
    """
    df = pd.read_csv(CSV_PATH)
    broken_files = []

    # iterrows()를 사용하여 csv의 각 이미지 파일 행 검사
    for idx, row in df.iterrows():
        img_path = IMAGE_DIR / row["filename"]
        try:
            with Image.open(img_path) as img:
                # Pillow의 verify() 메서드로 이미지 무결성 검사
                img.verify()
        except (UnidentifiedImageError, OSError, FileNotFoundError):
            broken_files.append(row["filename"])
            # 안전하게 삭제 시도
            if img_path.exists():
                try:
                    # Path.unlink()로 파일 삭제: Path 객체의 내장 메서드
                    img_path.unlink()
                except Exception as e:
                    print(f"[WARN] {img_path} 삭제 실패: {e}")

    if not broken_files:
        print("[OK] 손상된 이미지가 없습니다.")
        return

    # CSV에서 제거
    df = df[~df["filename"].isin(broken_files)]
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    # 로그 저장
    # parents=True: 상위 폴더가 없으면 생성
    # exist_ok=True: 이미 폴더가 있어도 에러 발생 안함
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(broken_files))

    print(f"[CLEANUP COMPLETE] 손상된 이미지 {len(broken_files)}개 삭제")
    print(f"로그 파일: {LOG_PATH}")

if __name__ == "__main__":
    cleanup_corrupted_images()
