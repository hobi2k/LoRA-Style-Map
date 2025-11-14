# src/dataset_clip.py
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd

class LoRAImageDataset(Dataset):
    """
    CLIP 임베딩 생성을 위한 이미지 데이터셋 클래스

    기능:
    - labels.csv에서 파일명 목록을 읽고
    - 실제 이미지 파일을 로드한 후,
    - CLIP Processor가 사용할 수 있도록 반환
    """

    def __init__(self, csv_path: Path, image_dir: Path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        row = self.data.iloc[idx]
        img_path = self.image_dir / row["filename"]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # 파일명도 함께 반환
        return image, row["filename"]
