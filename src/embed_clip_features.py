# src/embed_clip_features.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from src.dataset_clip import LoRAImageDataset
from src.config import Config

"""
CLIP 임베딩 생성 스크립트

목표:
- CLIP 모델의 비주얼 인코더를 통해 이미지 특징 벡터 추출
- 각 LoRA 이미지 → 512차원 벡터로 변환

참고 자료:
- Hugging Face CLIP 모델: https://huggingface.co/openai/clip-vit-base-patch32
- CLIP 모델 자료: https://johnowhitaker.github.io/tglcourse/clip.html

모델 특징:
- 이미지와 텍스트를 동일한 임베딩 공간에 매핑
- CLIPProcessor가 224×224로 맞춤 (CLIP 모델 고정 입력)
- CLIPProcessor가 ImageNet 평균/표준편차로 정규화
- CLIPProcessor가 비율 유지하며 중앙 크롭
- 이미지 임베딩 차원은 512

CLIP 구성요소:
- CLIPProcessor: 이미지 전처리(리사이즈 224, 센터크롭, 정규화) 담당.
- CLIPModel: 실제 추론(전방향 연산) 담당.
"""

# 경로 설정
CSV_PATH = Config.LABEL_CSV
IMAGE_DIR = Config.RAW_DIR / "all"
OUTPUT_DIR = Config.OUTPUT_DIR / "embeddings"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 모델 및 프로세서 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(model_name)

# 데이터셋 및 로더 구성
dataset = LoRAImageDataset(CSV_PATH, IMAGE_DIR)
# collate_fn=lambda x: x로 배치 내 튜플 리스트 형태 유지(pil 이미지를 그대로 전달하기 위함)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

# 임베딩 리스트
all_features, all_filenames = [], []

clip_model.eval()
with torch.no_grad():
    # tqdm으로 진행상황 표시, images: 배치 단위 이미지 리스트, filenames: 해당 이미지 파일명 리스트
    for batch in tqdm(dataloader, desc="Extracting CLIP features"):
        # collate_fn=lambda x: x 로 인해 batch는 [(image, filename)] 형태
        image, filename = batch[0]
        # return_tensors="pt"로 텐서 반환, padding=True로 배치 내 이미지 크기 최대 크기로 맞춤
        inputs = clip_processor(images=image, return_tensors="pt", padding=True).to(device)
        # get_image_features 메서드는 비전 인코더의 출력값 잠재 표현이 공유 임베딩 공간으로 사영된 이미지 임베딩을 반환.
        # **inputs: 딕셔너리 언패킹으로 입력 (CLIPProcessor가 반환한 inputs는 dict 형태)
        outputs = clip_model.get_image_features(**inputs)
        # 임베딩 출력값이 총합 1이 되도록 L2 정규화 
        outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)  # -> (batch_size, 512)
        # 각 이미지 임베딩 CPU로 옮긴 후 넘파이 배열로 변환하여 리스트에 추가
        all_features.append(outputs.cpu().numpy())
        # 파일명이 이미지와 매칭되도록 extend 사용
        all_filenames.append(filename)

# 배열 병합 및 저장
features = np.concatenate(all_features, axis=0) # -> (num_images, 512) 
# np.save를 사용하여 넘파이 배열로 저장: np.save(file, arr)
np.save(OUTPUT_DIR / "clip_features.npy", features)
np.save(OUTPUT_DIR / "clip_filenames.npy", np.array(all_filenames))

print(f"[DONE] {features.shape[0]}개 이미지의 CLIP 임베딩 저장 완료")
print(f"저장 경로: {OUTPUT_DIR}")
