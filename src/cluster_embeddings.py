# src/cluster_embeddings.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap
from pathlib import Path
from src.config import Config

"""
CLIP 임베딩 기반 LoRA 스타일 군집화 스크립트

기능:
- CLIP 임베딩(512차원)을 차원 축소 (PCA -> UMAP)
- KMeans로 유사한 이미지끼리 자동 분류
- 2D 시각화 결과를 scatter plot으로 저장

출력:
- cluster_labels.csv
- umap_clusters.png
"""

# 데이터 로드
EMB_PATH = Config.OUTPUT_DIR / "embeddings" / "clip_features.npy"
NAME_PATH = Config.OUTPUT_DIR / "embeddings" / "clip_filenames.npy"
OUTPUT_DIR = Config.OUTPUT_DIR / "clusters"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 8  # find_optimal_k.py로 군집 개수 결정 후 설정

features = np.load(EMB_PATH)
filenames = np.load(NAME_PATH)

print(f"[INFO] 불러온 임베딩: {features.shape}, 파일 이름: {len(filenames)}")

# 차원 축소 작업
print("[STEP] PCA 진행 (512 -> 50)")
pca = PCA(n_components=50, random_state=42)
pca_features = pca.fit_transform(features)

print("[STEP] UMAP 진행 (50 -> 2)")
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_features = umap_model.fit_transform(pca_features)

# 군집화 (K-Means) 
print(f"[STEP] k-mean 실행 (k={N_CLUSTERS})")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
labels = kmeans.fit_predict(umap_features)

# 시각화
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    umap_features[:, 0],
    umap_features[:, 1],
    c=labels,
    cmap="tab10",
    s=10,
    alpha=0.8
)
plt.title("CLIP Embeddings Clustering (UMAP + K-Means)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.colorbar(scatter, label="Cluster ID")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "umap_clusters.png", dpi=200)
plt.close()

# CSV 저장
cluster_df = pd.DataFrame({
    "filename": filenames,
    "cluster_id": labels
})
cluster_df.to_csv(OUTPUT_DIR / "cluster_labels.csv", index=False, encoding="utf-8")

print(f"[DONE] 군집화 완료: {N_CLUSTERS}개 군집")
print(f"[INFO] 시각화 저장: {OUTPUT_DIR / 'umap_clusters.png'}")
print(f"[INFO] CSV 저장: {OUTPUT_DIR / 'cluster_labels.csv'}")
