# src/compute_centroids.py
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import Config

"""
K-Means 클러스터 중심점을 계산하는 스크립트

입력:
- clip_features.npy : (N, 512)
- cluster_labels.csv : (N, cluster_id)

출력:
- cluster_centroids.npy : 각 클러스터당 512차원 평균 벡터
"""

EMB_PATH = Config.OUTPUT_DIR / "embeddings" / "clip_features.npy"
LABEL_PATH = Config.OUTPUT_DIR / "clusters" / "cluster_labels.csv"
OUT_PATH = Config.OUTPUT_DIR / "clusters" / "cluster_centroids.npy"

features = np.load(EMB_PATH)
df = pd.read_csv(LABEL_PATH)

labels = df["cluster_id"].values
n_clusters = labels.max() + 1

centroids = []

for cid in range(n_clusters):
    # features와 df가 인덱스 위치를 공유하므로 동일한 인덱스로 접근: labels(pandas serial) == cid인 벡터들 선택
    cluster_vecs = features[labels == cid]
    # 각 클러스터의 중심점 계산 (평균 벡터)
    centroid = cluster_vecs.mean(axis=0)
    centroids.append(centroid)

# np.stack으로 (n_clusters, 512)의 배열을 axis=0 기준으로 쌓기
centroids = np.stack(centroids, axis=0)
np.save(OUT_PATH, centroids)

print(f"[DONE] 클러스터 중심점 {n_clusters}개 저장 완료: {OUT_PATH}")
