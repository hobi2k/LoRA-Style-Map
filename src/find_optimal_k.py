# src/find_optimal_k.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import umap as umap
from pathlib import Path
from src.config import Config

"""
최적의 K를 찾기 위한 기능 스크립트

기능:
- 본래 CLIP 임베딩(512차원), PCA 결과(50차원), 또는 UMAP 결과(2차원)를 사용할 수 있으나,
KMeans의 k 최적값을 찾을 때는 UMAP 2차원 임베딩이 군집 구조를 가장 잘 드러내므로
이 스크립트에서는 UMAP 결과를 입력으로 사용합니다.
- K=2~20 구간에서 엘보우와 실루엣 점수를 계산
- 엘보우가 대략적인 k 범위를 제시한다면, 실루엣 점수는 각 데이터 점이 얼마나 자기 군집에 잘 속하는지를 수치로 나타냅니다.
- 시각화 그래프 저장
"""

# 경로 설정
EMB_PATH = Config.OUTPUT_DIR / "embeddings" / "clip_features.npy"
NAME_PATH = Config.OUTPUT_DIR / "embeddings" / "clip_filenames.npy"
OUTPUT_DIR = Config.OUTPUT_DIR / "clusters"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 임베딩 로드
features = np.load(EMB_PATH)
print(f"[INFO] 불러온 임베딩: {features.shape}")

# 차원 축소 작업
print("[STEP] PCA 진행 (512 -> 50)")
pca = PCA(n_components=50, random_state=42)
pca_features = pca.fit_transform(features)

print("[STEP] UMAP 진행 (50 -> 2)")
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_features = umap_model.fit_transform(pca_features)

# k 범위 설정
K_RANGE = range(2, 21)

# 엘보우 기법 사용
inertia = []
print("[INFO] 엘보우 기법 적용")
for k in K_RANGE:
    # n_init=10으로 설정하여 초기화 반복 횟수 증가
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(umap_features)
    inertia.append(kmeans.inertia_)

plt.plot(K_RANGE, inertia, marker='o')
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
# dpi 설정으로 고해상도 저장
plt.savefig(OUTPUT_DIR / "elbow_curve.png", dpi=200)
# 다음 작업을 위해 figure 객체 닫기
plt.close()
print("[SAVED] elbow_curve.png")

# 실루엣 점수 기법 사용
sil_scores = []
print("[INFO] 실루엣 점수 기법 적용")
for k in K_RANGE:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(umap_features)
    score = silhouette_score(umap_features, labels)
    sil_scores.append(score)

plt.plot(K_RANGE, sil_scores, marker='o')
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores for Optimal k")
plt.grid(True)
plt.savefig(OUTPUT_DIR / "silhouette_scores.png", dpi=200)
plt.close()
print("[SAVED] silhouette_scores.png")

print("[DONE] 최적 k 도출 완료")
