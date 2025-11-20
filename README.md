# LoRA Style Map

CLIP Embedding 기반 LoRA 자동 클러스터링 & 프롬프트 스타일 추천 엔진

---

# 프로젝트 개요

본 프로젝트는 CivitAI LoRA 모델들을 수집해서 CLIP 모델로 이미지를 임베딩하고 군집화(KMeans)를 수행합니다. 그 다음에는 스타일 맵(UMAP 시각화)을 만들고 실시간 LoRA 자동 군집화, 텍스트 프롬프트 추천 시스템으로 이어지는 파이프라인 전체를 구현합니다.

[데모 영상](https://drive.google.com/file/d/1uffQd-OB8J0OO2V_sduoNDcEpBRQkUPG/view?usp=drive_link)

## 프로젝트 구성요소
- 비지도 클러스터링 기반 스타일 맵 구축
- 실시간 LoRA 자동 스타일 태깅
- 프롬프트 임베딩 기반 추천 시스템
- Streamlit Web UI 서비스화

---

# 주요 기능

## 1. CivitAI LoRA 모델 자동 수집

- REST API 반복 호출로 LoRA 모델 대표이미지 수백 장 이상 자동 확보
- 중복 제거 및 메타데이터 CSV 기록
- 손상 이미지 자동 검출 & 삭제

## 2. CLIP 기반 이미지 임베딩 (512-dim)

- openai/clip-vit-base-patch32 비전 인코더 활용
- 모든 LoRA 이미지 512차원 벡터 변환
- .npy 파일로 저장하여 재사용

## 3. PCA + UMAP 차원 축소

- 512 -> 50 (PCA)
- 50 -> 2 (UMAP)
- 스타일 맵(UMAP 클러스터) 시각화 생성

## 4. KMeans 군집화

- Elbow/Silhouette 분석 후 K 선택
- 이미지 스타일 타입 자동 분리 (비지도 학습)
- 각 이미지의 cluster_id CSV 기록
- 클러스터 중심점(centroids.npy) 계산

## 5. 실시간 LoRA 자동 분류

- 입력: CivitAI LoRA ID
- 대표 이미지 -> CLIP 임베딩 -> centroid 거리 비교
- 결과: 가장 가까운 스타일 클러스터 반환

## 6. 텍스트 프롬프트 스타일 추천

- CLIP 텍스트 인코더 사용
- 프롬프트 임베딩 후 centroid와 코사인 유사도 비교
- 가장 유사한 스타일 추천

## 7/ Streamlit 웹 서비스

- 프롬프트 스타일 추천
- LoRA ID 자동 분류
- UMAP 스타일 지도 시각화
- 클러스터별 대표 이미지 탐색

---

# 전체 시스템 아키텍처

**STEP 1**: CivitAI REST API
        -> LoRA 이미지 수집

**STEP 2**: Raw Images (data/raw/all)

**STEP 3**: CLIP Image Encoder (512-dim)
        -> embeddings.npy 저장

**STEP 4**: PCA → UMAP (2D)
        -> 2D 시각화용 좌표 생성

**STEP 5**: KMeans(k)
        -> cluster_labels.csv
        -> cluster_centroids.npy

**STEP 6**: classifier_core.py
        -> 임베딩 변환 / 코사인 유사도 / 최근접 클러스터 계산

**STEP 7**: Downstream
  - classify_new (단일 LoRA 분류)
  - recommend_by_prompt (텍스트 → s스타일 추천)
  - auto_classify_civitai (주기적 자동 분류)
  - Streamlit 서비스 UI

---

# 실행 순서

**Step 1** 이미지 수집
python -m scripts.download_civitai_data
python -m scripts.cleanup_corrupted_images

**Step 2** CLIP 임베딩 생성
python -m src.embed_clip_features

**Step 3** 군집 분석 (PCA -> UMAP -> KMeans)
python -m src.cluster_embeddings

**Step 4** Streamlit 실행
streamlit run app.py

---

# 프로젝트 의의

비지도 학습 기반 스타일 분석
사람이 라벨링하지 않아도 LoRA 스타일을 자동 분리합니다.

멀티모달 모델(CLIP) 직접 활용
이미지/텍스트 임베딩을 모두 처리하는 구조를 만듭니다.

추천 시스템 & 시각화 결합
단순 분류가 아니라 "스타일 맵" 모델을 구축합니다.

실시간 API 응용
CivitAI와 연계된 실서비스형 기능을 구현합니다.

---

# 라이선스 및 출처

- CLIP 모델: OpenAI (MIT License)
- CivitAI API 사용

*※ 본 프로젝트의 데이터는 공개된 이미지의 링크를 활용합니다.*