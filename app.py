# streamlit run app.py
"""
Streamlit UI – LoRA CLIP Style Map & Recommendation System

기능
1. 텍스트 프롬프트 → 클러스터 추천
2. LoRA ID → 자동 분류
3. UMAP 스타일 맵 시각화 표시
4. 클러스터별 대표 이미지 미리보기

전제
- classifier_core.py
- recommend_by_prompt.py
- auto_classify_civitai.py
- cluster_centroids.npy, cluster_labels.csv, umap_clusters.png
"""
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

from src.recommend_by_prompt import recommend_cluster
from src.classifier_core import (
    load_clip,
    load_centroids,
    embed_image,
    embed_text,
    predict_cluster,
)
from src.config import Config

import requests
from io import BytesIO
from PIL import Image

# 페이지 설정
st.set_page_config(
    page_title="LoRA Style Map",
    # 레이아웃 설정
    layout="wide",
)

st.title("LoRA CLIP Style Map")
st.write("CLIP 임베딩 기반 자동 스타일 클러스터링")


# 에셋 불러오기
CLUSTER_DIR = Config.OUTPUT_DIR / "clusters"
centroid_path = CLUSTER_DIR / "cluster_centroids.npy"
cluster_labels_csv = CLUSTER_DIR / "cluster_labels.csv"
umap_img_path = CLUSTER_DIR / "umap_clusters.png"
raw_image_dir = Config.RAW_DIR / "all"

# 로드
centroids = np.load(centroid_path)
label_df = pd.read_csv(cluster_labels_csv)


# 탭 선언
tab1, tab2, tab3, tab4 = st.tabs([
    "Prompt 클러스터",
    "LoRA ID 기반 자동 클러스터",
    "UMAP 스타일 맵",
    "클러스터별 대표 이미지",
])


# TAB 1: Style Cluster Recommendation
with tab1:
    st.header("텍스트 프롬프트 기반 스타일 추천")
    text = st.text_area("프롬프트 입력", "")

    if st.button("클러스터 추천"):
        if text.strip() == "":
            st.warning("프롬프트를 입력해주세요.")
        else:
            cid = recommend_cluster(text)
            st.success(f"추천된 클러스터 ID: **{cid}**")

            # 클러스터 이미지 일부 미리보기
            st.subheader(f"클러스터 {cid} 대표 이미지 미리보기")
            examples = label_df[label_df.cluster_id == cid].sample(8, replace=True)

            cols = st.columns(4)
            
            # iterrows()는 (index, row)를 반환하므로 (idx, row) 또는 (_, row)로 언패킹한다
            for i, (idx, row) in enumerate(examples.iterrows()):
                img_path = raw_image_dir / row["filename"]
                if img_path.exists():
                    img = Image.open(img_path)
                    cols[i % 4].image(img, use_container_width=True)



# TAB 2: LoRA ID Automatic Clustering
with tab2:
    st.header("LoRA ID 기반 자동 클러스터링")

    lora_id = st.text_input("LoRA Model ID 입력 (예: 123456)")

    if st.button("LoRA 이미지 불러와서 분류"):
        if not lora_id.isdigit():
            st.warning("정확한 LoRA ID를 입력해주세요.")
        else:
            try:
                info_url = f"https://civitai.com/api/v1/models/{lora_id}"
                resp = requests.get(info_url, timeout=10).json()
                img_url = resp["modelVersions"][0]["images"][0]["url"]

                st.write(f"이미지 URL: {img_url}")
                img_data = requests.get(img_url).content
                img = Image.open(BytesIO(img_data))

                st.image(img, caption="LoRA preview", width=300)

                # CLIP 임베딩 -> 클러스터 분류
                clip_model, clip_processor, device = load_clip()
                # 대표 이미지 임베딩
                embedied_image = embed_image(img, clip_model, clip_processor, device)
                
                cluster_id = predict_cluster(embedied_image, centroids)

                st.success(f"이 LoRA 모델은 클러스터 **{cluster_id}**에 속합니다")

            except Exception as e:
                st.error(f"오류 발생: {e}")


# TAB 3: UMAP Style Map
with tab3:
    st.header("3. UMAP 기반 스타일 맵 시각화")
    if umap_img_path.exists():
        st.image(str(umap_img_path), use_container_width=True)
    else:
        st.warning("UMAP 이미지가 존재하지 않습니다. cluster_embeddings.py 먼저 실행해주세요.")


# TAB 4: Cluster Sample Browser
with tab4:
    st.header("4. 클러스터별 대표 이미지 탐색")

    cluster_list = sorted(label_df.cluster_id.unique())
    selected_cluster = st.selectbox("클러스터 선택", cluster_list)

    st.subheader(f"클러스터 {selected_cluster} 이미지")

    df_cluster = label_df[label_df.cluster_id == selected_cluster]

    cols = st.columns(4)

    # iterrows()로 인덱스와 row를 받는다
    for i, (idx, row) in enumerate(df_cluster.sample(min(24, len(df_cluster))).iterrows()):
        img_path = raw_image_dir / row["filename"]
        if img_path.exists():
            img = Image.open(img_path)
            cols[i % 4].image(img, use_container_width=True)
