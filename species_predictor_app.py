# species_predictor_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 모델 로드
model = joblib.load("random_forest_species_model_descriptive.pkl")

# 페이지 설정
st.set_page_config(page_title="Species Classifier", layout="wide")
st.title("🧬 생물 쌍 분류기: 같은 종인가요?")

st.markdown("""
이 웹앱은 두 생물의 **외형적 특징 50가지씩**을 바탕으로,  
**사람이 같은 종이라고 판단할지** AI가 예측하는 도구입니다.  
우리 정환이 사랑해!! 기다려줘서 고마워!! 히히 이거 한 줄 추가한다고.💕❤️

**입력 안내:**  
- 숫자만 입력 가능  
- 아래 각 항목 옆에 있는 설명을 보고 정확히 입력해주세요.
""")

# === 특징 설명 및 입력 범위 ===
features_info = [
    ("Has fur", [0, 1]),                        # binary
    ("Has feathers", [0, 1]),
    ("Has scales", [0, 1]),
    ("Has wings", [0, 1]),
    ("Number of legs", list(range(0, 11, 2))),  # 0, 2, ..., 10
    ("Number of fins", list(range(0, 7))),      # 0~6
    ("Has tail", [0, 1]),
    ("Has horns", [0, 1]),
    ("Has beak", [0, 1]),
    ("Colorful body", [0, 1, 2]),               # ordinal
    ("Patterned skin", [0, 1]),
    ("Body size", list(range(1, 11))),          # 1 (small) ~ 10 (large)
    ("Aggressiveness", list(range(0, 6))),      # 0~5
    ("Nocturnal", [0, 1]),
    ("Can fly", [0, 1]),
    ("Can swim", [0, 1]),
    ("Has claws", [0, 1]),
    ("Body symmetry", [0, 1]),
    ("Has antennae", [0, 1]),
    ("Has exoskeleton", [0, 1]),
    ("Has internal skeleton", [0, 1]),
    ("Makes sound", [0, 1]),
    ("Can camouflage", [0, 1]),
    ("Has teeth", [0, 1]),
    ("Breathes through gills", [0, 1]),
    
    # 26~50
    ("Has fur (intensity)", [0, 1, 2]),
    ("Beak sharpness", [0, 1, 2]),
    ("Leg length", [0, 1, 2]),
    ("Fin shape complexity", [0, 1, 2]),
    ("Wing span", [0, 1, 2]),
    ("Horns length", [0, 1, 2]),
    ("Body texture", [0, 1, 2]),
    ("Has shell", [0, 1]),
    ("Is warm-blooded", [0, 1]),
    ("Eye size", [0, 1, 2]),
    ("Color contrast", [0, 1, 2]),
    ("Number of eyes", [0, 2, 4, 6]),
    ("Tail length", [0, 1, 2]),
    ("Snout shape", [0, 1, 2]),
    ("Voice volume", [0, 1, 2]),
    ("Leg claws", [0, 1]),
    ("Webbed feet", [0, 1]),
    ("Ear shape", [0, 1, 2]),
    ("Can glow", [0, 1]),
    ("Vocal mimicry", [0, 1]),
    ("Flight maneuverability", [0, 1, 2]),
    ("Mouth size", [0, 1, 2]),
    ("Spine visibility", [0, 1, 2]),
    ("Defense mechanism", [0, 1, 2]),
    ("Hair density", [0, 1, 2])
]

# Organism 1 입력
st.subheader("🔵 Organism 1 특징 입력")
features_1 = []
cols1 = st.columns(5)
for i, (desc, options) in enumerate(features_info):
    with cols1[i % 5]:
        val = st.selectbox(f"{i+1}. {desc}", options, key=f"o1_{i}")
        features_1.append(val)

# Organism 2 입력
st.subheader("🟢 Organism 2 특징 입력")
features_2 = []
cols2 = st.columns(5)
for i, (desc, options) in enumerate(features_info):
    with cols2[i % 5]:
        val = st.selectbox(f"{i+1}. {desc}", options, key=f"o2_{i}")
        features_2.append(val)

# 예측
if st.button("🔍 예측하기"):
    input_data = np.array(features_1 + features_2).reshape(1, -1)
    
    # 입력 확인
    st.info(f"총 입력된 특징 수: {input_data.shape[1]}개")

    # 예측
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success(f"✅ AI의 판단: 두 생물은 **같은 종**일 가능성이 높습니다! (확률: {proba[1]:.2f})")
    else:
        st.error(f"❌ AI의 판단: 두 생물은 **다른 종**일 가능성이 높습니다. (확률: {proba[0]:.2f})")
