# species_predictor_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 모델 불러오기
model = joblib.load("random_forest_species_model_descriptive.pkl")

st.set_page_config(page_title="Species Classifier", layout="wide")
st.title("🧬 생물 쌍 분류기: 같은 종인가요?")

st.markdown("""
두 생물의 **50가지 외형적 특징**을 기반으로, 사람이 같은 종으로 판단할 가능성을 AI가 예측합니다.  
**범위/선택지**는 각 특징에 맞게 설정되어 있습니다.
""")

# 특징 설명 + 입력 방식 정의
feature_definitions = [
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
    ("Patterned skin", [0, 1, 2]),
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
    ("Has whiskers", [0, 1]),
    ("Length (cm)", list(range(1, 501))),       # 1~500cm
    ("Height (cm)", list(range(1, 501))),
    ("Weight (kg)", list(range(1, 201))),
    ("Has scales on legs", [0, 1]),
    ("Has webbed feet", [0, 1]),
    ("Has shell", [0, 1]),
    ("Number of eyes", list(range(0, 11))),
    ("Infrared vision", [0, 1]),
    ("Ultraviolet vision", [0, 1]),
    ("Habitat: water", [0, 1]),
    ("Habitat: land", [0, 1]),
    ("Habitat: air", [0, 1]),
    ("Has spinnerets", [0, 1]),
    ("Produces venom", [0, 1]),
    ("Can regenerate", [0, 1]),
    ("Has suction cups", [0, 1]),
    ("Segmented body", [0, 1]),
    ("Spine flexibility", list(range(1, 11))),
    ("Intelligence (subjective)", list(range(1, 6))),
    ("Average lifespan (years)", list(range(1, 101))),
    ("Reproductive rate", list(range(1, 11))),
    ("Parental care", [0, 1]),
    ("Social behavior", [0, 1])
]

# Organism 입력 함수
def input_organism_features(label_prefix):
    st.subheader(f"{label_prefix} 생물 특징 입력")
    inputs = []
    cols = st.columns(5)
    for i, (desc, options) in enumerate(feature_definitions):
        with cols[i % 5]:
            if len(options) <= 3:
                val = st.selectbox(f"{label_prefix} - {desc}", options, key=f"{label_prefix}_{i}")
            elif len(options) <= 20:
                val = st.slider(f"{label_prefix} - {desc}", min_value=min(options), max_value=max(options), key=f"{label_prefix}_{i}")
            else:
                val = st.number_input(f"{label_prefix} - {desc}", min_value=min(options), max_value=max(options), step=1, key=f"{label_prefix}_{i}")
            inputs.append(val)
    return inputs

# 두 생물의 특징 입력 받기
features_1 = input_organism_features("🔵 Organism 1")
features_2 = input_organism_features("🟢 Organism 2")

# 예측
if st.button("🔍 예측하기"):
    input_data = np.array(features_1 + features_2).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success(f"✅ AI의 판단: **같은 종**일 가능성이 높습니다! (확률: {proba[1]:.2f})")
    else:
        st.error(f"❌ AI의 판단: **다른 종**일 가능성이 높습니다. (확률: {proba[0]:.2f})")
