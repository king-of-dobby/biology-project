import streamlit as st
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

---

**입력 안내:**  
- 일부 항목은 드래그 앤 드롭 형식으로 입력할 수 있어요.  
- 크기/길이 관련 항목에는 단위를 확인해 주세요 (예: cm).  
- 범주형은 색상, 패턴, 음성 등 단계별 선택 가능해요.  
- 0/1은 해당 특성의 유무를 나타냅니다.
""")

# 항목 구분
drag_and_drop_features = {
    "Colorful body": [0, 1, 2],
    "Body texture": [0, 1, 2],
    "Snout shape": [0, 1, 2],
    "Flight maneuverability": [0, 1, 2],
    "Defense mechanism": [0, 1, 2],
}

binary_features = [
    "Has fur", "Has feathers", "Has scales", "Has wings",
    "Has tail", "Has horns", "Has beak", "Patterned skin",
    "Nocturnal", "Can fly", "Can swim", "Has claws",
    "Body symmetry", "Has antennae", "Has exoskeleton",
    "Has internal skeleton", "Makes sound", "Can camouflage",
    "Has teeth", "Breathes through gills", "Has shell",
    "Is warm-blooded", "Leg claws", "Webbed feet",
    "Can glow", "Vocal mimicry"
]

even_number_features = {
    "Number of legs": [0, 2, 4, 6, 8, 10],
    "Number of eyes": [0, 2, 4, 6, 8]
}

cm_input_features = {
    "Horns length (cm)": (1, 500),
    "Eye size (cm)": (1, 500),
    "Tail length (cm)": (1, 500),
    "Mouth size": (1, 500),
}

slider_features = {
    "Body size": (1, 10),
    "Aggressiveness": (0, 5),
    "Beak sharpness": (0, 5),
    "Leg length": (0, 2),
    "Fin shape complexity": (0, 2),
    "Wing span": (0, 2),
    "Color contrast": (0, 2),
    "Voice volume": (0, 6),
    "Ear shape": (1, 10),
    "Hair density": (1, 10),
}

# 50개 항목
features_order = [
    "Has fur", "Has feathers", "Has scales", "Has wings", "Number of legs",
    "Number of fins", "Has tail", "Has horns", "Has beak", "Colorful body",
    "Patterned skin", "Body size", "Aggressiveness", "Nocturnal", "Can fly",
    "Can swim", "Has claws", "Body symmetry", "Has antennae", "Has exoskeleton",
    "Has internal skeleton", "Makes sound", "Can camouflage", "Has teeth", "Breathes
