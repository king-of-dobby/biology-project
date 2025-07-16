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

**입력 안내:**  
- 숫자만 입력 가능  
- 항목별 단위와 설명을 참고하여 정확히 입력해주세요.
""")

# 특징 항목 설명 확장
features_info = [
    ("Has fur", [0, 1], "털 유무 (0=없음, 1=있음)"),
    ("Has feathers", [0, 1], "깃털 유무"),
    ("Has scales", [0, 1], "비늘 유무"),
    ("Has wings", [0, 1], "날개 유무"),
    ("Number of legs", [0, 2, 4, 6, 8, 10], "다리 개수 (짝수만)"),
    ("Number of fins", list(range(0, 7)), "지느러미 수"),
    ("Has tail", [0, 1], "꼬리 유무"),
    ("Has horns", [0, 1], "뿔 유무"),
    ("Has beak", [0, 1], "부리 유무"),
    ("Colorful body", [0, 1, 2], "몸 색상: 0=단색, 1=보통, 2=화려"),
    ("Patterned skin", [0, 1], "피부 무늬 유무"),
    ("Body size", list(range(1, 11)), "몸 크기 (1=작음 ~ 10=큼)"),
    ("Aggressiveness", list(range(0, 6)), "공격성 (0~5)"),
    ("Nocturnal", [0, 1], "야행성 여부"),
    ("Can fly", [0, 1], "비행 가능 여부"),
    ("Can swim", [0, 1], "수영 가능 여부"),
    ("Has claws", [0, 1], "발톱 유무"),
    ("Body symmetry", [0, 1], "좌우 대칭 여부"),
    ("Has antennae", [0, 1], "더듬이 유무"),
    ("Has exoskeleton", [0, 1], "외골격 유무"),
    ("Has internal skeleton", [0, 1], "내골격 유무"),
    ("Makes sound", [0, 1], "소리 유발 여부"),
    ("Can camouflage", [0, 1], "위장 능력 여부"),
    ("Has teeth", [0, 1], "이빨 유무"),
    ("Breathes through gills", [0, 1], "아가미 호흡 여부"),
    ("Has fur (intensity)", list(range(1, 11)), "털 양 (1=없음 ~ 10=풍부)"),
    ("Beak sharpness", list(range(0, 6)), "부리 날카로움 (0~5)"),
    ("Leg length", [0, 1, 2], "다리 길이 (0=짧음, 2=김)"),
    ("Fin shape complexity", [0, 1, 2], "지느러미 형태 복잡성"),
    ("Wing span", [0, 1, 2], "날개폭 범위"),
    ("Horns length (cm)", list(range(1, 501)), "뿔 길이 (cm)"),
    ("Body texture", [0, 1, 2], "몸 표면 질감"),
    ("Has shell", [0, 1], "껍질 유무"),
    ("Is warm-blooded", [0, 1], "항온성 여부"),
    ("Eye size (cm)", list(range(1, 501)), "눈 크기 (cm)"),
    ("Color contrast", [0, 1, 2], "색 대비 강도"),
    ("Number of eyes", [0, 2, 4, 6, 8], "눈 개수 (짝수만)"),
    ("Tail length (cm)", list(range(1, 501)), "꼬리 길이 (cm)"),
    ("Snout shape", [0, 1, 2], "주둥이 형태"),
    ("Voice volume", list(range(0, 7)), "소리 크기 (0~6)"),
    ("Leg claws", [0, 1], "다리 발톱 유무"),
    ("Webbed feet", [0, 1], "물갈퀴 유무"),
    ("Ear shape", list(range(1, 11)), "귀 형태 크기 (1~10)"),
    ("Can glow", [0, 1], "발광 여부"),
    ("Vocal mimicry", [0, 1], "소리 흉내 여부"),
    ("Flight maneuverability", [0, 1, 2], "비행 기동성"),
    ("Mouth size", list(range(1, 501)), "입 크기 (cm)"),
    ("Spine visibility", [0, 1, 2], "척추 외형 가시성"),
    ("Defense mechanism", [0, 1, 2], "방어 수단 다양성"),
    ("Hair density", list(range(1, 11)), "털 밀도 (1~10)"),
]

# ⬇️ 공통 입력 함수
def collect_features(label, prefix):
    st.subheader(f"🔍 {label} 특징 입력")
    features = []
    cols = st.columns(5)
    for i, (desc, options, help_text) in enumerate(features_info):
        with cols[i % 5]:
            if "cm" in desc.lower():
                val = st.number_input(f"{desc}", min_value=min(options), max_value=max(options), step=1,
                                      help=help_text, key=f"{prefix}_{i}")
            elif desc in ["Number of legs", "Number of eyes"]:
                val = st.select_slider(f"{desc}", options=options, help=help_text, key=f"{prefix}_{i}")
            elif len(options) > 20:
                val = st.number_input(f"{desc}", min_value=min(options), max_value=max(options), step=1,
                                      help=help_text, key=f"{prefix}_{i}")
            else:
                val = st.selectbox(f"{desc}", options, help=help_text, key=f"{prefix}_{i}")
            features.append(val)
    return features

# 생물 1, 2 입력
features_1 = collect_features("Organism 1", "o1")
features_2 = collect_features("Organism 2", "o2")

# 예측
if st.button("🔎 예측하기"):
    input_data = np.array(features_1 + features_2).reshape(1, -1)
    st.info(f"총 입력된 특징 수: {input_data.shape[1]}개")

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success(f"✅ AI의 판단: 두 생물은 **같은 종**일 가능성이 높습니다! (확률: {proba[1]:.2f})")
    else:
        st.error(f"❌ AI의 판단: 두 생물은 **다른 종**일 가능성이 높습니다. (확률: {proba[0]:.2f})")
