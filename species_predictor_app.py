
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
1000쌍의 생물을 50가지 특징을 기준으로 인간이 같은 종인지 아닌지 분류한 것을 학습시켰습니다.  
실제 생물 다양성을 생각하면 기준과 학습한 자료가 턱없이 부족하기에 정확도가 높은 AI는 아니지만  
그냥 어떤 인간이 이런 뻘짓을 했군... 하는 의미 정도로 봐주세요


**입력 안내:**  
- 숫자만 입력 가능  
- 일부 항목은 입력값 단위가 명시되어 있습니다.  
- 아래 각 항목 옆에 있는 설명을 보고 정확히 입력해주세요.
""")

# 특징 그룹 정의
descriptive_features = {
    "이진 특징 (예: 유무)": [
        ("Has fur", [0, 1], "털이 있나요? (0=없음, 1=있음)"),
        ("Has feathers", [0, 1], "깃털이 있나요?"),
        ("Has scales", [0, 1], "비늘이 있나요?"),
        ("Has wings", [0, 1], "날개가 있나요?"),
        ("Has tail", [0, 1], "꼬리가 있나요?"),
        ("Has horns", [0, 1], "뿔이 있나요?"),
        ("Has beak", [0, 1], "부리가 있나요?"),
        ("Patterned skin", [0, 1], "피부에 무늬가 있나요?"),
        ("Nocturnal", [0, 1], "야행성인가요?"),
        ("Can fly", [0, 1], "날 수 있나요?"),
        ("Can swim", [0, 1], "헤엄칠 수 있나요?"),
        ("Has claws", [0, 1], "발톱이 있나요?"),
        ("Body symmetry", [0, 1], "신체 좌우대칭인가요?"),
        ("Has antennae", [0, 1], "더듬이가 있나요?"),
        ("Has exoskeleton", [0, 1], "외골격이 있나요?"),
        ("Has internal skeleton", [0, 1], "내골격이 있나요?"),
        ("Makes sound", [0, 1], "소리를 내나요?"),
        ("Can camouflage", [0, 1], "위장 능력이 있나요?"),
        ("Has teeth", [0, 1], "이빨이 있나요?"),
        ("Breathes through gills", [0, 1], "아가미로 호흡하나요?"),
        ("Has shell", [0, 1], "껍질이 있나요?"),
        ("Is warm-blooded", [0, 1], "온혈 동물인가요?"),
        ("Leg claws", [0, 1], "다리에 발톱이 있나요?"),
        ("Webbed feet", [0, 1], "물갈퀴가 있나요?"),
        ("Can glow", [0, 1], "자체 발광하나요?"),
        ("Vocal mimicry", [0, 1], "소리 흉내를 낼 수 있나요?")
    ],
    "수치 또는 단계적 특징": [
        ("Number of legs", [0, 2, 4, 6, 8, 10], "다리 개수 (짝수)"),
        ("Number of fins", list(range(0, 7)), "지느러미 개수 (0~6)"),
        ("Colorful body", [0, 1, 2], "몸 색상 (0=단색, 2=화려)"),
        ("Body size", list(range(1, 11)), "몸집 크기 (1~10)"),
        ("Aggressiveness", list(range(0, 6)), "공격성 (0~5)"),
        ("Has fur (intensity)", list(range(1, 11)), "털의 밀도 (1~10)"),
        ("Beak sharpness", list(range(0, 6)), "부리 날카로움 (0~5)"),
        ("Leg length", [0, 1, 2], "다리 길이 단계 (0~2)"),
        ("Fin shape complexity", [0, 1, 2], "지느러미 모양 복잡도 (0~2)"),
        ("Wing span", [0, 1, 2], "날개 폭 단계 (0~2)"),
        ("Horns length (cm)", list(range(1, 501)), "뿔 길이 (cm)"),
        ("Body texture", [0, 1, 2], "몸 표면 질감 (0~2)"),
        ("Eye size (cm)", list(range(1, 501)), "눈 크기 (cm)"),
        ("Color contrast", [0, 1, 2], "색 대비 (0~2)"),
        ("Number of eyes", [0, 2, 4, 6, 8], "눈 개수"),
        ("Tail length (cm)", list(range(1, 501)), "꼬리 길이 (cm)"),
        ("Snout shape", [0, 1, 2], "주둥이 형태"),
        ("Voice volume", list(range(0, 7)), "소리 크기 (0~6)"),
        ("Ear shape", list(range(1, 11)), "귀 모양 (1~10)"),
        ("Flight maneuverability", [0, 1, 2], "비행 기동성 (0~2)"),
        ("Mouth size", list(range(1, 501)), "입 크기 (cm)"),
        ("Spine visibility", [0, 1, 2], "척추 돌출 정도"),
        ("Defense mechanism", [0, 1, 2], "방어 방식 (0~2)"),
        ("Hair density", list(range(1, 11)), "털 밀도 (1~10)")
    ]
}

def render_input_block(title, key_prefix):
    st.subheader(title)
    inputs = []
    for category, features in descriptive_features.items():
        with st.expander(f"🗂️ {category}"):
            cols = st.columns(3)
            for i, (label, options, help_text) in enumerate(features):
                with cols[i % 3]:
                    if len(options) > 30:
                        val = st.number_input(label, min_value=min(options), max_value=max(options),
                                              step=1, help=help_text, key=f"{key_prefix}_{label}")
                    else:
                        val = st.selectbox(label, options, help=help_text, key=f"{key_prefix}_{label}")
                    inputs.append(val)
    return inputs

# Organism 1 입력
features_1 = render_input_block("🔵 Organism 1 특징 입력", "o1")

# Organism 2 입력
features_2 = render_input_block("🟢 Organism 2 특징 입력", "o2")

# 예측
if st.button("🔍 예측하기"):
    input_data = np.array(features_1 + features_2).reshape(1, -1)
    st.info(f"총 입력된 특징 수: {input_data.shape[1]}개")

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success(f"✅ AI의 판단: 두 생물은 **같은 종**일 가능성이 높습니다! (확률: {proba[1]:.2f})")
    else:
        st.error(f"❌ AI의 판단: 두 생물은 **다른 종**일 가능성이 높습니다. (확률: {proba[0]:.2f})")

st.markdown("""
    <div style='text-align: center; font-size: 15px;'>
        Copyright 2025. Yoon Ji Young. All rights reserved.  
고생했다 나 자신🌻 
    </div>
""", unsafe_allow_html=True)
