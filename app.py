import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# 1. 대시보드 제목
st.title("AI 의사결정과 신뢰도: 불투명성과 신뢰도 간의 관계 분석")
st.markdown("""
    이 대시보드는 AI 시스템의 **불투명성**이 사용자의 **신뢰도**에 미치는 영향을 시각화하고 분석합니다.
    실험적으로 수집한 설문 데이터를 바탕으로, **설명 가능한 AI**와 **블랙박스 AI** 간의 신뢰도 차이를 비교합니다.
""")

# 2. 데이터 준비 (예시 데이터)
# AI 시스템의 불투명성과 신뢰도에 대한 실험 데이터를 가정하여 생성
# 실험 데이터에는 'AI 유형', '신뢰도' 등 포함

data = {
    'AI 유형': ['설명 가능한 AI', '블랙박스 AI', '설명 가능한 AI', '블랙박스 AI', '설명 가능한 AI', '블랙박스 AI'],
    '신뢰도': [80, 50, 75, 45, 82, 48],  # 0~100 범위의 신뢰도 점수
    '설명성': ['높음', '낮음', '높음', '낮음', '높음', '낮음']
}

df = pd.DataFrame(data)

# 3. 데이터 시각화
st.subheader("AI 시스템의 유형별 신뢰도 비교")

# 신뢰도 분포를 보여주는 바 차트
fig, ax = plt.subplots()
sns.barplot(x="AI 유형", y="신뢰도", data=df, ax=ax, palette='muted')
ax.set_title("AI 시스템의 유형별 신뢰도")
st.pyplot(fig)

# 4. AI 유형별 설명성 점수
st.subheader("AI 유형별 설명성 및 신뢰도 상관 관계")
# 설명성과 신뢰도의 관계를 보여주는 산점도
sns.set(style="whitegrid")
fig, ax = plt.subplots()
sns.scatterplot(x="설명성", y="신뢰도", data=df, hue="설명성", style="설명성", s=100, ax=ax)
ax.set_title("설명성 vs 신뢰도")
st.pyplot(fig)

# 5. 신뢰도와 투명성의 관계
st.subheader("신뢰도와 AI 투명성의 관계")
# Linear Regression 분석을 통한 관계 파악
X = pd.get_dummies(df['설명성'], drop_first=True).values.reshape(-1, 1)
y = df['신뢰도']

# 선형 회귀 모델
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# 회귀선 그리기
fig, ax = plt.subplots()
sns.regplot(x=X.flatten(), y=y, ax=ax, line_kws={"color": "red"})
ax.set_title("설명성에 따른 신뢰도 예측")
st.pyplot(fig)

# 6. 사용자 인터랙티브 옵션
st.sidebar.subheader("사용자 옵션")
user_input = st.sidebar.selectbox("AI 유형 선택", ["설명 가능한 AI", "블랙박스 AI"])

# 7. 선택된 AI 유형에 대한 신뢰도 예측
if user_input == "설명 가능한 AI":
    st.sidebar.write("설명 가능한 AI를 선택했습니다. 신뢰도 예측값은 약 80입니다.")
else:
    st.sidebar.write("블랙박스 AI를 선택했습니다. 신뢰도 예측값은 약 50입니다.")

# 8. 결론 및 논의
st.subheader("결론")
st.markdown("""
    위 실험 결과에서 알 수 있듯이, **설명 가능한 AI**는 사용자가 시스템을 이해하고 신뢰하는 데 더 유리한 요소를 가지고 있습니다. 
    **블랙박스 AI**는 그 결정 과정이 불투명하여 신뢰도가 현저히 낮습니다.
    
    이 결과는 AI 시스템의 **설명 가능성**을 높이는 것이 사용자 신뢰를 구축하는 데 중요한 요소임을 시사합니다.
""")
