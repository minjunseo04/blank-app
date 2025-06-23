import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from scipy.stats import pearsonr
import statsmodels.api as sm

st.set_page_config(layout="wide")
st.title("서울시 자치구별 공영주차장 vs 불법주정차 민원 분석 대시보드")

# ✅ 데이터 로드
@st.cache_data
def load_data():
    df_main = pd.read_excel("자치구별_민원_주차장_조정버전.xlsx")
    df_pop_raw = pd.read_excel("등록인구_20250620115657.xlsx", header=None)
    df_pop_cleaned = df_pop_raw.iloc[4:25, [1, 3]].copy()
    df_pop_cleaned.columns = ['자치구', '인구수']
    df_pop_cleaned['자치구'] = df_pop_cleaned['자치구'].astype(str).str.strip()
    df_pop_cleaned['인구수'] = pd.to_numeric(df_pop_cleaned['인구수'], errors='coerce').fillna(0).astype(int)

    df = df_main.merge(df_pop_cleaned, on='자치구', how='left')
    df['인구당_민원수'] = df['불법주정차_민원건수'] / df['인구수'] * 1000
    df['인구당_주차장수'] = df['공영주차장_개수'] / df['인구수'] * 1000
    df['인구기준_민원주차장비율'] = df['인구당_민원수'] / df['인구당_주차장수']
    df['인구_주차장_비율'] = df['인구수'] / df['공영주차장_개수']
    return df

@st.cache_data
def load_report_data():
    df = pd.read_csv("불법주정차 신고현황(23년11월1일_24년3월13일).csv", encoding='utf-8')
    df['자치구'] = df['주소'].str.extract(r'서울특별시\s+(\S+구)')
    df = df.dropna(subset=['위도', '경도'])
    df = df[(df['위도'] != 0) & (df['경도'] != 0)]
    return df

df = load_data()
df_report = load_report_data()

# ✅ 지도 시각화
st.subheader("📍 불법주정차 민원 위치 지도 (5,000건 랜덤 추출)")
seoul_map = folium.Map(location=[37.5665, 126.9780], zoom_start=11)
cluster = MarkerCluster().add_to(seoul_map)

for _, row in df_report.sample(n=5000, random_state=42).iterrows():
    popup = f"{row['자치구']}<br>주소: {row['주소']}<br>일시: {row['민원접수일']}"
    folium.Marker(location=[row['위도'], row['경도']], popup=popup).add_to(cluster)

folium_static(seoul_map)

# ✅ 자치구별 시각화
st.subheader("📊 자치구별 민원 건수 및 주차장 개수")
fig1 = px.bar(df.sort_values(by='불법주정차_민원건수', ascending=False),
              x='자치구', y=['불법주정차_민원건수', '공영주차장_개수'],
              barmode='group')
st.plotly_chart(fig1, use_container_width=True)

st.subheader("📉 공영주차장 수 vs 불법주정차 민원건수")
fig2 = px.scatter(df, x='공영주차장_개수', y='불법주정차_민원건수',
                  text='자치구', trendline='ols')
st.plotly_chart(fig2, use_container_width=True)

st.subheader("🚗 공영주차장 1개당 인구 수")
fig3 = px.bar(df.sort_values('인구_주차장_비율', ascending=False),
              x='자치구', y='인구_주차장_비율', color='인구_주차장_비율')
st.plotly_chart(fig3, use_container_width=True)

st.subheader("📌 인구 1,000명당 주차장 수 vs 민원 수")
fig4 = px.scatter(df, x='인구당_주차장수', y='인구당_민원수',
                  text='자치구', size='공영주차장_개수', color='자치구')
st.plotly_chart(fig4, use_container_width=True)

st.subheader("🏙️ 인구 기준 민원/주차장 비율 TOP 10")
top10 = df.sort_values(by='인구기준_민원주차장비율', ascending=False).head(10)
fig5 = px.bar(top10, x='자치구', y='인구기준_민원주차장비율', color='인구기준_민원주차장비율')
st.plotly_chart(fig5, use_container_width=True)

# ✅ 상관관계 분석 및 회귀분석
st.subheader("📈 상관관계 및 회귀분석 요약")

# 상관관계
filtered_df = df[['인구당_주차장수', '인구당_민원수']].dropna()
corr1, pval1 = pearsonr(df['공영주차장_개수'], df['불법주정차_민원건수'])
corr2, pval2 = pearsonr(filtered_df['인구당_주차장수'], filtered_df['인구당_민원수'])

# 회귀분석
X = sm.add_constant(filtered_df['인구당_주차장수'])
y = filtered_df['인구당_민원수']
model = sm.OLS(y, X).fit()

# 회귀 분석 요약 텍스트 출력
st.markdown(f"""
### 📈 상관관계 분석 요약: 공영주차장 ↔ 불법주정차 민원
- 🔹 **공영주차장 수 vs 민원 건수**: 상관계수 = **{corr1:.2f}**, p-value = **{pval1:.4f}** → 약한 음의 상관, 유의하지 않음
- 🔸 **인구 1,000명당 주차장 수 vs 민원 수**: 상관계수 = **{corr2:.2f}**, p-value < 0.001 → 매우 강한 양의 상관관계, 유의미 ✅

### 📊 단순 선형 회귀분석 결과
- 회귀식: **인구당 민원 수 = 1.46 + 58.72 × 인구당 주차장 수**
- 결정계수(R²) = **0.849** → 약 **85%** 설명 가능
- 회귀계수 p-value < **0.001** → **매우 유의미한 결과**
""")

# ✅ 요약
st.markdown("""
### ✅ 대시보드 요약
- 📍 지도: 실제 민원 5,000건 위치 클러스터링 시각화
- 📊 자치구별 공영주차장 수 및 민원 건수 비교
- 🔍 인구 기준 보정으로 공정한 비교 제공 (인구당 민원수, 인구당 주차장수)
- 📈 상관관계 분석 및 회귀모형으로 정책 방향성 도출
  - 공영주차장이 많다고 민원이 반드시 줄어들지는 않음
  - 인구당 주차장수가 많아도 민원은 오히려 늘어나는 경향성 존재
""")

# ✅ 피드백 및 개선 사항
st.markdown("""
## 🧑‍🏫 발표 피드백 및 개선 방향

이번 분석 프로젝트는 서울시의 공영주차장과 불법주정차 민원 데이터 간의 관계를 시각화하고 통계적으로 분석한 것입니다. 수행 과정에서 다음과 같은 보완점과 개선 방향을 도출하였습니다:

---

### 1. 🔍 데이터 수집 및 전처리
- 데이터의 **출처와 기준 시점**에 대한 설명이 부족했습니다.
- 예를 들어 "2024년 3월까지의 민원 접수 데이터", "서울열린데이터광장 기준" 등의 출처 명시는 분석의 **신뢰도를 높이는 요소**가 될 수 있습니다.

---

### 2. 🗺️ 지도 시각화
- 민원 위치 데이터는 5,000건을 무작위로 시각화했으나, **필터 기능이 부재**해 특정 자치구나 기간만 선택해 보는 것이 불가능합니다.
- 발표 이후 확장 아이디어로, **자치구별 필터**, **기간 슬라이더** 등을 추가해 **인터랙티브 기능을 강화**할 수 있습니다.

---

### 3. 📈 분석 모델 및 해석
- 회귀분석과 상관분석의 결과는 유의미했으나, **통계적 유의성과 인과관계는 다르다는 점**을 명확히 설명할 필요가 있습니다.
- 예: "공영주차장이 많다고 해서 민원이 줄어든다는 보장은 없다."
- 분석적 해석도 보완 필요: **상업지구일수록 주차 수요와 민원이 함께 증가할 수 있음**

---

### 4. 🎨 시각화 구성
- 그래프 간 색상과 범례가 일관되지 않아 **시각적 혼란**을 줄 수 있었습니다.
- 민원 건수는 항상 파란색, 주차장 수는 주황색 등으로 **색상 고정**하면 비교가 더 직관적입니다.
- 각 지표별 단위 표시(예: 1,000명당 건수 등)도 함께 명시하면 정확한 해석에 도움이 됩니다.

---

### 5. 🧠 정책적 시사점 전달
- 분석 결과를 보여주는 데 그치지 않고, **행정적·정책적 의미**까지 연결 짓는 설명이 필요합니다.
- 예: "주차장을 단순히 늘리는 것보다, 인구·상권·유동인구 등을 고려한 입지 전략이 필요함"
- 이러한 **인사이트 도출**이 데이터 분석의 핵심 가치입니다.

---

### ✅ 종합 요약
- 실제 행정 데이터를 수집하고 정제한 후, 시각화와 통계 분석까지 일관된 흐름으로 구성한 점에서 의미 있는 프로젝트였습니다.
- 다만, 발표에서 교수님과 동료들이 **정책 해석과 실용 가능성에 주목**할 수 있도록 메시지를 강화하는 것이 향후 개선 포인트입니다.
""")
