import streamlit as st
import time

# 여기부터 너가 복사한 전체 분석 대시보드 코드로 교체
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

