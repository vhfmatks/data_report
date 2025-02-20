"""
데이터 인사이트 생성 모듈
"""

import json
from typing import Dict, Any
from src.data.loader import convert_to_serializable
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

def generate_insights(df, schema: Dict, analysis_results: Dict, llm) -> str:
    """LLM을 사용하여 데이터 분석 결과로부터 인사이트 도출"""
    
    # 복합 시각화 생성
    st.write("### 📊 심층 분석 시각화")
    
    try:
        # 1. 시계열-수치 복합 분석
        datetime_cols = [col for col, info in schema.items() if info["data_type"] == "datetime" and col in df.columns]
        numeric_cols = [col for col, info in schema.items() if info["data_type"] == "numeric" and col in df.columns]
        
        if datetime_cols and numeric_cols:
            st.write("#### 1️⃣ 시계열-수치 데이터 분석")
            fig, ax1 = plt.subplots(figsize=(15, 8))
            
            # 주 Y축 - 첫 번째 수치형 변수
            time_col = datetime_cols[0]
            num_col = numeric_cols[0]
            
            color = 'tab:blue'
            ax1.set_xlabel('시간')
            ax1.set_ylabel(schema[num_col].get('display_name', num_col), color=color)
            
            # 월별 평균 계산
            monthly_data = df.groupby(pd.to_datetime(df[time_col]).dt.to_period('M'))[num_col].mean()
            ax1.plot(monthly_data.index.astype(str), monthly_data.values, color=color, marker='o')
            ax1.tick_params(axis='y', labelcolor=color)
            
            # 보조 Y축 - 두 번째 수치형 변수 (있는 경우)
            if len(numeric_cols) > 1:
                ax2 = ax1.twinx()
                num_col2 = numeric_cols[1]
                color = 'tab:orange'
                ax2.set_ylabel(schema[num_col2].get('display_name', num_col2), color=color)
                monthly_data2 = df.groupby(pd.to_datetime(df[time_col]).dt.to_period('M'))[num_col2].mean()
                ax2.plot(monthly_data2.index.astype(str), monthly_data2.values, color=color, marker='s')
                ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title("시계열-수치 데이터 추이 분석")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # 2. 범주-수치 복합 분석
        categorical_cols = [col for col, info in schema.items() if info["data_type"] == "categorical" and col in df.columns]
        
        if categorical_cols and numeric_cols:
            st.write("#### 2️⃣ 범주-수치 데이터 분석")
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            # 범주별 수치 통계
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(data=df, x=cat_col, y=num_col, ax=ax)
            plt.title(f"{schema[cat_col].get('display_name', cat_col)}별 {schema[num_col].get('display_name', num_col)} 분포")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # 3. 다변량 분석 (수치형 변수 간)
        if len(numeric_cols) >= 2:
            st.write("#### 3️⃣ 다변량 분석")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1], ax=ax)
            plt.title(f"{schema[numeric_cols[0]].get('display_name', numeric_cols[0])} vs {schema[numeric_cols[1]].get('display_name', numeric_cols[1])}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    except Exception as e:
        st.error(f"복합 시각화 생성 중 오류 발생: {str(e)}")
    
    # 데이터 기본 정보 수집
    basic_info = {
        "총 행 수": df.shape[0],
        "총 열 수": df.shape[1],
        "컬럼 목록": list(df.columns)
    }
    
    # 분석 결과를 요약하여 토큰 수를 줄임
    summarized_results = {
        "메타정보": analysis_results.get("메타정보", {}),
        "상관관계": analysis_results.get("상관관계", [])[:5],  # 상위 5개 상관관계만 포함
        "분석_계획": st.session_state.analysis_plan if "analysis_plan" in st.session_state else None
    }
    
    # 각 변수별 주요 통계만 포함
    for col, info in schema.items():
        if col in analysis_results:
            if info["data_type"] == "numeric":
                summarized_results[col] = {
                    "기본통계": {
                        k: analysis_results[col]["기본통계"][k] 
                        for k in ["평균", "중앙값", "표준편차"]
                    },
                    "이상치": {
                        "개수": analysis_results[col]["이상치"]["개수"],
                        "비율": analysis_results[col]["이상치"]["비율"]
                    }
                }
            elif info["data_type"] == "categorical":
                value_dist = analysis_results[col].get("고유값", {}).get("분포", {})
                summarized_results[col] = {
                    "고유값": {
                        "개수": analysis_results[col]["고유값"]["개수"],
                        "주요범주": dict(list(value_dist.items())[:3])  # 상위 3개 범주만 포함
                    }
                }
            elif info["data_type"] == "datetime":
                summarized_results[col] = {
                    "기간": analysis_results[col]["기간"]
                }
    
    # LLM 프롬프트 생성
    prompt = f"""데이터 분석가로서, 다음 데이터 분석 결과와 초기 분석 계획을 바탕으로 중요한 인사이트를 도출해주세요.

[초기 분석 계획]
{summarized_results["분석_계획"]}

[기본 정보]
{json.dumps(basic_info, indent=2, ensure_ascii=False)}

[주요 분석 결과]
{json.dumps(summarized_results, indent=2, ensure_ascii=False)}

다음 형식으로 응답해주세요:

1. 초기 분석 계획 대비 주요 발견사항
   - 분석 목표별 달성 여부와 핵심 발견
   - 예상했던 결과와 실제 결과의 차이점
   - 추가로 발견된 중요한 패턴이나 트렌드

2. 비즈니스 인사이트 및 제안사항
   - 데이터 기반 의사결정 포인트
   - 구체적인 개선 제안사항
   - 실행 가능한 액션 아이템

3. 추가 분석 필요 영역
   - 심층 분석이 필요한 부분
   - 추가 데이터 수집이 필요한 영역
   - 장기적인 모니터링이 필요한 지표

각 항목은 데이터 분석 결과를 기반으로 구체적인 수치와 함께 설명해주세요.
특히 초기 분석 계획에서 설정한 목표와 연계하여 인사이트를 도출해주세요.
"""
    
    response = llm.invoke(prompt)
    return response.content 