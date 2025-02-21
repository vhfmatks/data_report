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
import numpy as np
from src.utils.helpers import filter_unwanted_languages

def generate_insights(df: pd.DataFrame, schema: Dict, analysis_results: Dict, llm) -> str:
    """LLM을 사용하여 데이터 분석 결과로부터 인사이트 도출"""
    
    try:
        # 1. 분석 컨텍스트 수집
        context = {
            "분석_목적": st.session_state.get("purpose", "목적이 설정되지 않음"),
            "보고서_주제": st.session_state.get("topic", "주제가 설정되지 않음"),
            "분석_계획": st.session_state.get("analysis_plan", "계획이 설정되지 않음"),
            "데이터_정보": {
                "전체_행수": len(df),
                "전체_컬럼수": len(df.columns),
                "변수_정보": {
                    col: {
                        "이름": info.get("display_name", col),
                        "설명": info.get("description", ""),
                        "데이터타입": info["data_type"]
                    } for col, info in schema.items()
                }
            }
        }

        # 2. 분석 결과 요약
        analysis_summary = {
            "수치형_변수": {},
            "범주형_변수": {},
            "시계열_변수": {},
            "상관관계": []
        }

        # 수치형 변수 분석 결과 수집
        for col, info in schema.items():
            if info["data_type"] == "numeric" and col in analysis_results:
                stats = analysis_results[col].get("기본통계", {})
                analysis_summary["수치형_변수"][info.get("display_name", col)] = {
                    "평균": stats.get("평균", 0),
                    "중앙값": stats.get("중앙값", 0),
                    "표준편차": stats.get("표준편차", 0),
                    "이상치_비율": analysis_results[col].get("이상치", {}).get("비율", 0)
                }

        # 범주형 변수 분석 결과 수집
        for col, info in schema.items():
            if info["data_type"] == "categorical" and col in analysis_results:
                value_counts = analysis_results[col].get("고유값", {})
                top_categories = dict(list(value_counts.get("분포", {}).items())[:3])
                analysis_summary["범주형_변수"][info.get("display_name", col)] = {
                    "고유값_수": value_counts.get("개수", 0),
                    "상위_범주": top_categories
                }

        # 시계열 변수 분석 결과 수집
        for col, info in schema.items():
            if info["data_type"] == "datetime" and col in analysis_results:
                period = analysis_results[col].get("기간", {})
                analysis_summary["시계열_변수"][info.get("display_name", col)] = {
                    "시작": period.get("시작", "N/A"),
                    "종료": period.get("종료", "N/A"),
                    "기간_일수": period.get("기간", {}).get("일수", 0)
                }

        # 상관관계 분석 결과 수집
        if "상관관계" in analysis_results:
            analysis_summary["상관관계"] = [
                {
                    "변수쌍": f"{corr['변수1']}-{corr['변수2']}",
                    "상관계수": corr["상관계수"],
                    "강도": corr.get("강도", "정보 없음")
                }
                for corr in analysis_results["상관관계"][:5]  # 상위 5개만 포함
            ]

        # 3. LLM 프롬프트 생성
        prompt = f"""당신은 한국의 데이터 분석 전문가입니다. 반드시 한국어로만 응답해야 하며, 영어는 꼭 필요한 전문용어에만 사용하세요.
절대로 중국어, 일본어, 러시아어 등 다른 언어를 사용하지 마세요.

다음 정보를 바탕으로 심층적인 인사이트를 도출해주세요.

[분석 컨텍스트]
분석 목적: {context["분석_목적"]}
보고서 주제: {context["보고서_주제"]}

[분석 계획]
{context["분석_계획"]}

[데이터 기본 정보]
- 전체 데이터: {context["데이터_정보"]["전체_행수"]:,}행 × {context["데이터_정보"]["전체_컬럼수"]}열

[분석 결과 요약]
1. 수치형 변수 분석:
{json.dumps(analysis_summary["수치형_변수"], ensure_ascii=False, indent=2)}

2. 범주형 변수 분석:
{json.dumps(analysis_summary["범주형_변수"], ensure_ascii=False, indent=2)}

3. 시계열 분석:
{json.dumps(analysis_summary["시계열_변수"], ensure_ascii=False, indent=2)}

4. 주요 상관관계:
{json.dumps(analysis_summary["상관관계"], ensure_ascii=False, indent=2)}

다음 형식으로 인사이트를 제공해주세요:

1. 핵심 발견사항 (Key Findings)
   - 데이터에서 발견된 가장 중요한 패턴이나 트렌드
   - 예상과 다른 특이점이나 이상치
   - 변수 간 중요한 관계

2. 비즈니스 인사이트 (Business Insights)
   - 발견사항이 비즈니스에 주는 의미
   - 실행 가능한 제안사항
   - 개선 기회

3. 추가 분석 필요사항 (Further Analysis)
   - 더 깊이 있는 분석이 필요한 영역
   - 추가 데이터 수집이 필요한 부분
   - 검증이 필요한 가설

각 섹션에서 구체적인 수치와 근거를 포함하여 작성해주세요.
응답은 반드시 한국어로만 작성하며, 영어는 꼭 필요한 전문용어에만 사용해주세요."""

        # 4. LLM을 통한 인사이트 생성
        response = llm.invoke(prompt)
        response_content = filter_unwanted_languages(response.content)
        
        # 5. 시각화 및 상세 분석 추가
        st.write("### 📊 주요 시각화")
        
        # 수치형 변수들의 분포 비교
        numeric_cols = [col for col, info in schema.items() if info["data_type"] == "numeric"]
        if len(numeric_cols) > 0:
            st.write("#### 수치형 변수 분포 비교")
            fig, axes = plt.subplots(1, len(numeric_cols), figsize=(5*len(numeric_cols), 4))
            if len(numeric_cols) == 1:
                axes = [axes]
            
            for ax, col in zip(axes, numeric_cols):
                sns.boxplot(data=df, y=col, ax=ax)
                ax.set_title(schema[col].get("display_name", col))
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # 상관관계 히트맵
        if len(numeric_cols) > 1:
            st.write("#### 변수 간 상관관계")
            corr_matrix = df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, 
                       annot=True, 
                       cmap='RdYlBu_r',
                       center=0,
                       vmin=-1,
                       vmax=1,
                       ax=ax)
            plt.title("상관관계 히트맵")
            st.pyplot(fig)
            plt.close()

        # 시계열 트렌드
        datetime_cols = [col for col, info in schema.items() if info["data_type"] == "datetime"]
        if datetime_cols and numeric_cols:
            st.write("#### 시계열 트렌드")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            time_col = datetime_cols[0]
            value_col = numeric_cols[0]
            
            df_grouped = df.groupby(pd.to_datetime(df[time_col]).dt.to_period('M'))[value_col].mean()
            ax.plot(range(len(df_grouped)), df_grouped.values, marker='o')
            ax.set_xticks(range(len(df_grouped)))
            ax.set_xticklabels(df_grouped.index.astype(str), rotation=45)
            ax.set_title(f"{schema[value_col].get('display_name', value_col)} 추이")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # 6. LLM 응답 표시
        st.write("### 데이터 인사이트")
        st.markdown(response_content)
        
        return response_content
        
    except Exception as e:
        error_msg = f"인사이트 생성 중 오류가 발생했습니다: {str(e)}"
        st.error(error_msg)
        return error_msg 