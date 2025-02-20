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

def generate_insights(df, schema: Dict, analysis_results: Dict, llm) -> str:
    """LLM을 사용하여 데이터 분석 결과로부터 인사이트 도출"""
    
    # 심층 분석 시각화
    st.write("### 📊 심층 분석 시각화")
    
    try:
        # 1. 시계열 트렌드 및 이상치 분석
        datetime_cols = [col for col, info in schema.items() if info["data_type"] == "datetime" and col in df.columns]
        numeric_cols = [col for col, info in schema.items() if info["data_type"] == "numeric" and col in df.columns]
        
        if datetime_cols and numeric_cols:
            st.write("#### 1️⃣ 시계열 트렌드 및 이상치 분석")
            time_col = datetime_cols[0]
            num_col = numeric_cols[0]
            
            # 시계열 데이터 준비
            df['Year_Month'] = pd.to_datetime(df[time_col]).dt.to_period('M')
            monthly_stats = df.groupby('Year_Month')[num_col].agg(['mean', 'std']).reset_index()
            monthly_stats['Year_Month'] = monthly_stats['Year_Month'].astype(str)
            
            # 이동평균 계산
            monthly_stats['MA3'] = monthly_stats['mean'].rolling(window=3).mean()
            
            # 신뢰구간 계산
            monthly_stats['Upper'] = monthly_stats['mean'] + 2 * monthly_stats['std']
            monthly_stats['Lower'] = monthly_stats['mean'] - 2 * monthly_stats['std']
            
            # 차트 생성
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # 실제 값
            ax.plot(monthly_stats['Year_Month'], monthly_stats['mean'], 
                   marker='o', label='실제값', color='blue', alpha=0.7)
            
            # 이동평균
            ax.plot(monthly_stats['Year_Month'], monthly_stats['MA3'], 
                   label='3개월 이동평균', color='red', linestyle='--')
            
            # 신뢰구간
            ax.fill_between(monthly_stats['Year_Month'], 
                          monthly_stats['Lower'], monthly_stats['Upper'],
                          alpha=0.2, color='gray', label='95% 신뢰구간')
            
            plt.title(f"{schema[num_col].get('display_name', num_col)} 트렌드 분석")
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # 2. 분포 및 이상치 분석
        if numeric_cols:
            st.write("#### 2️⃣ 분포 및 이상치 분석")
            
            # 주요 수치형 변수 선택
            main_numeric = numeric_cols[0]
            
            # 데이터 준비
            data = df[main_numeric].dropna()
            
            # 사분위수 계산
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 이상치 식별
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            # 바이올린 플롯 + 박스플롯 결합
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 바이올린 플롯
            sns.violinplot(data=data, ax=ax, inner=None, color='lightgray')
            
            # 박스플롯
            sns.boxplot(data=data, ax=ax, width=0.2, color='white', 
                       showfliers=False, boxprops={'zorder': 2})
            
            # 이상치 표시
            if not outliers.empty:
                ax.scatter(x=[0] * len(outliers), y=outliers, 
                         color='red', alpha=0.5, label='이상치')
            
            plt.title(f"{schema[main_numeric].get('display_name', main_numeric)} 분포 및 이상치")
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # 이상치 통계 표시
            st.write(f"- 이상치 개수: {len(outliers):,}개 ({len(outliers)/len(data)*100:.1f}%)")
            st.write(f"- 정상 범위: {lower_bound:,.0f} ~ {upper_bound:,.0f}")
        
        # 3. 범주별 성과 분석
        if numeric_cols and categorical_cols:
            st.write("#### 3️⃣ 범주별 성과 분석")
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            # 범주별 통계 계산
            cat_stats = df.groupby(cat_col)[num_col].agg([
                ('평균', 'mean'),
                ('중앙값', 'median'),
                ('표준편차', 'std'),
                ('건수', 'size')
            ]).round(2)
            
            # 상위 10개 범주 선택
            top_categories = cat_stats.nlargest(10, '평균')
            
            # 다중 막대 그래프
            fig, ax = plt.subplots(figsize=(15, 8))
            
            x = np.arange(len(top_categories))
            width = 0.35
            
            # 평균 막대
            rects1 = ax.bar(x - width/2, top_categories['평균'], width, 
                          label='평균', color='skyblue')
            
            # 중앙값 막대
            rects2 = ax.bar(x + width/2, top_categories['중앙값'], width,
                          label='중앙값', color='lightgreen')
            
            # 건수 표시 (보조 축)
            ax2 = ax.twinx()
            ax2.plot(x, top_categories['건수'], color='red', marker='o',
                    label='건수', linestyle='--')
            
            # 축 레이블 및 범례
            ax.set_ylabel(schema[num_col].get('display_name', num_col))
            ax2.set_ylabel('건수')
            ax.set_title(f"상위 10개 {schema[cat_col].get('display_name', cat_col)}별 성과 분석")
            
            # x축 레이블
            ax.set_xticks(x)
            ax.set_xticklabels(top_categories.index, rotation=45, ha='right')
            
            # 범례 통합
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
    except Exception as e:
        st.error(f"심층 분석 시각화 생성 중 오류 발생: {str(e)}")
    
    # 데이터 기본 정보 수집 (간소화)
    basic_info = {
        "총_행수": df.shape[0],
        "총_열수": df.shape[1]
    }
    
    # 분석 결과 요약 (토큰 수 최적화)
    summarized_results = {
        "메타정보": {
            "분석시작": analysis_results.get("메타정보", {}).get("분석시작", ""),
            "분석종료": analysis_results.get("메타정보", {}).get("분석종료", "")
        }
    }
    
    # 주요 상관관계만 포함
    correlations = analysis_results.get("상관관계", [])
    if correlations:
        summarized_results["주요_상관관계"] = [
            {
                "변수쌍": f"{corr['변수1']}-{corr['변수2']}",
                "계수": round(corr['상관계수'], 2)
            }
            for corr in correlations[:3]  # 상위 3개만 포함
        ]
    
    # 각 변수별 핵심 통계만 포함
    for col, info in schema.items():
        if col in analysis_results:
            if info["data_type"] == "numeric":
                stats = analysis_results[col].get("기본통계", {})
                summarized_results[col] = {
                    "평균": round(stats.get("평균", 0), 2),
                    "중앙값": round(stats.get("중앙값", 0), 2),
                    "이상치비율": round(analysis_results[col].get("이상치", {}).get("비율", 0), 2)
                }
            elif info["data_type"] == "categorical":
                value_dist = analysis_results[col].get("고유값", {}).get("분포", {})
                top_categories = dict(list(value_dist.items())[:2])  # 상위 2개만 포함
                summarized_results[col] = {
                    "주요범주": top_categories
                }
            elif info["data_type"] == "datetime":
                period = analysis_results[col].get("기간", {})
                summarized_results[col] = {
                    "기간": f"{period.get('시작', '')} ~ {period.get('종료', '')}"
                }
    
    # 분석 계획 요약
    analysis_plan_summary = ""
    if "analysis_plan" in st.session_state:
        plan_lines = st.session_state.analysis_plan.split('\n')
        analysis_plan_summary = '\n'.join([line for line in plan_lines if line.startswith(('1.', '2.', '3.', '4.', '5.'))])
    
    # LLM 프롬프트 생성 (최적화)
    prompt = f"""데이터 분석가로서, 다음 분석 결과에서 핵심 인사이트를 도출해주세요.

분석계획:
{analysis_plan_summary}

기본정보:
{json.dumps(basic_info, ensure_ascii=False)}

주요결과:
{json.dumps(summarized_results, ensure_ascii=False)}

다음 형식으로 응답해주세요:

1. 핵심 발견사항 (3가지)
- 발견 1: (구체적 수치와 함께)
- 발견 2: (구체적 수치와 함께)
- 발견 3: (구체적 수치와 함께)

2. 개선 제안 (2가지)
- 제안 1: (실행 가능한 구체적 방안)
- 제안 2: (실행 가능한 구체적 방안)

3. 추가 분석 필요사항 (1가지)
- 분석주제: (구체적인 분석 방향)
"""
    
    response = llm.invoke(prompt)
    return response.content 