"""
데이터 시각화 모듈
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from typing import Dict, List, Tuple, Any, Optional
from src.config.settings import (
    SEABORN_STYLE,
    FONT_FAMILY,
    FIGURE_DEFAULT_SIZE,
    ITEMS_PER_PAGE,
    TOP_N_CATEGORIES
)
import json

# Seaborn 스타일 설정
sns.set_theme(style=SEABORN_STYLE)
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['axes.unicode_minus'] = False

def create_figure(figsize: Tuple[int, int] = FIGURE_DEFAULT_SIZE):
    """일관된 크기와 스타일의 figure 생성"""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def display_plot(title: str, fig: Any, col_name: str = None):
    """차트 표시"""
    if col_name:
        st.subheader(f"📊 {col_name} - {title}")
    else:
        st.subheader(f"📊 {title}")
    st.pyplot(fig)
    plt.close(fig)

def plot_numeric(df: pd.DataFrame, col: str, info: Dict, display_title: str):
    """수치형 데이터 시각화"""
    try:
        # 결측치 제거
        series = df[col].dropna()
        if series.empty:
            st.warning(f"{display_title}: 시각화할 데이터가 없습니다.")
            return
        
        st.write(f"### 📈 {display_title} 분석")
        
        # 히스토그램
        if "히스토그램" in info["visualization_methods"]:
            st.write(f"- {display_title} 히스토그램 생성 중...")
            fig, ax = create_figure()
            sns.histplot(data=series, bins=30, kde=True, ax=ax)
            ax.set_title("값 분포")
            display_plot("히스토그램", fig, display_title)
        
        # 박스플롯
        if "박스플롯" in info["visualization_methods"]:
            st.write(f"- {display_title} 박스플롯 생성 중...")
            fig, ax = create_figure()
            sns.boxplot(data=series, ax=ax)
            ax.set_title("박스플롯")
            display_plot("박스플롯", fig, display_title)
        
        # 바이올린 플롯
        if "바이올린" in info["visualization_methods"]:
            st.write(f"- {display_title} 바이올린 플롯 생성 중...")
            fig, ax = create_figure()
            sns.violinplot(data=series, ax=ax)
            ax.set_title("바이올린 플롯")
            display_plot("바이올린 플롯", fig, display_title)
            
    except Exception as e:
        st.error(f"{display_title} 시각화 중 오류 발생: {str(e)}")

def plot_categorical(df: pd.DataFrame, col: str, info: Dict, display_title: str):
    """범주형 데이터 시각화"""
    try:
        # 결측치 제거
        series = df[col].dropna()
        if series.empty:
            st.warning(f"{display_title}: 시각화할 데이터가 없습니다.")
            return
        
        st.write(f"### 📊 {display_title} 분석")
        
        # 코드-코드명 매핑 생성
        code_map = {}
        if "code_values" in info:
            code_map = {str(code): name for code, name in info["code_values"].items()}
        
        # 값 카운트 계산
        value_counts = series.value_counts()
        
        # 코드명으로 변환된 Series 생성
        value_counts_mapped = pd.Series({
            code_map.get(str(code), str(code)): count 
            for code, count in value_counts.items()
        })
        
        # 막대 그래프
        if "막대 그래프" in info["visualization_methods"]:
            st.write(f"- {display_title} 막대 그래프 생성 중...")
            # 상위 10개 범주만 선택
            top_n = value_counts_mapped.head(10)
            
            fig, ax = create_figure(figsize=(12, 6))
            bars = sns.barplot(x=top_n.index, y=top_n.values, ax=ax)
            ax.set_title("상위 10개 범주")
            
            # x축 레이블 회전 및 정렬
            plt.xticks(rotation=45, ha='right')
            
            # 막대 위에 값 표시
            for i, v in enumerate(top_n.values):
                ax.text(i, v, f'{int(v):,}', ha='center', va='bottom')
            
            plt.tight_layout()
            display_plot("범주별 빈도", fig, display_title)
        
        # 원형 차트
        if "원형 차트" in info["visualization_methods"]:
            st.write(f"- {display_title} 원형 차트 생성 중...")
            fig, ax = create_figure(figsize=(10, 8))
            
            # 상위 N개 범주 선택
            if len(value_counts_mapped) > TOP_N_CATEGORIES:
                top_values = value_counts_mapped[:TOP_N_CATEGORIES]
                others_sum = value_counts_mapped[TOP_N_CATEGORIES:].sum()
                
                values = pd.concat([top_values, pd.Series({'기타': others_sum})])
                labels = [f'{label}\n({int(value):,}명, {value/values.sum()*100:.1f}%)' 
                         for label, value in values.items()]
                
                plt.pie(values, labels=labels)
            else:
                labels = [f'{label}\n({int(value):,}명, {value/value_counts_mapped.sum()*100:.1f}%)' 
                         for label, value in value_counts_mapped.items()]
                plt.pie(value_counts_mapped, labels=labels)
            
            ax.set_title("범주별 비율")
            plt.tight_layout()
            display_plot("원형 차트", fig, display_title)
            
            # 빈도표 표시
            st.write("#### 📋 상세 빈도표")
            freq_df = pd.DataFrame({
                '범주': value_counts_mapped.index,
                '빈도': value_counts_mapped.values,
                '비율(%)': (value_counts_mapped.values / value_counts_mapped.sum() * 100).round(2)
            })
            freq_df['빈도'] = freq_df['빈도'].apply(lambda x: f'{int(x):,}')
            freq_df['비율(%)'] = freq_df['비율(%)'].apply(lambda x: f'{x:.2f}%')
            st.dataframe(freq_df, use_container_width=True)
            
    except Exception as e:
        st.error(f"{display_title} 시각화 중 오류 발생: {str(e)}")

def plot_datetime(df: pd.DataFrame, col: str, info: Dict, display_title: str):
    """시계열 데이터 시각화"""
    try:
        # datetime 형식으로 변환 및 결측치 제거
        date_series = pd.to_datetime(df[col], errors='coerce').dropna()
        if date_series.empty:
            st.warning(f"{display_title}: 시각화할 데이터가 없습니다.")
            return
        
        st.write(f"### 📅 {display_title} 분석")
        
        # 라인 차트
        if "라인 차트" in info["visualization_methods"]:
            st.write(f"- {display_title} 시계열 추이 차트 생성 중...")
            # 일별 집계
            daily_counts = date_series.value_counts().sort_index()
            
            fig, ax = create_figure(figsize=(12, 6))
            ax.plot(daily_counts.index, daily_counts.values)
            ax.set_title("시계열 추이")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            display_plot("시계열 추이", fig, display_title)
        
        # 월별 분포
        st.write(f"- {display_title} 월별 분포 차트 생성 중...")
        month_counts = date_series.dt.month.value_counts().sort_index()
        month_labels = ['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월']
        month_values = [month_counts.get(i, 0) for i in range(1, 13)]
        
        fig, ax = create_figure()
        sns.barplot(x=month_labels, y=month_values, ax=ax)
        ax.set_title("월별 분포")
        plt.xticks(rotation=45)
        plt.tight_layout()
        display_plot("월별 분포", fig, display_title)
        
        # 요일별 분포
        st.write(f"- {display_title} 요일별 분포 차트 생성 중...")
        weekday_counts = date_series.dt.dayofweek.value_counts().sort_index()
        weekday_labels = ['월', '화', '수', '목', '금', '토', '일']
        weekday_values = [weekday_counts.get(i, 0) for i in range(7)]
        
        fig, ax = create_figure()
        sns.barplot(x=weekday_labels, y=weekday_values, ax=ax)
        ax.set_title("요일별 분포")
        plt.tight_layout()
        display_plot("요일별 분포", fig, display_title)
            
    except Exception as e:
        st.error(f"{display_title} 시각화 중 오류 발생: {str(e)}")

def create_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str]):
    """상관관계 히트맵 생성"""
    try:
        if len(numeric_cols) < 2:
            return
        
        # 결측치 제거
        corr_df = df[numeric_cols].dropna()
        if corr_df.empty:
            st.warning("상관관계 분석: 시각화할 데이터가 없습니다.")
            return
        
        st.write("### 📊 변수 간 상관관계 분석")
        st.write("- 상관관계 히트맵 생성 중...")
        
        # 상관관계 계산
        corr_matrix = corr_df.corr().round(3)
        
        # 히트맵 생성
        fig, ax = create_figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='RdYlBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax
        )
        ax.set_title("상관관계 히트맵")
        plt.tight_layout()
        
        display_plot("상관관계 히트맵", fig)
        
    except Exception as e:
        st.error(f"상관관계 히트맵 생성 중 오류 발생: {str(e)}")

def create_complex_visualization(df: pd.DataFrame, schema: Dict):
    """복합 시각화 생성"""
    try:
        st.write("### 📈 복합 시각화")
        
        # 1. 수치형 변수들의 상관관계 및 분포 매트릭스
        numeric_cols = [col for col, info in schema.items() 
                       if info["data_type"] == "numeric" and col in df.columns]
        if len(numeric_cols) > 1:
            st.write("#### 수치형 변수 간 관계")
            
            # 결측치 제거
            numeric_df = df[numeric_cols].dropna()
            
            # 변수명을 display_name으로 변경
            display_names = {col: schema[col].get('display_name', col) for col in numeric_cols}
            numeric_df = numeric_df.rename(columns=display_names)
            display_numeric_cols = [display_names[col] for col in numeric_cols]
            
            try:
                # 산점도 행렬 생성
                fig = plt.figure(figsize=(12, 8))
                
                # 변수 개수에 따라 subplot 크기 조정
                n_vars = len(display_numeric_cols)
                for i in range(n_vars):
                    for j in range(n_vars):
                        plt.subplot(n_vars, n_vars, i * n_vars + j + 1)
                        
                        if i == j:  # 대각선: 히스토그램
                            plt.hist(numeric_df[display_numeric_cols[i]], bins=20, density=True, alpha=0.7)
                            sns.kdeplot(data=numeric_df[display_numeric_cols[i]], color='red', linewidth=1)
                        else:  # 비대각선: 산점도
                            plt.scatter(numeric_df[display_numeric_cols[j]], 
                                      numeric_df[display_numeric_cols[i]], 
                                      alpha=0.5)
                            
                        if i == n_vars-1:  # x축 레이블은 마지막 행에만
                            plt.xlabel(display_numeric_cols[j], fontsize=8)
                        if j == 0:  # y축 레이블은 첫 번째 열에만
                            plt.ylabel(display_numeric_cols[i], fontsize=8)
                            
                        plt.xticks(fontsize=6)
                        plt.yticks(fontsize=6)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # 상관관계 히트맵 추가
                st.write("#### 상관관계 히트맵")
                corr_matrix = numeric_df.corr()
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, 
                           annot=True, 
                           cmap='RdYlBu_r',
                           center=0,
                           vmin=-1,
                           vmax=1,
                           fmt='.2f',
                           ax=ax)
                plt.title("변수 간 상관관계")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"수치형 변수 시각화 중 오류 발생: {str(e)}")
        
        # 2. 범주형 변수들의 다중 막대 그래프
        categorical_cols = [col for col, info in schema.items() 
                          if info["data_type"] == "categorical" and col in df.columns]
        if len(categorical_cols) > 1:
            st.write("#### 범주형 변수 분포 비교")
            fig, axes = plt.subplots(1, len(categorical_cols), figsize=(15, 6))
            if len(categorical_cols) == 1:
                axes = [axes]
            
            for ax, col in zip(axes, categorical_cols):
                # 코드-코드명 매핑 적용
                code_map = {}
                if "code_values" in schema[col]:
                    code_map = {str(code): name for code, name in schema[col]["code_values"].items()}
                
                value_counts = df[col].value_counts().head(10)
                value_counts_mapped = pd.Series({
                    code_map.get(str(idx), str(idx)): val 
                    for idx, val in value_counts.items()
                })
                
                sns.barplot(x=value_counts_mapped.values, y=value_counts_mapped.index, ax=ax, orient='h')
                ax.set_title(f"{schema[col].get('display_name', col)}\n상위 10개 범주")
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # 3. 시계열 데이터의 복합 트렌드 분석
        datetime_cols = [col for col, info in schema.items() 
                        if info["data_type"] == "datetime" and col in df.columns]
        if datetime_cols and numeric_cols:
            st.write("#### 시계열 복합 트렌드 분석")
            fig, ax1 = plt.subplots(figsize=(15, 8))
            
            # 주 Y축 - 첫 번째 수치형 변수
            color = 'tab:blue'
            ax1.set_xlabel('시간')
            ax1.set_ylabel(schema[numeric_cols[0]].get('display_name', numeric_cols[0]), color=color)
            time_data = df.groupby(datetime_cols[0])[numeric_cols[0]].mean()
            ax1.plot(time_data.index, time_data.values, color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            if len(numeric_cols) > 1:
                # 보조 Y축 - 두 번째 수치형 변수
                ax2 = ax1.twinx()
                color = 'tab:orange'
                ax2.set_ylabel(schema[numeric_cols[1]].get('display_name', numeric_cols[1]), color=color)
                time_data2 = df.groupby(datetime_cols[0])[numeric_cols[1]].mean()
                ax2.plot(time_data2.index, time_data2.values, color=color)
                ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title("시계열 복합 트렌드 분석")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
    except Exception as e:
        st.error(f"복합 시각화 생성 중 오류 발생: {str(e)}")

def create_analysis_plan(df: pd.DataFrame, schema: Dict, purpose: str, topic: str, llm) -> str:
    """
    LLM을 활용하여 데이터 분석 계획 수립
    
    Args:
        df (pd.DataFrame): 분석할 데이터프레임
        schema (Dict): 데이터 스키마
        purpose (str): 데이터 활용 목적
        topic (str): 보고서 작성 주제
        llm: LLM 인스턴스
        
    Returns:
        str: 분석 계획
    """
    try:
        # 데이터 기본 정보 수집
        data_info = {
            "행 수": len(df),
            "열 수": len(df.columns),
            "변수 정보": {
                col: {
                    "이름": info.get("display_name", col),
                    "설명": info.get("description", ""),
                    "데이터타입": info["data_type"],
                    "분석방법": info.get("analysis_methods", []),
                    "시각화방법": info.get("visualization_methods", [])
                } for col, info in schema.items() if col in df.columns
            }
        }
        
        # LLM 프롬프트 생성
        prompt = f"""무조건 한국어로 대답하세요.

데이터 분석 전문가로서, 다음 정보를 바탕으로 체계적인 분석 계획을 수립해주세요.

[데이터 활용 목적]
{purpose}

[보고서 작성 주제]
{topic}

[데이터 기본 정보]
- 전체 행 수: {data_info['행 수']:,}개
- 전체 열 수: {data_info['열 수']}개

[사용 가능한 변수 목록]
{json.dumps(data_info['변수 정보'], indent=2, ensure_ascii=False)}

다음 형식으로 분석 계획을 작성해주세요:

1. 분석 목표
   - 핵심 분석 목표
   - 세부 분석 목표

2. 분석 방법
   - 기초 통계 분석
   - 심화 통계 분석
   - 시각화 계획

3. 예상되는 인사이트
   - 도출 가능한 인사이트
   - 활용 방안

4. 분석 단계
   - 1단계: 데이터 탐색
   - 2단계: 기초 분석
   - 3단계: 심화 분석
   - 4단계: 결과 정리

5. 유의사항
   - 데이터 처리 시 주의점
   - 해석 시 고려사항
"""
        
        # LLM을 통한 분석 계획 생성
        response = llm.invoke(prompt)
        response_content = response.content
        
        # <think> 태그 필터링
        if "<think>" in response_content:
            filtered_content = []
            in_think_block = False
            
            for line in response_content.split('\n'):
                if "<think>" in line:
                    in_think_block = True
                    continue
                elif "</think>" in line:
                    in_think_block = False
                    continue
                
                if not in_think_block:
                    filtered_content.append(line)
            
            response_content = '\n'.join(filtered_content)
        
        return response_content
        
    except Exception as e:
        return f"분석 계획 수립 중 오류가 발생했습니다: {str(e)}"

def visualize_data(df: pd.DataFrame, schema: Dict, llm=None):
    """전체 데이터 시각화 수행"""
    try:
        st.title("📊 데이터 분석 및 시각화")
        
        # 데이터 활용 목적 및 보고서 주제 입력
        st.write("## 📝 분석 목적 설정")
        purpose = st.text_area(
            "데이터 활용 목적을 입력해주세요",
            placeholder="예: 고객 세그먼트별 구매 패턴 분석을 통한 마케팅 전략 수립",
            help="데이터를 어떤 목적으로 활용할 계획인지 구체적으로 작성해주세요."
        )
        
        topic = st.text_area(
            "보고서 작성 주제를 입력해주세요",
            placeholder="예: 2023년 고객 구매 행동 분석 보고서",
            help="최종적으로 작성할 보고서의 주제를 작성해주세요."
        )
        
        # 분석 계획 수립
        if purpose and topic and llm:
            if st.button("🎯 분석 계획 수립"):
                with st.spinner("AI가 분석 계획을 수립하고 있습니다..."):
                    analysis_plan = create_analysis_plan(df, schema, purpose, topic, llm)
                    st.write("## 📋 분석 계획")
                    st.markdown(analysis_plan)
                    
                    # 분석 계획 승인 및 분석 시작 버튼
                    if st.button("✅ 분석 계획 승인 및 분석 시작"):
                        st.divider()
                        
                        # LLM을 통한 맞춤형 분석 방법 추천
                        st.write("## 🤖 AI 분석 방법 추천")
                        with st.spinner("AI가 분석 방법을 추천하고 있습니다..."):
                            # 분석 추천을 위한 프롬프트 생성
                            recommendation_prompt = f"""무조건 한국어로 대답하세요.

데이터 분석 전문가로서, 다음 정보를 바탕으로 구체적인 분석 방법과 시각화 방법을 추천해주세요.

[분석 목적]
{purpose}

[분석 계획]
{analysis_plan}

[데이터 정보]
{json.dumps({col: {"type": info["data_type"], "name": info.get("display_name", col)} 
             for col, info in schema.items()}, indent=2, ensure_ascii=False)}

다음 형식으로 추천해주세요:

1. 단계별 분석 방법
   - 각 단계에서 필요한 구체적인 분석 방법
   - 사용할 통계적 기법
   - 예상되는 결과물

2. 시각화 추천
   - 각 분석에 적합한 차트 유형
   - 복합 시각화 방안
   - 인터랙티브 요소 추가 방안

3. 고급 분석 기법
   - 시계열 분석 방법 (해당되는 경우)
   - 군집 분석 방법 (해당되는 경우)
   - 예측 모델링 방안 (해당되는 경우)

각 추천에 대해 왜 이 방법이 적합한지 근거를 함께 제시해주세요."""

                            recommendations = llm.invoke(recommendation_prompt)
                            recommendations_content = recommendations.content
                            
                            # <think> 태그 필터링
                            if "<think>" in recommendations_content:
                                filtered_content = []
                                in_think_block = False
                                
                                for line in recommendations_content.split('\n'):
                                    if "<think>" in line:
                                        in_think_block = True
                                        continue
                                    elif "</think>" in line:
                                        in_think_block = False
                                        continue
                                    
                                    if not in_think_block:
                                        filtered_content.append(line)
                                
                                recommendations_content = '\n'.join(filtered_content)
                            
                            st.markdown(recommendations_content)
                            
                            # 추천된 분석 방법 승인
                            if st.button("✨ 추천 분석 방법 적용"):
                                st.divider()
                        
                        # 기본 정보 표시
                        st.write("## 📋 데이터 기본 정보")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("전체 행 수", f"{len(df):,}")
                        with col2:
                            st.metric("전체 열 수", f"{len(df.columns):,}")
                        with col3:
                            st.metric("메모리 사용량", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
                        
                        # 변수별 시각화
                        st.write("## 📈 변수별 분석")
                        
                        for col, info in schema.items():
                            if col not in df.columns:
                                continue
                                
                            display_title = info.get('display_name', col)
                            
                            if info["data_type"] == "numeric":
                                plot_numeric(df, col, info, display_title)
                            elif info["data_type"] == "categorical":
                                plot_categorical(df, col, info, display_title)
                            elif info["data_type"] == "datetime":
                                plot_datetime(df, col, info, display_title)
                        
                        # 상관관계 분석
                        st.write("## 📊 상관관계 분석")
                        numeric_cols = [col for col, info in schema.items() 
                                    if info["data_type"] == "numeric" and col in df.columns]
                        create_correlation_heatmap(df, numeric_cols)
                        
                        # 복합 시각화
                        st.write("## 📈 복합 시각화")
                        create_complex_visualization(df, schema)
        else:
            st.info("분석 목적과 보고서 주제를 입력한 후 분석 계획을 수립해주세요.")
            
    except Exception as e:
        st.error(f"데이터 시각화 중 오류 발생: {str(e)}")

def plot_time_series_analysis(results, date_col, value_col):
    """시계열 분석 결과를 시각화합니다."""
    st.write("#### 시계열 분석 결과")
    
    # 시계열 분해 시각화
    if "decomposition" in results:
        fig = plt.figure(figsize=(12, 10))
        plt.subplot(411)
        plt.plot(results["decomposition"]["trend"], label="추세")
        plt.title("시계열 분해 - 추세")
        plt.legend()
        
        plt.subplot(412)
        plt.plot(results["decomposition"]["seasonal"], label="계절성")
        plt.title("시계열 분해 - 계절성")
        plt.legend()
        
        plt.subplot(413)
        plt.plot(results["decomposition"]["resid"], label="잔차")
        plt.title("시계열 분해 - 잔차")
        plt.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # 추세 분석 결과
    if "trend_analysis" in results:
        st.write("##### 추세 분석")
        trend = results["trend_analysis"]
        st.write(f"- 기울기: {trend['slope']:.4f}")
        st.write(f"- 절편: {trend['intercept']:.4f}")
        st.write(f"- R² 값: {trend['r_squared']:.4f}")
        st.write(f"- p-value: {trend['p_value']:.4f}")
    
    # 계절성 분석 결과
    if "seasonality" in results:
        st.write("##### 계절성 분석")
        fig = plt.figure(figsize=(10, 6))
        plt.plot(results["seasonality"]["acf_values"])
        plt.title("자기상관함수 (ACF)")
        plt.axhline(y=0, linestyle="--", color="gray")
        plt.axhline(y=0.2, linestyle="--", color="red")
        plt.axhline(y=-0.2, linestyle="--", color="red")
        st.pyplot(fig)

def plot_cluster_analysis(results, df, features):
    """군집 분석 결과를 시각화합니다."""
    st.write("#### 군집 분석 결과")
    
    if "error" in results:
        st.error(f"군집 분석 중 오류 발생: {results['error']}")
        return
    
    # 군집별 통계
    st.write("##### 군집별 특성")
    for i, stats in enumerate(results["cluster_stats"]):
        st.write(f"군집 {i+1}:")
        st.write(f"- 크기: {stats['size']} ({stats['percentage']:.1f}%)")
        st.write("- 특성별 통계:")
        for feature, values in stats["features"].items():
            st.write(f"  - {feature}:")
            st.write(f"    - 평균: {values['mean']:.2f}")
            st.write(f"    - 표준편차: {values['std']:.2f}")
    
    # 2D 시각화 (첫 두 특성 사용)
    if len(features) >= 2:
        from sklearn.preprocessing import StandardScaler
        
        X = df[features[:2]]
        X_scaled = StandardScaler().fit_transform(X)
        
        fig = plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                            c=results["labels"], cmap="viridis")
        plt.title("군집 분포 (2D)")
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.colorbar(scatter)
        st.pyplot(fig)

def plot_prediction_results(results):
    """예측 모델 결과를 시각화합니다."""
    st.write("#### 예측 모델 결과")
    
    if "error" in results:
        st.error(f"예측 모델링 중 오류 발생: {results['error']}")
        return
    
    # 모델 성능 지표
    st.write("##### 모델 성능")
    metrics = results["metrics"]
    st.write(f"- RMSE: {metrics['rmse']:.2f}")
    st.write(f"- R² Score: {metrics['r2']:.4f}")
    
    # 특성 중요도
    st.write("##### 특성 중요도")
    importance = results["feature_importance"]
    fig = plt.figure(figsize=(10, 6))
    plt.bar(importance.keys(), importance.values())
    plt.title("특성 중요도")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # 실제값 vs 예측값
    st.write("##### 실제값 vs 예측값")
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(results["predictions"]["actual"], 
               results["predictions"]["predicted"], 
               alpha=0.5)
    plt.plot([min(results["predictions"]["actual"]), 
              max(results["predictions"]["actual"])],
             [min(results["predictions"]["actual"]), 
              max(results["predictions"]["actual"])], 
             'r--')
    plt.xlabel("실제값")
    plt.ylabel("예측값")
    plt.title("실제값 vs 예측값 비교")
    st.pyplot(fig) 