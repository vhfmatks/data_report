"""
데이터 분석 파이프라인 메인 애플리케이션
"""

import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from datetime import datetime
import matplotlib.font_manager as fm
import matplotlib as mpl
import platform

from src.config.settings import GROQ_API_KEY, MODEL_NAME, TEMPERATURE, FONT_FAMILY
from src.data.loader import load_data, preprocess_data
from src.data.schema import load_predefined_schema, suggest_schema_with_llm, parse_schema_text
from src.analysis.analyzer import analyze_data, analyze_time_series, analyze_clusters, create_prediction_model
from src.analysis.visualizer import (
    visualize_data,
    create_analysis_plan,
    plot_numeric,
    plot_categorical,
    plot_datetime,
    create_complex_visualization
)
from src.analysis.insights import generate_insights
from src.reports.report_generator import generate_report
from src.utils.helpers import display_metrics, display_visualizations, create_download_button

# 환경 변수 로드
load_dotenv()

# LLM 초기화
llm = ChatGroq(
    temperature=TEMPERATURE,
    model_name=MODEL_NAME,
    groq_api_key=GROQ_API_KEY
)

# 페이지 설정
st.set_page_config(
    page_title="데이터 분석 파이프라인",
    page_icon="📊",
    layout="wide"
)

# 한글 폰트 CSS 설정
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif;
}

.streamlit-expanderHeader {
    font-family: 'Noto Sans KR', sans-serif;
}

.stMarkdown, .stText {
    font-family: 'Noto Sans KR', sans-serif;
}

/* 차트 내 한글 폰트 설정 */
.js-plotly-plot .plotly .gtitle, 
.js-plotly-plot .plotly .xtitle,
.js-plotly-plot .plotly .ytitle,
.js-plotly-plot .plotly .legend .legendtext,
.js-plotly-plot .plotly .xtick text,
.js-plotly-plot .plotly .ytick text {
    font-family: 'Noto Sans KR', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# matplotlib 한글 폰트 설정
import matplotlib as plt

# matplotlib 폰트 설정
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['axes.unicode_minus'] = False

# 세션 상태 초기화
if 'data' not in st.session_state:
    st.session_state.data = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'schema' not in st.session_state:
    st.session_state.schema = None
if 'schema_defined' not in st.session_state:
    st.session_state.schema_defined = False
if 'analysis_plan_created' not in st.session_state:
    st.session_state.analysis_plan_created = False
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False
if 'insights_generated' not in st.session_state:
    st.session_state.insights_generated = False

# 제목
st.title("📊 데이터 분석 파이프라인")

# 데이터 업로드 섹션
st.header("1. 데이터 업로드")
uploaded_file = st.file_uploader("CSV 파일을 업로드해주세요", type=["csv"])

if uploaded_file is not None:
    try:
        # 데이터 로드
        df, info = load_data(uploaded_file)
        if st.session_state.original_data is None:
            st.session_state.original_data = df.copy()
        st.session_state.data = df
        
        # 데이터 형태 파악 섹션
        st.header("2. 데이터 형태 파악")
        
        # 기본 정보 표시
        display_metrics(info)
        
        # 데이터 미리보기
        st.subheader("데이터 미리보기")
        st.dataframe(df.head(), use_container_width=True)
        
        # 스키마 정의 섹션
        st.header("3. 스키마 정의 및 분석 목적 설정")
        
        # 사전 정의된 스키마 로드 시도
        if not st.session_state.schema_defined:
            st.subheader("3-1. 스키마 정의")
            predefined_schema = load_predefined_schema()
            
            if predefined_schema is None:
                st.error("사전 정의된 스키마를 로드하는데 실패했습니다.")
                predefined_schema = {}
            
            schema = predefined_schema.copy()
            
            # 사전 정의되지 않은 컬럼 확인
            undefined_columns = [col for col in df.columns if col not in predefined_schema]
            
            if undefined_columns:
                st.write("schema.md에 정의되지 않은 컬럼에 대한 스키마를 정의해주세요.")
                
                # 스키마 일괄 정의 옵션
                schema_input_method = st.radio(
                    "스키마 정의 방법 선택",
                    ["LLM 추천 사용", "YAML 텍스트로 정의", "수동으로 정의"]
                )
                
                if schema_input_method == "LLM 추천 사용":
                    # 정의되지 않은 컬럼만 포함하는 데이터프레임 생성
                    undefined_df = df[undefined_columns]
                    with st.spinner("LLM이 추가 컬럼의 스키마를 분석중입니다..."):
                        suggested_schema = suggest_schema_with_llm(undefined_df, llm)
                        if suggested_schema:
                            st.success("LLM이 스키마를 제안했습니다!")
                            schema.update(suggested_schema)
                            st.session_state.schema = schema
                            st.session_state.schema_defined = True
                        else:
                            st.error("LLM 스키마 제안에 실패했습니다. 다른 방법을 선택해주세요.")
                
                elif schema_input_method == "YAML 텍스트로 정의":
                    schema_text = st.text_area(
                        "추가 스키마 정의 (YAML 형식)",
                        height=400
                    )
                    
                    if st.button("추가 스키마 적용"):
                        additional_schema = parse_schema_text(schema_text)
                        if additional_schema:
                            schema.update(additional_schema)
                            st.session_state.schema = schema
                            st.session_state.schema_defined = True
                            st.success("추가 스키마가 성공적으로 적용되었습니다!")
                        else:
                            st.error("스키마 형식이 올바르지 않습니다.")
                
                elif schema_input_method == "수동으로 정의":
                    st.write("각 컬럼에 대한 스키마를 수동으로 정의해주세요.")
                    for col in undefined_columns:
                        st.write(f"#### {col} 컬럼 정의")
                        data_type = st.selectbox(
                            f"{col} 데이터 타입",
                            ["numeric", "categorical", "datetime", "text"],
                            key=f"type_{col}"
                        )
                        display_name = st.text_input(
                            f"{col} 표시 이름",
                            value=col,
                            key=f"name_{col}"
                        )
                        description = st.text_area(
                            f"{col} 설명",
                            key=f"desc_{col}"
                        )
                        
                        schema[col] = {
                            "data_type": data_type,
                            "display_name": display_name,
                            "description": description,
                            "analysis_methods": [],
                            "visualization_methods": []
                        }
                    
                    if st.button("수동 스키마 적용"):
                        st.session_state.schema = schema
                        st.session_state.schema_defined = True
                        st.success("스키마가 성공적으로 적용되었습니다!")
            else:
                st.success("모든 컬럼이 schema.md에 정의되어 있습니다.")
                st.session_state.schema = schema
                st.session_state.schema_defined = True
        
        # 분석 목적 설정
        if st.session_state.schema_defined and st.session_state.schema:
            st.subheader("3-2. 분석 목적 설정")
            
            purpose = st.text_area(
                "데이터 활용 목적을 입력해주세요",
                value="프랜차이즈 사업 내 가맹점 모집 및 사업 효율성(실적)분석",
                help="데이터를 어떤 목적으로 활용할 계획인지 구체적으로 작성해주세요."
            )
            
            topic = st.text_area(
                "보고서 작성 주제를 입력해주세요",
                value="2025년 신규 프랜차이즈 매장 모집 관련 분석",
                help="최종적으로 작성할 보고서의 주제를 작성해주세요."
            )
            
            # 세션 상태에 저장
            st.session_state.purpose = purpose
            st.session_state.topic = topic
            
            # 분석 계획 수립
            if purpose and topic:
                if not st.session_state.analysis_plan_created and st.button("🎯 분석 계획 수립"):
                    with st.spinner("AI가 분석 계획을 수립하고 있습니다..."):
                        analysis_plan = create_analysis_plan(df, st.session_state.schema, purpose, topic, llm)
                        st.session_state.analysis_plan = analysis_plan  # 분석 계획 저장
                        st.session_state.analysis_plan_created = True
                
                if st.session_state.analysis_plan_created:
                    st.write("### 분석 계획")
                    st.markdown(st.session_state.analysis_plan)
                    
                    # 분석 계획 승인 및 분석 시작 버튼
                    if not st.session_state.analysis_started and st.button("✅ 분석 계획 승인 및 분석 시작"):
                        st.session_state.analysis_started = True
                
                if st.session_state.analysis_started:
                    st.divider()
                    
                    # 데이터 분석 수행
                    with st.spinner("데이터 분석을 수행하고 있습니다..."):
                        st.header("4. 스키마 기반 데이터 분석")
                        st.write("분석 계획에 따라 단계별로 분석을 수행합니다.")
                        
                        # 데이터 전처리
                        st.subheader("4-1. 데이터 전처리")
                        st.write("##### 전처리 진행 상황")
                        with st.status("데이터 전처리 중...") as status:
                            st.write("결측치 확인 중...")
                            missing_data = df.isnull().sum()
                            if missing_data.any():
                                st.write("- 결측치가 있는 컬럼:")
                                for col, count in missing_data[missing_data > 0].items():
                                    st.write(f"  - {col}: {count}개 ({count/len(df)*100:.2f}%)")
                            else:
                                st.write("- 결측치 없음")
                            
                            st.write("데이터 타입 변환 중...")
                            df = preprocess_data(df)
                            st.write("- 날짜형 데이터 변환 완료")
                            st.write("- 수치형 데이터 변환 완료")
                            
                            status.update(label="전처리 완료!", state="complete")
                        
                        # 기본 통계 분석
                        st.subheader("4-2. 기본 통계 분석")
                        st.write("##### 분석 진행 상황")
                        with st.status("기본 통계 분석 중...") as status:
                            st.write("변수별 기술 통계량 계산 중...")
                            
                            # 수치형 변수 통계
                            numeric_cols = [col for col, info in st.session_state.schema.items() 
                                         if info["data_type"] == "numeric" and col in df.columns]
                            if numeric_cols:
                                st.write("수치형 변수 기초 통계:")
                                st.dataframe(df[numeric_cols].describe())
                            
                            # 범주형 변수 통계
                            categorical_cols = [col for col, info in st.session_state.schema.items() 
                                             if info["data_type"] == "categorical" and col in df.columns]
                            if categorical_cols:
                                st.write("범주형 변수 기초 통계:")
                                for col in categorical_cols:
                                    st.write(f"- {col} 범주 분포:")
                                    st.dataframe(df[col].value_counts().head())
                            
                            analysis_results = analyze_data(df, st.session_state.schema)
                            status.update(label="기본 통계 분석 완료!", state="complete")
                        
                        # 데이터 시각화
                        st.subheader("4-3. 데이터 시각화")
                        st.write("##### 시각화 진행 상황")
                        with st.status("데이터 시각화 중...") as status:
                            # 수치형 변수 시각화
                            if numeric_cols:
                                st.write("수치형 변수 분포 시각화:")
                                for col in numeric_cols:
                                    display_name = st.session_state.schema[col].get('display_name', col)
                                    st.write(f"- {display_name} 차트 생성 중...")
                                    plot_numeric(df, col, st.session_state.schema[col], display_name)
                            
                            # 범주형 변수 시각화
                            categorical_cols = [col for col, info in st.session_state.schema.items() 
                                             if info["data_type"] == "categorical" and col in df.columns]
                            if categorical_cols:
                                st.write("범주형 변수 분포 시각화:")
                                for col in categorical_cols:
                                    display_name = st.session_state.schema[col].get('display_name', col)
                                    st.write(f"- {display_name} 차트 생성 중...")
                                    plot_categorical(df, col, st.session_state.schema[col], display_name)
                            
                            # 시계열 변수 시각화
                            datetime_cols = [col for col, info in st.session_state.schema.items() 
                                          if info["data_type"] == "datetime" and col in df.columns]
                            if datetime_cols:
                                st.write("시계열 변수 분포 시각화:")
                                for col in datetime_cols:
                                    display_name = st.session_state.schema[col].get('display_name', col)
                                    st.write(f"- {display_name} 차트 생성 중...")
                                    plot_datetime(df, col, st.session_state.schema[col], display_name)
                            
                            # 복합 시각화 생성
                            st.write("복합 시각화 생성:")
                            create_complex_visualization(df, st.session_state.schema)
                            
                            status.update(label="시각화 완료!", state="complete")
                            st.success("모든 차트가 생성되었습니다.")
                        
                        # 상관관계 분석
                        st.subheader("4-4. 상관관계 분석")
                        st.write("##### 상관관계 분석 진행 상황")
                        with st.status("상관관계 분석 중...") as status:
                            if len(numeric_cols) > 1:
                                st.write("수치형 변수들 간의 상관관계를 분석합니다.")
                                correlation_matrix = df[numeric_cols].corr()
                                
                                # 주요 상관관계 표시
                                st.write("주요 상관관계:")
                                for i in range(len(numeric_cols)):
                                    for j in range(i+1, len(numeric_cols)):
                                        corr = correlation_matrix.iloc[i, j]
                                        if abs(corr) > 0.5:  # 유의미한 상관관계만 표시
                                            st.write(f"- {numeric_cols[i]} ↔ {numeric_cols[j]}: {corr:.3f}")
                                
                                st.write("상관관계 매트릭스:")
                                st.dataframe(correlation_matrix)
                                status.update(label="상관관계 분석 완료!", state="complete")
                            else:
                                st.info("상관관계 분석을 위한 수치형 변수가 충분하지 않습니다.")
                                status.update(label="상관관계 분석 생략", state="complete")
                        
                        # 분석 완료 메시지
                        st.success("모든 데이터 분석 단계가 완료되었습니다.")
                        
                        # 추천된 분석 방법에 따른 추가 분석 수행
                        if "recommended_analysis" in st.session_state:
                            st.subheader("4-3. AI 추천 분석")
                            with st.spinner("추천된 분석 방법을 적용하고 있습니다..."):
                                # 시계열 분석 (해당되는 경우)
                                if "time_series_analysis" in st.session_state.recommended_analysis:
                                    st.write("##### 시계열 분석")
                                    time_series_results = analyze_time_series(
                                        df,
                                        st.session_state.recommended_analysis["time_series_analysis"]
                                    )
                                    analysis_results["time_series"] = time_series_results
                                
                                # 군집 분석 (해당되는 경우)
                                if "cluster_analysis" in st.session_state.recommended_analysis:
                                    st.write("##### 군집 분석")
                                    cluster_results = analyze_clusters(
                                        df,
                                        st.session_state.recommended_analysis["cluster_analysis"]
                                    )
                                    analysis_results["clusters"] = cluster_results
                                
                                # 예측 모델링 (해당되는 경우)
                                if "predictive_modeling" in st.session_state.recommended_analysis:
                                    st.write("##### 예측 모델링")
                                    prediction_results = create_prediction_model(
                                        df,
                                        st.session_state.recommended_analysis["predictive_modeling"]
                                    )
                                    analysis_results["predictions"] = prediction_results
                        
                        # 다음 단계 안내
                        st.subheader("다음 단계")
                        st.write("분석 결과를 바탕으로 인사이트를 도출하시겠습니까?")
                        
                        # 인사이트 도출 버튼 활성화
                        if not st.session_state.get("insights_generated", False) and st.button("🔍 데이터 인사이트 도출"):
                            st.header("5. 데이터 인사이트")
                            with st.spinner("AI가 데이터를 심층 분석하고 있습니다..."):
                                insights = generate_insights(df, st.session_state.schema, analysis_results, llm)
                                st.markdown(insights)
                                st.session_state.insights = insights  # 인사이트 저장
                                st.session_state.insights_generated = True  # 인사이트 생성 완료 표시
                        
                        # 인사이트가 생성된 경우에만 보고서 생성 버튼 표시
                        if st.session_state.get("insights_generated", False):
                            st.write("### 다음 단계")
                            st.write("분석 결과와 인사이트를 바탕으로 보고서를 생성하시겠습니까?")
                            
                            if st.button("📊 결과 보고서 생성"):
                                st.header("6. 결과 보고서")
                                with st.spinner("보고서를 생성하고 있습니다..."):
                                    try:
                                        report_content = generate_report(
                                            df,
                                            st.session_state.schema,
                                            analysis_results,
                                            st.session_state.insights,
                                            None  # visualizations는 현재 사용하지 않음
                                        )
                                        
                                        # Word 문서 다운로드 버튼
                                        st.download_button(
                                            label="📥 Word 보고서 다운로드",
                                            data=report_content,
                                            file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                        )
                                        
                                        st.success("보고서가 성공적으로 생성되었습니다! 위 버튼을 클릭하여 다운로드하실 수 있습니다.")
                                    except Exception as e:
                                        st.error(f"보고서 생성 중 오류가 발생했습니다: {str(e)}")
                                        st.error("상세 오류 정보:")
                                        st.exception(e)
            else:
                st.info("분석 목적과 보고서 주제를 입력한 후 분석 계획을 수립해주세요.")
        
    except Exception as e:
        st.error(f"데이터 로드 중 오류가 발생했습니다: {str(e)}")
        st.error("상세 오류 정보:")
        st.exception(e)
else:
    st.info("CSV 파일을 업로드하면 데이터 형태를 자동으로 분석해드립니다.")

# 푸터
st.markdown("---")
st.markdown("SAAB 데이터 분석 파이프라인") 