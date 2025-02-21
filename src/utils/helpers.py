"""
유틸리티 함수 모듈
"""

import streamlit as st
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import re

def filter_unwanted_languages(text: str) -> str:
    """
    <think> 태그를 필터링하는 함수
    
    Args:
        text (str): 필터링할 텍스트
        
    Returns:
        str: 필터링된 텍스트
    """
    # <think> 태그 제거
    filtered_content = []
    in_think_block = False
    
    for line in text.split('\n'):
        if '<think>' in line:
            in_think_block = True
            continue
        elif '</think>' in line:
            in_think_block = False
            continue
        if not in_think_block:
            filtered_content.append(line)
    
    return '\n'.join(filtered_content)

def display_metrics(info: Dict[str, Any]):
    """메트릭 표시"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("총 행 수", info["총 행 수"])
    with col2:
        st.metric("총 열 수", info["총 열 수"])
    with col3:
        st.metric("메모리 사용량", info["메모리 사용량"])

def display_visualizations(visualizations: Dict[str, List[Tuple[str, Any]]]):
    """시각화 결과 표시"""
    for col, plots in visualizations.items():
        if col != "complex":  # 개별 변수 시각화
            for title, fig in plots:
                st.write(f"#### {title}")
                st.pyplot(fig)
                plt.close(fig)  # 메모리 관리
        else:  # 복합 시각화
            st.header("복합 시각화")
            for title, fig in plots:
                st.write(f"#### {title}")
                st.pyplot(fig)
                plt.close(fig)  # 메모리 관리

def create_download_button(content: str, filename: str):
    """다운로드 버튼 생성"""
    st.download_button(
        label="📥 결과 보고서 다운로드",
        data=content,
        file_name=filename,
        mime="text/markdown"
    ) 