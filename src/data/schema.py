"""
데이터 스키마 관리 모듈
"""

import yaml
import pandas as pd
from typing import Dict, Optional
from pathlib import Path
from src.config.settings import SCHEMA_PATH

# 데이터 타입 매핑
DATA_TYPES = {
    "수치형": "numeric",
    "범주형": "categorical",
    "날짜형": "datetime",
    "텍스트형": "text"
}

ANALYSIS_TYPES = {
    "numeric": {
        "분석 방법": ["기술 통계", "분포 분석", "이상치 탐지", "시계열 분석"],
        "시각화": ["히스토그램", "박스플롯", "바이올린"]
    },
    "categorical": {
        "분석 방법": ["빈도 분석", "교차 분석"],
        "시각화": ["막대 그래프", "원형 차트", "카운트플롯"]
    },
    "datetime": {
        "분석 방법": ["시계열 분해", "추세 분석", "계절성 분석"],
        "시각화": ["라인 차트", "히트맵"]
    },
    "text": {
        "분석 방법": ["단어 빈도 분석", "감성 분석"],
        "시각화": ["워드클라우드", "막대 그래프"]
    }
}

def load_predefined_schema() -> Dict:
    """schema.md 파일에서 사전 정의된 스키마 로드"""
    predefined_schema = {}
    
    try:
        with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 마크다운 테이블 파싱
        rows = content.split('\n')
        headers = []
        for row in rows:
            if '|' in row:
                cols = [col.strip() for col in row.split('|')[1:-1]]
                if '---' in row:
                    continue
                if not headers:
                    headers = cols
                    continue
                
                if len(cols) >= 4:
                    col_name = cols[1].strip()
                    display_name = cols[2].strip()
                    description = cols[3].strip()
                    
                    # 데이터 타입 추론
                    if any(keyword in col_name.upper() for keyword in ['DATE', 'YYMM']):
                        data_type = "datetime"
                        analysis_methods = ["시계열 분해", "추세 분석"]
                        viz_methods = ["라인 차트"]
                    elif any(keyword in col_name.upper() for keyword in ['CD', 'NM']):
                        data_type = "categorical"
                        analysis_methods = ["빈도 분석", "교차 분석"]
                        viz_methods = ["막대 그래프", "원형 차트"]
                    elif any(keyword in col_name.upper() for keyword in ['AMT', 'CNT']):
                        data_type = "numeric"
                        analysis_methods = ["기술 통계", "분포 분석", "이상치 탐지"]
                        viz_methods = ["히스토그램", "박스플롯"]
                    else:
                        data_type = "categorical"
                        analysis_methods = ["빈도 분석"]
                        viz_methods = ["막대 그래프"]
                    
                    predefined_schema[col_name] = {
                        "display_name": display_name,
                        "data_type": data_type,
                        "analysis_methods": analysis_methods,
                        "visualization_methods": viz_methods,
                        "description": description
                    }
        
        return predefined_schema
    except Exception as e:
        print(f"사전 정의된 스키마를 로드하는 중 오류가 발생했습니다: {str(e)}")
        return {}

def parse_schema_text(schema_text: str) -> Optional[Dict]:
    """텍스트로 입력된 스키마를 파싱"""
    try:
        return yaml.safe_load(schema_text)
    except:
        return None

def suggest_schema_with_llm(df: pd.DataFrame, llm) -> Optional[Dict]:
    """LLM을 사용하여 데이터 스키마 제안"""
    sample_data = df.head().to_json(orient='records')
    data_info = df.dtypes.to_dict()
    data_info = {k: str(v) for k, v in data_info.items()}
    
    prompt = f"""데이터셋의 스키마를 분석하고 YAML 형식으로 제안해주세요.
    각 컬럼에 대해 다음 정보를 포함해야 합니다:
    - data_type: (numeric/categorical/datetime/text)
    - analysis_methods: 적절한 분석 방법 목록
    - visualization_methods: 적절한 시각화 방법 목록
    - description: 컬럼에 대한 설명

    데이터 타입 정보:
    {data_info}

    샘플 데이터:
    {sample_data}

    응답은 다음 YAML 형식으로 작성해주세요:
    column_name:
      data_type: type
      analysis_methods:
        - method1
        - method2
      visualization_methods:
        - viz1
        - viz2
      description: description
    """
    
    response = llm.invoke(prompt)
    try:
        suggested_schema = yaml.safe_load(response.content)
        return suggested_schema
    except:
        return None 