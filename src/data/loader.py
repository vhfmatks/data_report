"""
데이터 로딩 및 전처리 모듈
"""

import pandas as pd
from datetime import datetime
from typing import Optional, Tuple

def load_data(file) -> Tuple[pd.DataFrame, dict]:
    """
    CSV 파일을 로드하고 기본 정보를 반환
    """
    df = pd.read_csv(file)
    info = {
        "총 행 수": df.shape[0],
        "총 열 수": df.shape[1],
        "메모리 사용량": f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
    }
    return df, info

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터 전처리 수행
    """
    # 날짜형 컬럼 변환
    date_columns = df.select_dtypes(include=['datetime64']).columns
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    
    return df

def convert_to_serializable(obj):
    """
    객체를 JSON 직렬화 가능한 형태로 변환
    """
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, (pd.Int64Dtype, pd.Float64Dtype)):
        return float(obj)
    elif pd.isna(obj):
        return None
    return obj 