"""
애플리케이션 설정 파일
"""

import os
from pathlib import Path

# 기본 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Seaborn 스타일 설정
SEABORN_STYLE = "whitegrid"
FONT_FAMILY = "Malgun Gothic"

# 시각화 설정
FIGURE_DEFAULT_SIZE = (10, 6)
ITEMS_PER_PAGE = 50
TOP_N_CATEGORIES = 7

# API 설정
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "mixtral-8x7b-32768"
TEMPERATURE = 0

# 파일 경로
SCHEMA_PATH = BASE_DIR / "schema.md" 