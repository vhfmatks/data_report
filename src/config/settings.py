"""
애플리케이션 설정 파일
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# 기본 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 운영체제별 기본 한글 폰트 설정
if platform.system() == 'Windows':
    FONT_FAMILY = 'Malgun Gothic'
elif platform.system() == 'Darwin':  # macOS
    FONT_FAMILY = 'AppleGothic'
else:  # Linux
    FONT_FAMILY = 'NanumGothic'

# Seaborn 스타일 설정
SEABORN_STYLE = "whitegrid"

# matplotlib 한글 폰트 설정
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

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