"""
애플리케이션 설정 파일
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import matplotlib as mpl

# 기본 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 기본 폰트 설정 (app.py와 같은 위치의 malgun.ttf 사용)
DEFAULT_FONT_PATH = BASE_DIR / 'malgun.ttf'

# 폰트 설정
if os.path.exists(DEFAULT_FONT_PATH):
    # malgun.ttf 폰트 등록
    fm.fontManager.addfont(str(DEFAULT_FONT_PATH))
    font_name = 'Malgun Gothic'
    
    # matplotlib 폰트 설정
    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False
    
    # matplotlib 전역 설정
    mpl.rcParams['font.family'] = font_name
    mpl.rcParams['axes.unicode_minus'] = False
    
    FONT_FAMILY = font_name
else:
    print("Warning: malgun.ttf 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    FONT_FAMILY = 'sans-serif'

# Seaborn 스타일 설정
SEABORN_STYLE = "whitegrid"

# 시각화 설정
FIGURE_DEFAULT_SIZE = (10, 6)
ITEMS_PER_PAGE = 50
TOP_N_CATEGORIES = 7

# API 설정
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "deepseek-r1-distill-llama-70b"
TEMPERATURE = 0.5

# 파일 경로
SCHEMA_PATH = BASE_DIR / "schema.md" 