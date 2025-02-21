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

# 운영체제별 기본 한글 폰트 설정
if platform.system() == 'Windows':
    font_path = 'C:/Windows/Fonts/malgun.ttf'  # 윈도우의 맑은 고딕 폰트 경로
    font_name = 'Malgun Gothic'
elif platform.system() == 'Darwin':  # macOS
    font_path = '/System/Library/Fonts/AppleGothic.ttf'  # macOS의 애플고딕 폰트 경로
    font_name = 'AppleGothic'
else:  # Linux
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'  # 리눅스의 나눔고딕 폰트 경로
    font_name = 'NanumGothic'

# 폰트 추가 및 설정
font_added = fm.FontProperties(fname=font_path).get_name() if os.path.exists(font_path) else None
FONT_FAMILY = font_added if font_added else font_name

# matplotlib 한글 폰트 설정
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# matplotlib 전역 설정
mpl.rcParams['font.family'] = FONT_FAMILY
mpl.rcParams['axes.unicode_minus'] = False

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