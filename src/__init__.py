"""
데이터 분석 파이프라인 패키지
"""

from src.config import settings
from src.data import loader, schema
from src.analysis import analyzer, visualizer, insights
from src.reports import report_generator
from src.utils import helpers

__version__ = "1.0.0" 