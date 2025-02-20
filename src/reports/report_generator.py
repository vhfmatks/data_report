"""
결과 보고서 생성 모듈
"""

from datetime import datetime
from typing import Dict, Any, List, Tuple

def generate_report(
    df,
    schema: Dict,
    analysis_results: Dict,
    insights: str,
    visualizations: Dict[str, List[Tuple[str, Any]]]
) -> str:
    """결과 보고서 생성"""
    
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 보고서 내용 생성
    report_content = f"""# 데이터 분석 결과 보고서

생성일시: {report_date}

## 1. 개요
- 분석 데이터셋: {df.shape[0]:,}행 × {df.shape[1]:,}열
- 데이터 유형:
    - 수치형 변수: {len([col for col, info in schema.items() if info['data_type'] == 'numeric'])}개
    - 범주형 변수: {len([col for col, info in schema.items() if info['data_type'] == 'categorical'])}개
    - 시계열 변수: {len([col for col, info in schema.items() if info['data_type'] == 'datetime'])}개

## 2. 주요 발견사항
{insights}

## 3. 변수별 상세 분석
"""
    
    # 각 변수별 분석 결과 추가
    for col, info in schema.items():
        display_name = info.get('display_name', col)
        report_content += f"""
### {display_name}({col})
- 설명: {info['description']}
- 데이터 타입: {info['data_type']}
"""
        
        # 데이터 타입별 상세 정보 추가
        if info['data_type'] == 'numeric':
            stats = analysis_results[col]["기술통계"]
            outliers = analysis_results[col]["이상치"]
            report_content += f"""
- 기술 통계:
    - 평균: {stats['mean']:.2f}
    - 중앙값: {stats['50%']:.2f}
    - 표준편차: {stats['std']:.2f}
    - 최소값: {stats['min']:.2f}
    - 최대값: {stats['max']:.2f}
- 이상치:
    - 개수: {outliers['개수']}
    - 비율: {outliers['비율']:.2f}%
"""
            
        elif info['data_type'] == 'categorical':
            value_counts = analysis_results[col]["고유값"]
            mode = analysis_results[col]["최빈값"]
            report_content += f"""
- 고유값 수: {value_counts['개수']}
- 최빈값: {mode['값']} ({mode['비율']:.1f}%)
- 상위 5개 범주:
"""
            for category, count in list(value_counts['분포'].items())[:5]:
                report_content += f"    - {category}: {count:,}개\n"
            
        elif info['data_type'] == 'datetime':
            period = analysis_results[col]["기간"]
            report_content += f"""
- 기간 정보:
    - 시작: {period['시작']}
    - 종료: {period['종료']}
    - 총 일수: {period['일수']}일
"""
    
    # 상관관계 분석 결과 추가
    if "correlations" in analysis_results and analysis_results["correlations"]:
        report_content += "\n## 4. 상관관계 분석\n"
        for corr in analysis_results["correlations"]:
            report_content += f"- {corr['변수1']} ↔ {corr['변수2']}: {corr['상관계수']:.3f}\n"
    
    return report_content 