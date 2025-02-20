# 데이터 분석 파이프라인

Streamlit과 LangGraph를 활용한 데이터 분석 파이프라인 프로젝트입니다.

## 주요 기능

1. 데이터분석 기획

   - 비즈니스 목표 정의
   - KPI 설정
   - 데이터 요구사항 정의

2. 데이터 탐색(EDA)

   - 데이터 기초 통계 분석
   - 결측치 분석
   - 이상치 탐지
   - 분포 분석

3. 데이터분석(SQL)

   - SQL 쿼리 작성 및 실행
   - 데이터 추출 및 가공

4. 차트 및 인사이트
   - 다양한 시각화 도구 제공
   - 인사이트 도출
   - 개선 제안사항 도출

## 설치 방법

1. 가상환경 생성 및 활성화

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

2. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

3. 환경 변수 설정
   `.env` 파일을 생성하고 다음 내용을 추가:

```
GROQ_API_KEY=your_groq_api_key_here
```

## 실행 방법

```bash
streamlit run app.py
```

## 프로젝트 구조

```
.
├── README.md
├── requirements.txt
├── .env
├── app.py
└── agents.py
```

## 기술 스택

- Python 3.8+
- Streamlit
- LangGraph
- LangChain
- Plotly
- SQLAlchemy
- Groq (Mixtral-8x7b-32768)

## 라이선스

MIT License
