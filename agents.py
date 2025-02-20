from typing import TypedDict, Annotated, Sequence
from langgraph.graph import Graph
from langchain.chat_models import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 상태 정의
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "채팅 히스토리"]
    next_step: Annotated[str, "다음 단계"]

# 프롬프트 템플릿
PLANNING_PROMPT = """데이터 분석 기획을 위한 조언자로서, 다음 사항들을 고려하여 조언해주세요:

1. 비즈니스 목표 정의
2. KPI 설정
3. 데이터 요구사항

현재 상황: {current_situation}

이전 메시지들: {chat_history}
"""

EDA_PROMPT = """데이터 탐색(EDA) 전문가로서, 다음 데이터에 대한 분석을 제공해주세요:

데이터 정보: {data_info}

분석해야 할 부분:
1. 기술 통계
2. 결측치 분석
3. 이상치 탐지
4. 분포 분석

이전 메시지들: {chat_history}
"""

SQL_PROMPT = """SQL 전문가로서, 다음 분석 요구사항에 대한 쿼리를 작성해주세요:

분석 요구사항: {analysis_requirements}

데이터베이스 스키마: {database_schema}

이전 메시지들: {chat_history}
"""

VISUALIZATION_PROMPT = """데이터 시각화 전문가로서, 다음 데이터에 대한 최적의 시각화 방법을 제안해주세요:

데이터 특성: {data_characteristics}
시각화 목적: {visualization_purpose}

이전 메시지들: {chat_history}
"""

# 에이전트 함수들
def planning_agent(state: AgentState) -> AgentState:
    """데이터 분석 기획 에이전트"""
    chat = ChatGroq(
        temperature=0,
        model_name="mixtral-8x7b-32768",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    prompt = ChatPromptTemplate.from_template(PLANNING_PROMPT)
    
    messages = state["messages"]
    response = chat.invoke(
        prompt.format_messages(
            current_situation="새로운 데이터 분석 프로젝트 시작",
            chat_history=messages
        )
    )
    
    return {
        "messages": [*messages, response],
        "next_step": "eda" if "다음 단계로" in response.content else "planning"
    }

def eda_agent(state: AgentState) -> AgentState:
    """EDA 에이전트"""
    chat = ChatGroq(
        temperature=0,
        model_name="mixtral-8x7b-32768",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    prompt = ChatPromptTemplate.from_template(EDA_PROMPT)
    
    messages = state["messages"]
    response = chat.invoke(
        prompt.format_messages(
            data_info="샘플 데이터셋",
            chat_history=messages
        )
    )
    
    return {
        "messages": [*messages, response],
        "next_step": "sql" if "SQL 분석 필요" in response.content else "eda"
    }

def sql_agent(state: AgentState) -> AgentState:
    """SQL 분석 에이전트"""
    chat = ChatGroq(
        temperature=0,
        model_name="mixtral-8x7b-32768",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    prompt = ChatPromptTemplate.from_template(SQL_PROMPT)
    
    messages = state["messages"]
    response = chat.invoke(
        prompt.format_messages(
            analysis_requirements="매출 트렌드 분석",
            database_schema="sales_table(date, amount, product_id)",
            chat_history=messages
        )
    )
    
    return {
        "messages": [*messages, response],
        "next_step": "visualization" if "시각화 필요" in response.content else "sql"
    }

def visualization_agent(state: AgentState) -> AgentState:
    """시각화 에이전트"""
    chat = ChatGroq(
        temperature=0,
        model_name="mixtral-8x7b-32768",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    prompt = ChatPromptTemplate.from_template(VISUALIZATION_PROMPT)
    
    messages = state["messages"]
    response = chat.invoke(
        prompt.format_messages(
            data_characteristics="시계열 데이터",
            visualization_purpose="트렌드 파악",
            chat_history=messages
        )
    )
    
    return {
        "messages": [*messages, response],
        "next_step": "end"
    }

# 워크플로우 정의
def create_analysis_workflow() -> Graph:
    workflow = Graph()
    
    # 노드 추가
    workflow.add_node("planning", planning_agent)
    workflow.add_node("eda", eda_agent)
    workflow.add_node("sql", sql_agent)
    workflow.add_node("visualization", visualization_agent)
    
    # 엣지 추가
    workflow.add_edge("planning", "eda")
    workflow.add_edge("eda", "sql")
    workflow.add_edge("sql", "visualization")
    
    # 조건부 엣지
    workflow.set_entry_point("planning")
    
    return workflow

# 워크플로우 실행 함수
def run_analysis_workflow(initial_message: str) -> list[BaseMessage]:
    workflow = create_analysis_workflow()
    
    initial_state = {
        "messages": [HumanMessage(content=initial_message)],
        "next_step": "planning"
    }
    
    final_state = workflow.run(initial_state)
    return final_state["messages"] 