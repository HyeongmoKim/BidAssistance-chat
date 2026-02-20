# graph.py
import operator
import os
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from usage_tool import usage_tool
from search_tool_nltojson import extract_notice_query

import json

# 환경 변수 로드
load_dotenv()

# =================================================================
# 1. State Definition
# =================================================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# =================================================================
# 2. Tool Definitions
# =================================================================
# search_tool is imported from search_tool_nltojson.py as extract_notice_query

tools = [extract_notice_query, usage_tool]

# =================================================================
# 3. Graph Nodes Setup
# =================================================================
llm = ChatOpenAI(model="gpt-5-nano",temperature=1)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(
    content="""
너는 공공입찰 플랫폼 챗봇의 중앙 판단 에이전트이다.

사용자 질문을 다음 세 가지 유형 중 하나로 분류하라.

[유형 1: 사이트 기능, 사용법 / 챗봇 질문]
- 사이트의 화면, 메뉴, 버튼, 기능 사용 방법을 묻는 질문, 챗봇의 사용에 관한 질문
- 이 경우 반드시 usage_tool을 호출하라.
- usage_tool의 반환값을 그대로 사용자에게 자연어로 전달하라.

예시:
- 로그인은 어디서 하나요?
- 공고 검색은 어떻게 하나요?
- PDF 다운로드 버튼이 안 보여요
- 챗봇에게 질문을 하고 싶어요
- 공고 지원은 어디서 확인하나요?

[유형 2: 공고 조회 / 조건 추출 질문]
- 공고 검색, 지역, 금액, 기간 등 데이터 조회를 위한 조건을 묻는 질문
- 이 경우 반드시 extract_notice_query 도구를 호출하라.
- 도구가 반환한 JSON을 가공하거나 자연어로 설명하지 말고 그대로 출력하라.

예시:
- 부산에서 이번달 공고 보여줘
- 5억 이상 공고 낙찰하한율 평균 알려줘

[유형 3: 범위 밖 질문]

이 경우 너는 절대 추가 설명을 하지 않는다.
절대 추천, 정보 제공, 잡담을 하지 않는다.
절대 두 문장 이상 출력하지 않는다.

무조건 아래 문장만 출력하고 즉시 종료하라:

"저는 공공입찰 공고 조회와 사이트 이용 안내만 지원합니다."

중요 규칙:
- 세가지 유형 중 하나만 선택하라.
- 유형 1,2만 도구를 호출한다.
- 유형 3은 도구 호출을 금지한다.
- 두 개의 도구를 동시에 호출하지 마라.
"""
)

llm_postprocess=ChatOpenAI(model="gpt-5-nano",temperature=1)

SYSTEM_PROMPT_notice_result=SystemMessage(
    content="""
    너는 공공입찰 공고 조회 결과(JSON 리스트)를 사용자에게 보기 좋게 자연어로 요약하는 도우미이다.

    입력은 항상 공고 정보가 담긴 JSON 리스트이다.

    각 JSON 객체의 필드 의미는 아래와 같다:

    - bidRealId = 공고번호
    - name = 공고명
    - region = 지역
    - organization = 기관명
    - startDate = 입찰 시작일
    - endDate = 입찰 종료일
    - openDate = 개찰일
    - basicPrice = 기초금액
    - estimatePrice = 추정가격
    - minimumBidRate = 낙찰하한율
    - bidRange = 예가범위

    규칙:
    1. 반드시 입력 JSON에 있는 정보만 사용한다.
    2. 없는 값(null)은 "정보 없음"으로 출력한다.
    3. 공고는 1), 2), 3) 번호로 구분한다.
    4. 금액은 천 단위 콤마를 넣어 "~원"으로 출력한다.

    출력 형식:

    [공고 목록]

    1) 공고명 (공고번호)
    - 지역/기관:
    - 입찰기간: 시작일 ~ 종료일
    - 개찰일:
    - 기초 금액:
    - 추정가격 :
    - 낙찰하한율 : 
    - 예가범위 :

    입력 JSON 리스트를 받으면 즉시 위 형식으로 요약하라.
"""
)
'''
SYSTEM_PROMPT_report=SystemMessage(
    content="""
    너는 공공입찰 공고 제안/투찰 분석 보고서를 사용자에게 핵심만 요약하는 도우미이다.

    입력은 텍스트 파일이다.

    섹션으로는 
    # 1. 공고요약
    # 2. 참가자격
    # 3. 낙찰가 예측
    # 4. 권고 액션
    단, 내용에 근거가 불충분하면 '가정'으로 되어있다.
    
    3줄 정도로 간단히 요약해라.
"""
)
'''

def agent_node(state: AgentState):
    """LLM이 다음 행동(답변 or 툴호출)을 결정"""   
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

def postprocess_node(state: AgentState):
    """공고 정보 조회 데이터를 llm을 통해 자연어로 풀어줌"""
    last = state["messages"][-1]

    data = json.loads(last.content)
    if data.get("type")=="notice_result":

        payload = data.get("payload", [])
        response = llm_postprocess.invoke(
            [SYSTEM_PROMPT_notice_result, HumanMessage(content=json.dumps(payload, ensure_ascii=False))]
        )
        return {"messages":[response]}
    '''
    elif data.get("type")=="report":
        payload = data.get("payload", [])
        response = llm_postprocess.invoke(
            [SYSTEM_PROMPT_report, HumanMessage(content=payload)]
        )
    else:
        return state
    '''
    return state

def human_input_node(state: AgentState):
    """
    [시작 노드] 사용자 입력 확인용 (로그 출력)
    """
    last_msg = state["messages"][-1] if state["messages"] else None
    if last_msg and isinstance(last_msg, HumanMessage):
        print(f"\n[System] 사용자 입력 수신: {last_msg.content[:20]}...")
    return None

def router_node(state: AgentState):
    """입력이 질문인지 db조회 결과인지 확인해주는 노드"""
    '''
    raw = state["messages"][-1].content
    if not raw:
        return "agent"

    s = raw.strip()

    # JSON일 가능성이 있는 경우만 파싱 시도
    if not (s.startswith("{") and s.endswith("}")):
        return "agent"

    try:
        obj = json.loads(s)
    except Exception:
        return "agent"

    if obj.get("type") in ("notice_result", "report"):
        return "postprocess"

    return "agent"
    '''

    raw = state["messages"][-1].content or ""
    text=raw.strip()
    try:
        data = json.loads(text)

        if data.get("type") in ["notice_result", "report"]:
            return "postprocess"

    except:
        pass

    return "agent"


# =================================================================
# 4. Graph Construction
# =================================================================
workflow = StateGraph(AgentState)

# 노드 등록
workflow.add_node("human_input", human_input_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("postprocess", postprocess_node)

# 엣지 연결
workflow.add_edge(START, "human_input")

workflow.add_conditional_edges("human_input",router_node,
    {
        "agent": "agent",
        "postprocess": "postprocess"
    })

workflow.add_conditional_edges("agent", tools_condition,{"tools":"tools","__end__":END})

workflow.add_edge("postprocess",END)
workflow.add_edge("tools", END)

memory = MemorySaver()

# 외부에서 import해서 쓸 수 있는 최종 앱 객체
graph_app = workflow.compile(checkpointer=memory)
