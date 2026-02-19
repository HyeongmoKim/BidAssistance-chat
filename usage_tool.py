# usage_tool.py : 실사용 toolnode
# python -m pip install langchain-openai langchain faiss-cpu langchain-community langchain-text-splitters unstructured[xlsx]
# pip install langchainhub faiss-cpu langchain-openai langchain langgraph typing typing_extensions langchain_core langchain-community
# pip install -U langchain-openai langchain-community langchain-text-splitters
# pip install -q pillow unstructured msoffcrypto-tool

# -*- coding: utf-8 -*-
from langchain.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List, Dict
import os

from vector_db_embedding import *
# =========================
# 기본 설정
# =========================
# 사용할 API 키 불러오기

from dotenv import load_dotenv
load_dotenv()

'''
def load_api_keys(filepath="api_key.txt"): 
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()
        
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_api_keys(os.path.join(BASE_DIR, "usage_api.txt"))   # API 키 로드 및 환경변수 설정
'''

# EMBEDDING_MODEL = "text-embedding-3-small"     # Embedding 모델(text-embedding-3-small) 설정
LLM_MODEL = "gpt-5-nano"                       # LLM 모델(gpt-5-nano) 설정

# embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)  # Embedding 모델 초기화
llm = ChatOpenAI(model=LLM_MODEL, temperature=1)    # LLM 모델 초기화

# =========================
# Tool: usage_tool (사용자 질문 처리 -> 답변값(response) 반환)
# =========================
@tool
def usage_tool(query: str):
    """
    이 도구는 사용자가 묻는 질문이
    사이트의 기능, 화면 사용법, 메뉴 설명, 서비스 이용 방법과 관련된 경우에만 사용한다.

    공고 조회, 조건 검색, 데이터 조회 목적의 질문에는 절대 사용하지 않는다.

    이 도구는 다음과 같은 질문에 사용된다:
    - 사이트에서 특정 기능을 어떻게 사용하는지 묻는 경우
    - 화면 구성이나 버튼의 역할을 설명해 달라는 질문
    - 서비스 이용 흐름이나 절차를 설명해 달라는 질문

    입력값:
    - query: 사용자의 원본 질문

    출력:
    - 사용자에게 친절한 존댓말 자연어 설명 문자열
    """

    # 1️⃣ FAISS 로드 (완전 분리)
    image_faiss = load_image_faiss(IMAGE_FAISS_DIR)
    api_faiss = load_api_faiss(API_FAISS_DIR)
    text_faiss = load_text_faiss(TEXT_FAISS_DIR)
    # 2️⃣ 벡터 유사도 검색
    image_docs = search_image_context(image_faiss, query)
    api_docs = search_api_context(api_faiss, query)
    text_docs = search_text_context(text_faiss,query)
    # 3️⃣ 컨텍스트 분리 정리
    contexts = build_context(image_docs, api_docs, text_docs)
    image_context = contexts["image"]
    api_context = contexts["api"]
    text_context = contexts["text"]


    # 4️⃣ LLM 프롬프트
    prompt = f"""
        Role 설정 : 
        너는 소형·중형 건설사를 대상으로 운영하는 "나라장터 기반 조달·입찰 인텔리전스 플랫폼"사이트의 
        편리한 이용을 도와주는 챗봇이다. 
        아래의 3가지 문서 "웹페이지 스크린샷 기반 정보", "API 정의서 엑셀 기반 정보", "홈페이지 사용 설명서"와
        "요구사항"의 내용을 참고하여 사용자의 질문에 대한 답변을 생성하라.

        요구사항:
        - API 정의서의 내용을 가져와서 설명하지 말 것, API 정의서는 웹사이트의 기능이 어떻게 동작하는지에 대한 참고 자료로 사용하라.
        - 소형·중형 건설사에서 종사하는 사람이 이해하기 쉽게 설명한다.
        - 300 ~ 700자 이내의 범위로 답변을 생성하라.
        - "상황별 팁", "시나리오", "팁"의 정보는 제공하지 않는다.
        - 답변 시작문구의 내용 소개, 끝문구의 추천 내용은 간단하고 짧게 작성하라.
        - 답변에 영문자를 사용하지 말 것.
        - 답변을 임의로 지어내서 답하지 않는다.
        - 모든 답변은 공손한 존대말을 사용하라.
        
        [웹페이지 스크린샷 기반 정보]
        {image_context}

        [API 정의서 엑셀 기반 정보]
        {api_context}

        [홈페이지 사용 설명서]
        {text_context}

        [질문]
        {query}
    """
    #성능이 떨어져서 프롬프트 수정해서 사용함
    prompt_revised=f"""
    [역할]
    너는 소형·중형 건설사를 대상으로 운영하는
    "나라장터 기반 조달·입찰 인텔리전스 플랫폼" 웹사이트의
    사용 방법을 안내하는 전용 챗봇이다.

    ────────────────────────
    [목적]
    사용자가 실제 홈페이지에서 수행해야 할 절차를
    정확하고 간결하게 안내하는 것이 목적이다.
    답변은 반드시 제공된 문서 내용에 근거하여 작성한다.

    ────────────────────────
    [핵심 동작 원칙]
    - 사용자가 바로 실행할 수 있도록 절차 안내를 우선한다.
    - 문맥상 가장 가능성이 높은 기능을 기준으로 바로 안내한다.
    - 문서에 근거가 확인되면 추가 질문 없이 단계별로 안내한다.
    - 추가 정보 요청은 최후의 경우에만 한 문장으로 간결하게 한다.
    - 내부 검증 과정, 문서 존재 여부 판단, 근거 비교 과정은 사용자에게 노출하지 않는다.

    ────────────────────────
    [불명확 질문 처리]
    - 하나의 기능으로 자연스럽게 해석 가능하면 되묻지 않는다.
    - 두 가지 이상 서로 다른 기능으로 명확히 갈리는 경우에만 한 문장으로 확인한다.
    - "문서에 없다", "표현을 다시 말해달라"와 같은 표현은 사용하지 않는다.

    ────────────────────────
    [문서 우선순위]
    1. 홈페이지 사용 설명서
    2. 웹페이지 스크린샷 기반 정보
    3. API 정의서 엑셀 기반 정보

    - 설명이 다를 경우 홈페이지 사용 설명서를 기준으로 안내한다.
    - API 관련 용어, 필드명, 요청·응답 구조는 답변에 직접 언급하지 않는다.

    ────────────────────────
    [추측 제한 규칙]
    - 문서에 전혀 근거가 없는 새로운 기능을 생성하지 않는다.
    - 문서에 존재하는 기능과 의미상 동일한 경우에는 해당 기능으로 판단하여 안내한다.
    - 일반적인 보안 절차, 신원 확인 요구, 개인정보 입력 요청과 같은
    문서에 명시되지 않은 절차는 생성하지 않는다.

    ────────────────────────
    [답변 작성 규칙]
    - 300~600자 이내로 작성한다.
    - 절차 중심으로 단계적으로 설명한다.
    - 소형·중형 건설사 종사자가 이해하기 쉬운 표현을 사용한다.
    - 불필요한 예시, 팁, 배경 설명은 제공하지 않는다.
    - 영문자, 약어, 내부 시스템 용어는 사용하지 않는다.
    - 모든 문장은 공손한 존댓말로 작성한다.
    - 불필요한 사과, 책임 회피 표현, 보안 안내 문구는 사용하지 않는다.

    ────────────────────────
    [홈페이지 사용 설명서]
    {text_context}

    [웹페이지 스크린샷 기반 정보]
    {image_context}

    [API 정의서 엑셀 기반 정보]
    {api_context}

    [질문]
    {query}
    """
    # 5️⃣ 답변 생성
    response = llm.invoke(prompt_revised)
    return response.content # 답변 내용(response.content) 반환
