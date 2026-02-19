import os
import pandas as pd
from PIL import Image
import pytesseract

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import ImageCaptionLoader, UnstructuredExcelLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict
import re

# from usage_tool import usage_tool : 테스트 하려면 활성화

# test_usage_tool.py : 현재 위치에 faiss_db 폴더를 생성 + usage_tool.py 동작 테스트
# C:\BGPJ\BidAssitance\AI_Models\usage_data\images
# =========================
# 경로 설정
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "usage_data", "images")
API_EXCEL_DIR = os.path.join(BASE_DIR, "usage_data", "api정의서.xlsx")
TEXT_DIR=os.path.join(BASE_DIR,"usage_data","홈페이지 사용 설명서.txt")

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMAGE_FAISS_DIR = BASE_DIR / "faiss_db" / "image_faiss"
API_FAISS_DIR= BASE_DIR / "faiss_db" / "api_faiss"
TEXT_FAISS_DIR= BASE_DIR / "faiss_db" / "txt_faiss"

# =========================
# FAISS 생성 임베딩 모델 설정
# =========================
from dotenv import load_dotenv
load_dotenv()


embeddings = OpenAIEmbeddings(model = "text-embedding-3-small") # 임베딩 모델 초기화

# =========================
# 1️⃣ 이미지 → image FAISS 생성 (ImageCaptionLoader, 다른 코드에서 필요시 붙여놓기) 
# =========================
def build_image_faiss():
    print("🔹 image FAISS 생성 중 (ImageCaptionLoader)...")
    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(f"이미지 디렉터리가 없습니다: {IMAGE_DIR}")

    # 1️⃣ 이미지 파일 전체 수집
    image_paths = [
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_paths:
        raise RuntimeError("처리할 이미지 파일이 없습니다.")

    # 2️⃣ ImageCaptionLoader 사용
    loader = ImageCaptionLoader(image_paths)
    documents = loader.load()
    # 3️⃣ metadata 보강
    for doc in documents:
        doc.metadata["source"] = "image"
        doc.metadata["type"] = "screenshot"
    # 4️⃣ FAISS 생성
    faiss = FAISS.from_documents(documents, embeddings)
    faiss.save_local(IMAGE_FAISS_DIR)

# =========================
# 2️⃣ 엑셀 → api FAISS 생성 (UnstructuredExcelLoader, 다른 코드에서 필요시 붙여놓기)
# =========================
def build_api_faiss():
    print("🔹 api FAISS 생성 중 (UnstructuredExcelLoader)...")
    if not os.path.exists(API_EXCEL_DIR):
        raise FileNotFoundError(f"엑셀 파일이 없습니다: {API_EXCEL_DIR}")

    #엑셀 로드
    df = pd.read_excel(API_EXCEL_DIR)

    documents = []

    # ✅ 2. row 하나를 Document 하나로 변환
    for i, row in df.iterrows():
        rest_api = str(row["REST API"])
        input_data = str(row["입력데이터"])
        output_data = str(row["반환데이터"])
        error_data = str(row["오류데이터"])

        content = f"""
        [API URL]
        {rest_api}

        [설명]
        이 API는 {rest_api} 요청을 처리합니다.

        [입력데이터]
        {input_data}

        [반환데이터]
        {output_data}

        [오류데이터]
        {error_data}
        """

    doc = Document(
        page_content=content,
        metadata={
            "source": "api_excel",
            "row": i,
            "api_name": rest_api
        }
    )

    documents.append(doc)

    if not documents:
        raise RuntimeError("엑셀에서 로드된 문서가 없습니다.")

    print(f"총 {len(documents)}개의 API row 문서 생성 완료")


    # 2️⃣ 메타데이터 보강 (권장)
    for idx, doc in enumerate(documents):
        doc.metadata.update({
            "source": "api_excel",
            "element_id": idx
        })
    # 3️⃣ FAISS 생성
    faiss = FAISS.from_documents(documents, embeddings)
    faiss.save_local(API_FAISS_DIR)

def parse_manual_txt(filepath: str) -> List[Document]:
    """[페이지명] 단위로 txt 설명서를 Document로 분리"""

    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # [게시글 페이지] 같은 헤더 기준 split
    pattern = r"\[(.*?)\]"
    splits = re.split(pattern, raw_text)

    documents = []

    # splits 구조:
    # ["", "전체 페이지 공통 사항", "내용...", "게시글 페이지", "내용...", ...]

    for i in range(1, len(splits), 2):
        header = splits[i].strip()
        content = splits[i + 1].strip()

        if not content:
            continue
        
        # 페이지 / 섹션 분리
        if " - " in header:
            page, section = header.split(" - ", 1)
        else:
            page = header
            section = "개요"

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": "homepage_manual",
                    "page": page.strip(),
                    "section":section.strip(),
                    "title":header
                }
            )
        )

    return documents

 # 수정 필요
def build_text_faiss():
    print("🔹 text FAISS 생성 중 (TextLoader)...")
    if not os.path.exists(TEXT_DIR):
        raise FileNotFoundError(f"텍스트 파일이 없습니다: {TEXT_DIR}")

    # 1️⃣ 페이지 단위 parsing
    documents = parse_manual_txt(TEXT_DIR)

    if not documents:
        raise RuntimeError("페이지 단위로 분리된 문서가 없습니다.")

    print(f"✅ 페이지 단위 문서 수: {len(documents)}")

    # 2️⃣ 메타데이터 보강 (권장)
    for idx, doc in enumerate(documents):
        doc.metadata.update({
            "source": "homepage_manual",
            "element_id": idx,
            "page": doc.metadata.get("page"),
            "section": doc.metadata.get("section"),
            "type": "ui_manual"
        })
    # 3️⃣ FAISS 생성
    faiss = FAISS.from_documents(documents, embeddings)
    faiss.save_local(TEXT_FAISS_DIR)

# =========================
# FAISS 값 불러오기 (image / api / text 분리)
# =========================
def load_image_faiss(image_db_path: str) -> FAISS:
    """웹페이지 스크린샷 기반 vectorDB (OCR / Image Caption 결과가 벡터화되어 있음)"""
    if not os.path.exists(image_db_path):
        raise FileNotFoundError(f"Image FAISS DB not found: {image_db_path}")

    return FAISS.load_local(
        image_db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

def load_api_faiss(api_db_path: str) -> FAISS:
    """API 정의서 엑셀 기반 vectorDB (API row 단위 벡터화)"""
    if not os.path.exists(api_db_path):
        raise FileNotFoundError(f"API FAISS DB not found: {api_db_path}")

    return FAISS.load_local(
        api_db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

def load_text_faiss(text_db_path: str) -> FAISS:
    """API 정의서 엑셀 기반 vectorDB (text 벡터화)"""
    if not os.path.exists(text_db_path):
        raise FileNotFoundError(f"Text FAISS DB not found: {text_db_path}")

    return FAISS.load_local(
        text_db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

# =========================
# 벡터 검색 (목적 분리)
# =========================
def search_image_context(image_faiss: FAISS, query: str, k: int = 5) -> List[Document]:
    """UI / 화면 / 사용자 동작 관점 검색"""
    results=image_faiss.similarity_search_with_score(query, k=k)
    return [doc for doc, score in results]

def search_api_context(api_faiss: FAISS, query: str, k: int = 5) -> List[Document]:
    """API 기능 / 요청 / 응답 / 필드 관점 검색"""
    results=api_faiss.similarity_search_with_score(query, k=k)
    return [doc for doc, score in results]

def search_text_context(text_faiss: FAISS, query: str, k: int = 5) -> List[Document]:
    """홈페이지 기능 검색"""
    results=text_faiss.similarity_search_with_score(query, k=k)
    return [doc for doc, score in results]


# =========================
# 컨텍스트 정리 (image / api 분리)
# =========================
def build_context(img_docs: List[Document], api_docs: List[Document], text_docs: List[Document]) -> Dict[str, str]:
    """image / api / text 컨텍스트를 구조적으로 분리하여 반환"""
    image_context = []
    api_context = []
    text_context = []

    if img_docs:
        for d in img_docs:
            image_context.append(d.page_content)
    if api_docs:
        for d in api_docs:
            api_context.append(d.page_content)
    if text_docs:
        for d in text_docs:
            text_context.append(
                f"[페이지명: {d.metadata.get('page','')}]\n"
                f"[섹션: {d.metadata.get('section','')}]\n"
                f"{d.page_content}"
            )

    return {
        "image": "\n\n".join(image_context),
        "api": "\n\n".join(api_context),
        "text": "\n\n".join(text_context)
    }

# # =========================
# # 3️⃣ usage_tool 테스트 (usage_tool 정상출력 테스트용, 필요시 생략 가능)
# # =========================
# def test_usage_tool():
#     query = "게시글을 작성하려면 어떻게 해야돼?"

#     result = usage_tool.invoke({
#         "query": query,
#         "img": IMAGE_FAISS_DIR,
#         "api": API_FAISS_DIR
#     })

#     print("\n====================")
#     print("🤖 AI 응답 결과")
#     print("====================\n")
#     print(result)

# # =========================
# # main
# # =========================
#if __name__ == "__main__":      # Python 스크립트가 직접 실행될 때만 작성된 세 개의 함수를 순차적으로 호출.
    # build_image_faiss()
    # build_api_faiss()
    #test_usage_tool()
    #build_text_faiss()
