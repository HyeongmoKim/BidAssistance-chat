# main.py
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator, root_validator
from dotenv import load_dotenv
from pyngrok import ngrok
from langchain_core.messages import HumanMessage
from typing import Optional, Dict, Any
import requests
from pathlib import Path
import PyPDF2
from datetime import datetime
import logging
import uuid
import json
from langchain_core.messages import ToolMessage

# 분리된 그래프 앱 import
from graph import graph_app

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =================================================================
# 1. Config & Setup
# =================================================================
class Config:
    NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    @classmethod
    def check(cls):
        if not cls.OPENAI_API_KEY:
            print("⚠️ Warning: OPENAI_API_KEY가 설정되지 않았습니다.")

Config.check()

# =================================================================
# 2. FastAPI App Setup
# =================================================================
app = FastAPI(
    title="LangGraph Chatbot API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
    )

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP 요청 로깅 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    logger.info(f"Request: {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Response: {request.method} {request.url.path} - {response.status_code} - {duration:.2f}s")
        return response
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise

# 요청 데이터 모델
class ChatRequest(BaseModel):
    type: str="query"
    query: str=""
    payload: Optional[Dict[str,Any]]=None
    thread_id: str = "default_session"  # 세션 구분을 위한 ID

class AnalyzeRequest(BaseModel):
    text: Optional[str] = None
    file_url: Optional[str] = None  # 파일 URL
    pdf_path: Optional[str] = None  # 파일 경로
    
    @root_validator(pre=True)
    def check_at_least_one(cls, values):
        # 최소 하나의 입력 소스 검증
        if not any([values.get('text'), values.get('file_url'), values.get('pdf_path')]):
            raise ValueError('At least one input source required (text, file_url, or pdf_path)')
        return values

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str
    path: str

# HTTPException 예외 처리
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP_{exc.status_code}",
            detail=exc.detail,
            timestamp=datetime.now().isoformat(),
            path=str(request.url.path)
        ).dict()
    )

# 일반 예외 처리
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="INTERNAL_SERVER_ERROR",
            detail=str(exc),
            timestamp=datetime.now().isoformat(),
            path=str(request.url.path)
        ).dict()
    )

@app.get("/status_check")
def root():
    return {"status": "running", "message": "LangGraph API is active"}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    LangGraph를 실행하여 답변을 생성하는 엔드포인트
    """
    try:
        if req.type == "query":
            content = req.query

        else:
            # payload 기반 후처리 입력
            content = json.dumps(
                {
                    "type": req.type,
                    "payload": req.payload
                },
                ensure_ascii=False
            )
        
        #질문 형태가 아닌데 담겨오는 값이 없을 때
        if req.type != "query" and req.payload is None:
            raise HTTPException(status_code=400, detail="payload is required")

        # LangGraph 입력 메시지 생성
        
        inputs = {"messages": [HumanMessage(content=content)]}
        config = {"configurable": {"thread_id": req.thread_id}}
        
        # 그래프 실행 (invoke는 동기 함수이므로 async def 안에서는 주의 필요)
        # LangGraph의 invoke()는 최종 상태를 반환합니다.
        final_state = await graph_app.ainvoke(inputs, config=config)
        
        # 마지막 메시지(AI 답변) 추출
        last_message = final_state["messages"][-1]
        final_text = last_message.content if last_message else ""

        # ✅ 응답 type 결정 (요청 type(req.type) 말고 "결과" 기준)
        resp_type = "chat"

        # 후처리 요청이면 summary로 고정
        if req.type in ("notice_result", "report"):
            resp_type = "summary"
        else:
            # tool이 마지막이거나 JSON처럼 보이면 search로 분류
            if isinstance(last_message, ToolMessage):
                resp_type = "search"
            else:
                s = (final_text or "").strip()
                if s.startswith("{") and s.endswith("}"):
                    resp_type = "search"

        return {
            "type": resp_type,  # ✅ 핵심: req.type이 아니라 resp_type
            "response": final_text,
            "thread_id": req.thread_id
        }


    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/input_content")
async def input_content_endpoint(req: AnalyzeRequest):
    """
    파일 URL 또는 PDF 경로에서 파일을 읽어 분석하는 엔드포인트
    (텍스트는 /chat 엔드포인트 사용)
    """
    content = None
    source = None
    
    # 파일 URL에서 다운로드 및 처리
    if req.file_url:
        try:
            response = requests.get(req.file_url, timeout=30)
            response.raise_for_status()
            content = response.text
            source = "file_url"
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")
    
    # PDF 파일 경로에서 텍스트 추출
    elif req.pdf_path:
        try:
            pdf_path = Path(req.pdf_path)
            if not pdf_path.exists():
                raise HTTPException(status_code=400, detail=f"PDF file not found: {req.pdf_path}")
            
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text()
            source = "pdf_path"
        except HTTPException:
            # HTTPException은 그대로 전파
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")
    
    # 텍스트 입력 (plain text는 /chat 엔드포인트 사용 권장)
    elif req.text:
        content = req.text
        source = "text"
    
    # LangGraph를 실행하여 분석
    try:
        inputs = {"messages": [HumanMessage(content=str(content))]}
        # 고유한 thread_id 생성
        thread_id = f"analyze_{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}
        
        final_state = await graph_app.ainvoke(inputs, config=config)
        last_message = final_state["messages"][-1]
        
        return {
            "source": source,
            "content_length": len(str(content)),
            "content_preview": str(content)[:200] + "..." if len(str(content)) > 200 else str(content),
            "response": last_message.content,
            "thread_id": thread_id,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in graph execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# =================================================================
# 3. Server Execution
# =================================================================
if __name__ == "__main__":
    # ngrok 설정 (외부 접속 필요 시)
    if Config.NGROK_AUTH_TOKEN:
        ngrok.set_auth_token(Config.NGROK_AUTH_TOKEN)
        public_url = ngrok.connect(8000)
        print(f"\n🌍 Public URL: {public_url.public_url}\n")
    else:
        print("\n[Info] 로컬 모드로 실행됩니다. (http://localhost:8000)\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
