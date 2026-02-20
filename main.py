# main.py
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator, root_validator
from dotenv import load_dotenv
from pyngrok import ngrok
from langchain_core.messages import HumanMessage
from typing import Optional, Dict, Any, Union, List
import requests
from pathlib import Path
import PyPDF2
from datetime import datetime
import logging
import uuid
import json
from langchain_core.messages import ToolMessage
import tempfile

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

# ==========================================
# TFT + RAG Pipeline 초기화
# ==========================================

from BidAssitanceModel import BidRAGPipeline, extract_text_from_hwp, extract_text_from_hwpx, extract_text_from_pdf
from tft_v3_predictor import predict_sajeong_percent, FEATURES, get_highest_probability_ranges_v3
from get_probability_from_model import ProbabilityPredictor
import re
import uuid
import os

def parsenumber(value: Any) -> Optional[float]:
    """
    다양한 형태의 숫자 문자열을 float로 변환
    예: "1,000,000원" -> 1000000.0
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    s = re.sub(r'[^0-9.\-]', '', s.replace(',', ''))
    try:
        return float(s)
    except:
        return None
    
tft_predictor = None

class TFTPredictorAdapter:
    """RAG 파이프라인에서 사용할 TFT 모델 어댑터 - top_ranges 지원"""

    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, requirements: Dict[str, Any], retrieved_context: str = "") -> Dict[str, Any]:
        """입찰 요구사항을 기반으로 TFT 모델로 예측 수행 - top_ranges 포함"""
        try:
            if not self.predictor:
                return {
                    "error": "Model not loaded",
                    "point_estimate": 0,
                    "confidence": "error",
                    "rationale": "TFT Model not loaded"
                }

            # 입력 데이터 파싱
            pr_range = parsenumber(requirements.get('expected_price_range')) or 0.0
            lower_rate = parsenumber(requirements.get('award_lower_rate')) or 0.0
            estimate = parsenumber(requirements.get('estimate_price')) or 0.0
            budget = parsenumber(requirements.get('budget')) or 0.0

            input_dict = {
                '예가범위': pr_range,
                '낙찰하한율': lower_rate,
                '추정가격': estimate,
                '기초금액': budget
            }

            # TFT 모델로 확률 높은 상위 3개 구간 예측
            result = self.predictor.get_highest_probability_ranges(
                input_dict,
                bin_width=0.001,
                top_k=3
            )

            if result and result.get("top_ranges"):
                top_ranges = result["top_ranges"]

                # 🔍 디버그: top_ranges 상세 출력
                print("=" * 60)
                print(" [DEBUG] TFTPredictorAdapter - top_ranges 상세:")
                for i, r in enumerate(top_ranges[:3], start=1):
                    center_val = r.get("center")
                    prob_val = r.get("probability")

                    # center / probability가 [값, 소수자리] 형태면 값만 꺼냄
                    if isinstance(center_val, list):
                        center_val = center_val[0]
                    if isinstance(prob_val, list):
                        prob_val = prob_val[0]

                    # range_display 없으면 lower/upper로 만들어줌
                    range_display = r.get("range_display")
                    if not range_display and r.get("lower") is not None and r.get("upper") is not None:
                        range_display = f"{r['lower']:.2f}% ~ {r['upper']:.2f}%"

                    print(f"  {i}순위:")
                    print(f"    range_display: {range_display}")
                    print(f"    center: {center_val:.2f}%")
                    print(f"    probability: {prob_val:.2f}%")
                print("=" * 60)
                # 낙찰가 계산: 기초금액 × 배율(1+사정율) × 낙찰하한율
                # center는 배율 (1 + 사정율) 형태
                pred_multiplier = float(top_ranges[0]["center"])

                # center가 99.xx 같은 퍼센트로 들어오는 경우 방어
                if pred_multiplier > 2:
                    pred_multiplier /= 100.0

                # 낙찰가 = 기초금액 × 투찰배율(99%)
                award_price = round(budget * pred_multiplier) if budget else None

                # 퍼센트는 금액에서 역산 → 항상 일치
                predicted_percent = (award_price / budget) * 100 if (award_price and budget) else None

                return {
                    "currency": "KRW",
                    "point_estimate": award_price,  # 원 단위 낙찰가
                    "predicted_sashiritsu": abs(pred_multiplier - 1),  # 사정율 (배율에서 변환)
                    "predicted_min": abs(result["statistics"]["q25"] - 1),  # 사정율 하한
                    "predicted_max": abs(result["statistics"]["q75"] - 1),  # 사정율 상한
                    "confidence": "high",
                    "top_ranges": top_ranges,
                    "statistics": result["statistics"],
                    "rationale": f"TFT Model - Top {len(top_ranges)} 확률 구간 분석 완료",
                    "model_type": "QuantileTransformerRegressor"
                }
            else:
                return {
                    "error": "Prediction failed",
                    "point_estimate": 0,
                    "confidence": "low",
                    "rationale": "TFT 예측 결과 없음"
                }

        except Exception as e:
            print(f" TFT 예측 오류: {e}")
            return {
                "error": str(e),
                "point_estimate": 0,
                "confidence": "error",
                "rationale": f"Prediction Failed: {str(e)}"
            }
        
def v3_award_predict(requirements: Dict[str, Any], retrieved_context: str = "") -> Dict[str, Any]:
    try:
        pr_range = parsenumber(requirements.get('expected_price_range')) or 0.0
        lower_rate_raw = parsenumber(requirements.get('award_lower_rate')) or 0.0
        estimate = parsenumber(requirements.get('estimate_price')) or 0.0
        budget = parsenumber(requirements.get('budget')) or 0.0

        lower_rate = lower_rate_raw
        if lower_rate > 1:
            lower_rate = lower_rate / 100.0

        feat = {name: 0.0 for name in FEATURES}
        if "예가범위" in feat: feat["예가범위"] = float(pr_range)
        if "낙찰하한율" in feat: feat["낙찰하한율"] = float(lower_rate_raw)
        if "추정가격" in feat: feat["추정가격"] = float(estimate)
        if "기초금액" in feat: feat["기초금액"] = float(budget)

        #  top3 확률 구간
        dist = get_highest_probability_ranges_v3(feat, bin_width=0.0001, top_k=3)
        top_ranges = dist.get("top_ranges", [])
        statistics = dist.get("statistics", {})

        #  중앙값 예측(배율)
        pred_multiplier = float(predict_sajeong_percent(feat))
        if pred_multiplier > 2:
            pred_multiplier /= 100.0

        award_price = round(budget * pred_multiplier) if budget else None
        predicted_percent = (award_price / budget) * 100 if (award_price and budget) else None
        lower_bound_price = round(budget * pred_multiplier * lower_rate) if (budget and lower_rate) else None


        converted = []
        for r in top_ranges:
            # dist에서 오는 값들
            center = float(r.get("center", 0.0))
            low = float(r.get("lower", 0.0))
            high = float(r.get("upper", 0.0))
            prob = float(r.get("probability", 0.0))

            # center/lower/upper가 배율(1.00xx) 형태면 퍼센트(100.xx)로 변환
            # 예: 1.0027 -> 100.27
            if center <= 2.0:
                center *= 100.0
                low *= 100.0
                high *= 100.0

            # 확률(prob)은 get_highest_probability_ranges_v3 결과가 보통 이미 % 스케일(예: 31.12)이라 가정
            # 만약 0~1로 오는 경우(예: 0.3112)이면 %로 변환
            if 0.0 <= prob <= 1.0:
                prob *= 100.0

            converted.append({
                **r,

                # ✅ LLM이 그대로 보고서에 쓰는 필드들
                "range_display": f"{low:.2f}% ~ {high:.2f}%",
                "rate": round(center, 2),  # 사정율(퍼센트 표기 값) → {rate:.2f}로 바로 출력 가능
                "probability": round(prob, 2),  # 확률(%) → {probability:.2f}로 바로 출력 가능

                # 참고용(숫자 보관)
                "lower": round(low, 2),
                "upper": round(high, 2),
                "range": [round(low, 2), round(high, 2)],
            })

        top_ranges = converted

        # statistics도 깔끔하게 (q25/q50/q75가 배율이면 %로 변환)
        if isinstance(statistics, dict):
            for k in ("q25", "q50", "q75"):
                v = statistics.get(k)
                if isinstance(v, (int, float)):
                    v = float(v)
                    if v <= 2.0:  # 배율이면
                        v *= 100.0
                    statistics[k] = round(v, 2)

        return {
            "currency": "KRW",
            "point_estimate": award_price,
            "predicted_percent": predicted_percent,
            "confidence": "high",
            "rationale": "TFT v3(pt) median quantile prediction (multiplier)",
            "model_type": "v3_pt",
            "pred_multiplier": pred_multiplier,
            "lower_bound_price": lower_bound_price,
            "top_ranges": top_ranges,
            "statistics": statistics
        }

    except Exception as e:
        return {
            "error": str(e),
            "point_estimate": 0,
            "confidence": "error",
            "rationale": f"V3 Prediction Failed: {str(e)}"
        }

# RAG Pipeline 생성
adapter = TFTPredictorAdapter(tft_predictor)

rag_pipeline = BidRAGPipeline(
    doc_dir="./rag_corpus",
    index_dir="./rag_index",
    award_predict_fn=v3_award_predict
)

print("🚀 RAG + TFT Pipeline Ready")

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
    type: str="choose query | notice_result | report"
    query: str="user question"
    payload: Optional[Union[Dict[str, Any], List[Dict[str, Any]],str]] = None
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

@app.post("/chat/file")
async def analyze(
    file: UploadFile = File(...),      # Spring에서 보낸 파일
    text: str = Form(...),             # Spring에서 보낸 질문 ("이 문서 요약해줘")
    thread_id: str = Form("default")   # 세션 ID
):
    """입찰공고 분석 + TFT 예측 + PDF 생성"""
    try:
        # 1) 업로드 파일 이름 확인
        filename = file.filename.lower()

        # 2) 임시 저장
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(await file.read())

        # 3) 파일 텍스트 추출
        if filename.endswith(".pdf"):
            extracted_text = extract_text_from_pdf(tmp_path)

        elif filename.endswith(".hwp"):
            extracted_text = extract_text_from_hwp(tmp_path)
        elif filename.endswith(".hwpx"):
            extracted_text = extract_text_from_hwpx(tmp_path)
        else:
            os.remove(tmp_path)
            raise HTTPException(
                status_code=400,
                detail="지원하지 않는 파일 형식입니다. (pdf/hwp/hwpx만 가능)"
            )
        
        # 4) 추출 실패 체크
        if not extracted_text.strip():
            os.remove(tmp_path)
            raise HTTPException(
                status_code=400,
                detail="파일에서 텍스트를 추출하지 못했습니다."
            )
        
        # 1. RAG 파이프라인 분석 수행
        result = rag_pipeline.analyze(
            extracted_text,
            thread_id=thread_id
        )

        report_md = result.get("report_markdown", "")
        prediction_result = result.get("prediction_result", {})
        os.remove(tmp_path)

        # 2. 응답 반환
        return {
            "report": report_md,
            #"pdf_link": final_url,
            "thread_id": thread_id
        }

    except Exception as e:
        print(f"❌ /chat/file 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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

        # 응답 type 결정 (요청 type(req.type) 말고 "결과" 기준)
        resp_type = "chat"

        # 후처리 요청이면 summary로 고정
        if req.type in ("notice_result"):
            resp_type = "search"
        
        # pydantic 에러 감지
        parsed=None
        try:
            parsed = json.loads(final_text)
        except:
            pass
        if isinstance(parsed, dict) and parsed.get("__error__") == "pydantic_validation":
            return {
                "type": resp_type,
                "response": "질문이 조금 모호합니다. \n원하시는 조건을 조금 더 자세히 말씀해 주시길바랍니다.",
                "thread_id": req.thread_id
        }
        
        return {
            "type": resp_type,
            "response": final_text,
            "thread_id": req.thread_id
        }
        
    except Exception as e:
        print(f"Error: {e}")
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
