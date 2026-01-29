# 1. Python 3.11 슬림 이미지 사용
FROM python:3.11-slim

# 2. 시스템 패키지 설치
# 수정됨: libgl1-mesa-glx -> libgl1 (최신 데비안 버전 호환)
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-kor \
    libmagic-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 라이브러리 설치
# (이 단계 전에 반드시 requirements.txt 파일이 폴더에 있어야 함)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 및 데이터 복사
COPY . .

# 6. 환경 변수 설정
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 7. 포트 노출
EXPOSE 8000

# 8. 실행 명령어
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]