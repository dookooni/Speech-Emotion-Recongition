# 1. AWS 공식 딥러닝 컨테이너 이미지를 베이스로 사용 (더 안정적)
FROM 763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-inference:1.12.1-gpu-py38-cu113-ubuntu20.04

# 작업 디렉토리 설정
WORKDIR /opt/ml/code

# 시스템 패키지 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    sox \
    libsox-fmt-all \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
# Gunicorn과 Flask를 추가하여 웹 서버를 실행합니다.
RUN pip install --no-cache-dir -r requirements.txt gunicorn flask

# 모델 코드 복사
COPY inference.py .
COPY model.py .

# SageMaker에서 사용하는 환경 변수 설정
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# 2. 웹 서버 실행을 위한 포트 노출
EXPOSE 8080

# 3. SageMaker 추론 서버 실행 (Gunicorn 사용)
# Gunicorn이 8080 포트에서 inference.py 안에 있는 app 객체를 실행합니다.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "inference:app"]