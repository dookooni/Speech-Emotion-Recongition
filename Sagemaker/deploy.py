import sagemaker
from sagemaker.huggingface.model import HuggingFaceModel
import boto3

# --- 1. 기본 정보 설정 ---
role_arn = "arn:aws:iam::891084863368:role/service-role/AmazonSageMaker-ExecutionRole-20250826T164246"
model_s3_uri = "s3://ildan-model/emotional-analysis-model/model.tar.gz"
# 이름 중복을 피하기 위해 새 이름 사용 또는 이전 리소스 정리를 권장합니다.
endpoint_name = "ser-model-public" 
boto_session = boto3.Session(region_name="ap-northeast-2")

# --- 2. SageMaker 모델 생성 ---
sagemaker_session = sagemaker.Session(boto_session=boto_session)

# VPC 설정을 사용하지 않고 공개망에 배포합니다.
huggingface_model = HuggingFaceModel(
    model_data=model_s3_uri,
    role=role_arn,
    sagemaker_session=sagemaker_session,
    transformers_version="4.28",
    pytorch_version="2.0",
    py_version="py310"
)

print("SageMaker 모델 객체를 생성했습니다. 이제 실시간 엔드포인트 배포를 시작합니다...")

# --- 3. 실시간 엔드포인트 배포 ---
# 더 큰 디스크를 가진 GPU 인스턴스 유형을 직접 지정합니다.
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.2xlarge", # <-- 이미지 크기 오류 해결을 위한 인스턴스
    endpoint_name=endpoint_name
)

print(f"엔드포인트 '{predictor.endpoint_name}' 배포를 요청했습니다.")
print("배포가 완료될 때까지 기다립니다... ☕️")

# --- 4. 배포 완료 확인 ---
client = boto3.client("sagemaker")
waiter = client.get_waiter('endpoint_in_service')
waiter.wait(EndpointName=endpoint_name)

print("🎉 배포가 성공적으로 완료되었습니다!")
print(f"Endpoint Name: {endpoint_name}")