"""
SageMaker Async Inference for Speech Emotion Recognition (SER)
음성 감정 인식을 위한 SageMaker 비동기 추론 엔드포인트
"""
import json
import os
import sys
import logging
import torch
import numpy as np
import librosa
import soundfile as sf
from io import BytesIO
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
import tarfile
import tempfile

# 로깅 설정 (먼저 설정)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SageMaker 환경에서 모듈 경로 설정
def setup_module_paths():
    """SageMaker 배포 환경에서 모듈 경로를 설정합니다."""
    current_dir = Path(__file__).parent
    
    # 가능한 모든 경로를 sys.path에 추가
    paths_to_add = [
        str(current_dir),
        str(current_dir.parent),
        str(current_dir.parent.parent),
        '/opt/ml/code',
        '/opt/ml/code/SER',
        '/opt/ml/code/SER/Sagemaker/code',
    ]
    
    for path in paths_to_add:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    
    logger.info(f"📂 Module paths configured. Current working directory: {os.getcwd()}")

# 경로 설정 실행
setup_module_paths()

from transformers import Wav2Vec2Processor, Wav2Vec2Config

# 안전한 모델 import
custom_Wav2Vec2ForEmotionClassification = None

def import_model_class():
    """다양한 경로에서 모델 클래스를 안전하게 import합니다."""
    global custom_Wav2Vec2ForEmotionClassification
    
    import_attempts = [
        "model",
        "SER.Sagemaker.code.model", 
        "Sagemaker.code.model",
        "code.model",
    ]
    
    for module_name in import_attempts:
        try:
            logger.info(f"🔍 Attempting to import from: {module_name}")
            module = __import__(module_name, fromlist=['custom_Wav2Vec2ForEmotionClassification'])
            custom_Wav2Vec2ForEmotionClassification = getattr(module, 'custom_Wav2Vec2ForEmotionClassification')
            logger.info(f"✅ Successfully imported from: {module_name}")
            return custom_Wav2Vec2ForEmotionClassification
        except (ImportError, AttributeError) as e:
            logger.warning(f"❌ Failed to import from {module_name}: {e}")
            continue
    
    logger.error("❌ Failed to import custom_Wav2Vec2ForEmotionClassification from any location")
    raise ImportError("Could not find custom_Wav2Vec2ForEmotionClassification in any known location")

# 모델 클래스 import 실행
custom_Wav2Vec2ForEmotionClassification = import_model_class()

# 전역 변수
model = None
processor = None
device = None
emotion_labels = ["Anxious", "Dry", "Kind", "Other"]
MODEL_SAMPLE_RATE = 16000  # 모델의 샘플링 레이트
MODEL_MAX_DURATION = 10.0
ROOT = Path(__file__).parent


def model_fn(model_dir: str):
    """
    SageMaker에서 모델을 로드하는 함수
    Args:
        model_dir: 모델이 저장된 디렉토리 경로
    Returns:
        로드된 모델과 프로세서를 포함한 딕셔너리
    """
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(model_dir)
    weight_dir = model_dir / "model" / "wav2vec"
    logger.info(f"Loading model from: {model_dir}")
    logger.info(f"Using device: {device}")
    
    try:    
        # 모델 로드
        logger.info("Loading model...")
        
        model_config = Wav2Vec2Config.from_pretrained(
            "kresnik/wav2vec2-large-xlsr-korean",
            num_labels= 4,
            finetuning_task="emotion_classification"
        )
        
        model_config.num_speakers = 6
        model_config.char_vocab_size = 100

        model = custom_Wav2Vec2ForEmotionClassification.from_pretrained(
            weight_dir,
            config=model_config,
            ignore_mismatched_sizes=True
        )
        
        processor = Wav2Vec2Processor.from_pretrained(weight_dir)
        model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully!")
        
        return {
            'model': model,
            'processor': processor,
            'device': device,
            'emotion_labels': emotion_labels
        }
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def input_fn(request_body: bytes, content_type: str) -> np.ndarray:
    """
    클라이언트로부터 받은 요청을 모델이 처리할 수 있는 NumPy 배열로 변환합니다.
    """
    logger.info(f"Received request with content_type: {content_type}")
    try:
        if content_type == 'application/json':
            # 1. JSON 파싱 및 Base64 데이터 추출
            data = json.loads(request_body)
            audio_base64 = data['instances'][0]['audio_data']
            
            # 2. Base64 디코딩하여 바이트로 변환
            audio_bytes = base64.b64decode(audio_base64)
            
            # 3. ✅ 바이트를 Raw PCM으로 직접 처리 (sf.read 대신 frombuffer 사용)
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            audio = audio.astype(np.float32) / 32768.0 # float32로 정규화
            sr = MODEL_SAMPLE_RATE # Raw PCM이므로 샘플링 레이트를 직접 지정

        elif content_type == 'audio/wav':
            audio_bytes = request_body
            audio, sr = sf.read(BytesIO(audio_bytes))
            
        elif content_type == 'application/octet-stream':
            audio_bytes = request_body
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            audio = audio.astype(np.float32) / 32768.0 # float32로 정규화
            sr = MODEL_SAMPLE_RATE 
            
        else:
            raise ValueError(f"Unsupported content type: {content_type}")

        # --- 공통 후처리 로직 ---
        if audio.ndim > 1: audio = np.mean(audio, axis=1) # 스테레오 -> 모노
        if sr != MODEL_SAMPLE_RATE:
            logger.info(f"Resampling from {sr} Hz to {MODEL_SAMPLE_RATE} Hz")
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=MODEL_SAMPLE_RATE)
        if np.max(np.abs(audio)) > 0: audio = audio / np.max(np.abs(audio)) # 정규화

        logger.info(f"Decoded audio array with shape: {audio.shape}")
        return audio.astype(np.float32)
        
    except Exception as e:
        logger.error(f"ERROR in input_fn: {e}")
        raise ValueError(f"Failed to process input audio: {e}")

def predict_fn(input_data: np.ndarray, model_pack: dict) -> dict:
    """
    전처리된 오디오 배열을 받아 모델 추론을 수행하고, 결과를 딕셔너리로 반환합니다.
    """
    model = model_pack['model']
    processor = model_pack['processor']
    device = model_pack['device']
    emotion_labels = model_pack['emotion_labels']
    audio = input_data
    
    target_length = int(MODEL_SAMPLE_RATE * MODEL_MAX_DURATION)
    if len(audio) > target_length:
        start_idx = (len(audio) - target_length) // 2
        audio = audio[start_idx : start_idx + target_length]
    elif len(audio) < target_length:
        pad_length = target_length - len(audio)
        audio = np.pad(audio, (0, pad_length), mode='constant')
        
    inputs = processor(audio, sampling_rate=MODEL_SAMPLE_RATE, return_tensors="pt")
    input_values = inputs.input_values.to(device)
    
    try:
        with torch.no_grad():
            outputs = model(input_values)
            logits = outputs.get('emotion_logits')
            if logits is None:
                raise ValueError("Model output dictionary does not contain 'emotion_logits' key")

            probabilities = torch.softmax(logits, dim=-1).squeeze()
            predicted_idx = torch.argmax(probabilities).item()
            probabilities_np = probabilities.cpu().numpy()

        # 4. 결과 포맷팅
        results = {
            'predicted_emotion': emotion_labels[predicted_idx],
            'confidence': float(probabilities_np[predicted_idx]),
            'emotion_scores': {label: float(prob) for label, prob in zip(emotion_labels, probabilities_np)}
        }
        print(f"Prediction successful: {results['emotion_scores']}")
        return results

    except Exception as e:
        print(f"ERROR during prediction: {e}")
        return {'error': str(e)}


def output_fn(prediction: dict, accept: str) -> bytes:
    """
    최종 예측 결과를 클라이언트가 요청한 형식(JSON)으로 인코딩합니다.
    """
    if accept != 'application/json':
        raise ValueError(f"Unsupported accept type: {accept}")

    # bytes 객체만 반환하도록 수정
    return json.dumps(prediction, ensure_ascii=False).encode('utf-8')

# SageMaker 비동기 추론을 위한 추가 함수들
def health_check():
    """헬스 체크 함수"""
    return {"status": "healthy", "model_loaded": model is not None}


def batch_predict(batch_input: List[Dict[str, Any]], model_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    배치 예측 함수 (여러 오디오 파일 동시 처리)
    
    Args:
        batch_input: 배치 입력 데이터 리스트
        model_dict: 모델 딕셔너리
        
    Returns:
        배치 예측 결과 리스트
    """
    results = []
    for input_data in batch_input:
        result = predict_fn(input_data, model_dict)
        results.append(result)
    return results


# SageMaker Inference Container를 위한 엔트리포인트
if __name__ == "__main__":
    import sys
    
    # 테스트 모드
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Testing SageMaker inference...")
        
        # 더미 모델 로드 (실제 환경에서는 SageMaker가 호출)
        model_dir = ROOT / "pre_trained" / "Wav2Vec2_Adversary" # SageMaker 기본 모델 경로
            
        if os.path.exists(model_dir):
            model_dict = model_fn(model_dir)
            print("Model loaded successfully!")
            
            # 더미 오디오 데이터로 테스트
            dummy_audio = np.random.randn(16000).astype(np.float32)  # 1초 더미 오디오
            
            # 오디오를 바이트로 변환
            with BytesIO() as buf:
                sf.write(buf, dummy_audio, 16000, format='WAV')
                audio_bytes = buf.getvalue()
            
            # 예측 테스트
            input_data = {'audio_data': audio_bytes, 'format': 'binary'}
            result = predict_fn(input_data, model_dict)
            print(f"Test prediction result: {result}")
        else:
            print(f"Model directory not found: {model_dir}")
    else:
        print("SageMaker inference server ready!")
