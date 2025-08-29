# SER (Speech Emotion Recognition) 모듈

한국어 음성 감정 분석을 위한 Wav2Vec2 기반 Fine-tuning 모듈입니다.

## 📋 개요

이 모듈은 Hugging Face의 [`kresnik/wav2vec2-large-xlsr-korean`](https://huggingface.co/kresnik/wav2vec2-large-xlsr-korean) 모델을 기반으로 4가지 감정을 분류하는 음성 감정 분석 시스템입니다.

### 🎯 기반 모델 정보
- **모델**: [kresnik/wav2vec2-large-xlsr-korean](https://huggingface.co/kresnik/wav2vec2-large-xlsr-korean)
- **파라미터 수**: 317M
- **원본 성능**: WER 4.74%, CER 1.78% (Zeroth-Korean ASR)
- **샘플링 레이트**: 16kHz
- **라이선스**: Apache 2.0

### 지원하는 감정 클래스
- Anxious (불안)
- Kind (친절)
- Dry (건조함)
- Others (기타)

## 🏗️ 프로젝트 구조

```
SER/
├── Sagemaker/
    ├── code /                    # Sagemaker 추론을 위한 model.tar.gz 파일 구성
    ├── deploy.py                 # Sagemaker 배포
    ├── Dockerfile                # Sagemaker 배포용 도커 이미지
├── config.py                     # 설정 관리
├── model.py                      # Wav2Vec2 모델 구현
├── preprocessing.py              # 오디오 전처리
├── datasets.py                   # 데이터 로더
├── trainer.py                    # 1 GPU 훈련
├── trainer_accelerate.py         # Multi GPU 훈련
├── trainer_accelerate_text.py    # Text 버전 Multi GPU 훈련
├── test_dataset.py               # 모델 평가
├── utils.py                      # 유틸리티 함수
├── data_utils.py                 # 데이터 유틸리티 함수
├── model_utils.py                # 모델 유틸리티 함수
├── requirements.txt              # 의존성
└── README.md                     # 문서
```

## ⚡ 빠른 시작

### 1. 의존성 설치

```bash
cd Backend/SER
pip install -r requirements.txt
```

### 2. 기본 사용법

#### 모델 훈련

```bash
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes=3 trainer_accelerate_text.py conda activate ser2
```

#### 추론

#### 모델 평가

```bash
python -m SER.evaluate \
    --model_path ./results \
    --data_dir ./test_data \
    --output_dir ./evaluation_results
```


## 📊 데이터 준비

### 디렉토리 구조 방식

```
data/
├── F0001/
│   ├── wav_48000/
│   │   ├── file1.wav
│   │   └── file2.wav
│   ├── script.txt
│   └── ...
```

### CSV 파일 방식

```csv
file_path,emotion
/path/to/audio1.wav,Neutral
/path/to/audio2.wav,Angry
/path/to/audio3.wav,Joy
...
```

## ⚙️ 설정

### 모델 설정 (`config.py`)

```python
from SER.config import model_config

# 기본 설정 확인
print(f"모델: {model_config.model_name}")
print(f"감정 클래스: {model_config.emotion_labels}")

# 설정 수정
model_config.max_duration = 15.0  # 최대 오디오 길이 변경
```

### 훈련 설정

```python
from SER.config import training_config

# 훈련 파라미터 조정
training_config.batch_size = 16
training_config.learning_rate = 5e-5
training_config.num_epochs = 20
```

## 📈 모델 성능 모니터링

### 훈련 중 모니터링

```python
# 훈련 히스토리 시각화
from SER.utils import plot_training_history

history = {
    'train_loss': [0.8, 0.6, 0.4, 0.3],
    'val_loss': [0.9, 0.7, 0.5, 0.4],
    'val_accuracy': [0.6, 0.7, 0.8, 0.85],
    'val_f1': [0.55, 0.65, 0.75, 0.82]
}

plot_training_history(history, save_path='training_history.png')
```

### 평가 및 분석

```python
from SER.evaluate import ModelEvaluator

# 평가기 생성
evaluator = ModelEvaluator("./results")

# 데이터셋 평가
results = evaluator.evaluate_dataset(audio_paths, true_labels)

# 혼동 행렬 시각화
evaluator.plot_confusion_matrix()

# 클래스별 성능 시각화
evaluator.plot_class_metrics()

# 보고서 생성
report = evaluator.generate_evaluation_report()
print(report)
```

## 🔧 고급 사용법

### 커스텀 데이터 로더

```python
from SER.data_loader import SpeechEmotionDataset
from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")

dataset = SpeechEmotionDataset(
    data_paths=audio_paths,
    labels=labels,
    processor=processor,
    is_training=True
)
```

### 커스텀 전처리

```python
from SER.preprocessing import AudioPreprocessor

preprocessor = AudioPreprocessor(
    target_sr=16000,
    max_duration=15.0,
    normalize=True
)

# 오디오 전처리
processed_audio = preprocessor.preprocess(
    "audio_file.wav",
    apply_augmentation=True
)
```

### 배치 추론

```python
# 여러 파일 동시 처리
audio_files = ["file1.wav", "file2.wav", "file3.wav"]
results = inference.predict_batch(audio_files)

# 디렉토리 전체 처리
results = inference.predict_directory("./audio_folder")

# 감정 분포 분석
distribution = inference.analyze_emotions_distribution(results)
print(f"가장 많은 감정: {max(distribution['emotion_counts'], key=distribution['emotion_counts'].get)}")
```

## 🎯 최적화 팁

### 1. kresnik 모델 하이퍼파라미터 (권장)

```python
# kresnik/wav2vec2-large-xlsr-korean (317M 파라미터)에 최적화된 설정
training_config.learning_rate = 3e-5  # 큰 모델에 적합한 낮은 학습률
training_config.batch_size = 4        # GPU 메모리 고려
training_config.gradient_accumulation_steps = 4  # 효과적 배치 크기 = 16
training_config.warmup_steps = 1000   # 더 긴 웜업
training_config.num_epochs = 5        # 큰 모델은 적은 에포크로도 효과적

# GPU 메모리에 따른 조정
# RTX 3090/4090 (24GB): batch_size = 8, gradient_accumulation_steps = 2
# RTX 3080/4080 (12GB): batch_size = 4, gradient_accumulation_steps = 4  
# RTX 3070/4070 (8GB):  batch_size = 2, gradient_accumulation_steps = 8
```

### 2. 데이터 증강

```python
from SER.config import data_config

# 데이터 증강 활성화
data_config.data_augmentation = True
data_config.apply_noise_reduction = True
```

### 3. 조기 종료

```python
# 조기 종료 설정
training_config.early_stopping_patience = 5
training_config.early_stopping_threshold = 0.001
```

## 🐛 문제 해결

### 일반적인 문제들

1. **CUDA 메모리 부족**
   ```python
   training_config.batch_size = 4  # 배치 크기 줄이기
   training_config.fp16 = True     # Mixed precision 사용
   ```

2. **오디오 로드 실패**
   ```python
   # 지원되는 형식: .wav, .mp3, .m4a, .flac
   # librosa와 soundfile 설치 확인
   pip install librosa soundfile
   ```

3. **모델 로드 실패**
   ```python
   # 인터넷 연결 확인 (Hugging Face 모델 다운로드)
   # 또는 로컬에 모델 캐시 확인
   ```

### 로깅 활성화

```python
from SER.utils import setup_logging

# 디버그 로깅 활성화
logger = setup_logging(log_level="DEBUG", log_file="ser_debug.log")
```

## 📚 참고자료

- [Wav2Vec2 논문](https://arxiv.org/abs/2006.11477)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [librosa 문서](https://librosa.org/doc/latest/)

## 🤝 기여하기

버그 리포트나 기능 제안은 이슈로 등록해주세요.

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

**참고**: 실제 데이터셋이 준비되면 `data_loader.py`의 `prepare_sample_data()` 함수를 수정하여 실제 데이터 경로를 설정해주세요.
