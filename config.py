import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

# SER 폴더의 루트 경로
SER_ROOT = Path(__file__).parent
print(f"SER_ROOT: {SER_ROOT}")
PROJECT_ROOT = SER_ROOT.parent
DATA_ROOT = Path("/data/ghdrnjs/SER")

@dataclass
class ModelConfig:
    model_name: str = "kresnik/wav2vec2-large-xlsr-korean"
    w2v2_model_name: str = "kresnik/wav2vec2-large-xlsr-korean"
    prosody_model_name: str = "speechbrain/spkrec-ecapa-voxceleb"
    emotion_labels: List[str] = field(default_factory=lambda: ["Anxious", "Dry", "Kind", "Other"])
    sample_rate: int = 16000
    max_duration: float = 10.0
    num_labels: int = field(init=False)
    label2id: Dict[str, int] = field(init=False)
    id2label: Dict[int, str] = field(init=False)
    
    def __post_init__(self):
        self.num_labels = len(self.emotion_labels)
        self.label2id = {label: i for i, label in enumerate(self.emotion_labels)}
        self.id2label = {i: label for i, label in enumerate(self.emotion_labels)}

@dataclass
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 3e-5
    num_epochs: int = 20
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True

@dataclass
class DataConfig:
    data_augmentation: bool = True
    apply_noise_reduction: bool = True
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    supported_formats: List[str] = field(default_factory=lambda: ['.wav', '.mp3', '.m4a', '.flac', '.ogg'])
    min_audio_length: float = 0.5  # seconds
    max_audio_length: float = 10.0  # seconds
    # Large 데이터셋 설정 추가
    use_large_dataset: bool = True
    large_balance_ratio: float = 0.8 

@dataclass
class PathConfig:
    ser_root: Path = SER_ROOT
    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: DATA_ROOT / "small")
    large_data_dir: Path = field(default_factory=lambda: DATA_ROOT / "large" / "large")  # Large 데이터셋 경로 추가
    output_dir: Path = field(default_factory=lambda: SER_ROOT / "results")
    cache_dir: Path = field(default_factory=lambda: SER_ROOT / "cache")
    log_dir: Path = field(default_factory=lambda: SER_ROOT / "logs")
    checkpoint_dir: Path = field(default_factory=lambda: SER_ROOT / "checkpoints")
    char_vocab_file: Path = field(default_factory=lambda: SER_ROOT / "char_to_id.json")
    
    def __post_init__(self):
        # 필요한 디렉터리 생성
        for path in [self.data_dir, self.large_data_dir, self.output_dir, self.cache_dir, self.log_dir, self.checkpoint_dir]:
            path.mkdir(parents=True, exist_ok=True)

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    char_vocab: Optional[List[str]] = None
    char2id: Optional[Dict[str, int]] = None
    id2char: Optional[Dict[int, str]] = None
    
    def __post_init__(self):
        self._load_char_vocab()
    
    def _load_char_vocab(self):
        """Character Vocabulary 로드"""
        if self.paths.char_vocab_file.exists():
            try:
                with open(self.paths.char_vocab_file, 'r', encoding='utf-8') as f:
                    self.char2id = json.load(f)
                self.char_vocab = list(self.char2id.keys())
                self.id2char = {i: char for char, i in self.char2id.items()}
                print(f"✅ Character Vocabulary 로드 완료 ({len(self.char_vocab)}개)")
            except Exception as e:
                print(f"⚠️ Character Vocabulary 로드 실패: {e}")
        else:
            print("⚠️ char_to_id.json을 찾을 수 없습니다. build_vocab.py를 실행하세요.")
    
    def save_to_file(self, file_path: Path):
        """설정을 파일로 저장"""
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'paths': {k: str(v) for k, v in self.paths.__dict__.items()}
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, file_path: Path) -> 'Config':
        """파일에서 설정 로드"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # TODO: 파일에서 로드한 설정으로 Config 객체 생성
        return cls()

# 전역 설정 인스턴스
config = Config()

# 하위 호환성을 위한 별칭
SAMPLE_RATE = config.model.sample_rate
MAX_DURATION = config.model.max_duration
EMOTION_LABELS = config.model.emotion_labels
LABEL2ID = config.model.label2id
ID2LABEL = config.model.id2label