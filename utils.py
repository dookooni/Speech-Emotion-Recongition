import os
import json
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from config import config

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """로깅 설정"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 로그 레벨 설정
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # 핸들러 설정
    handlers = [logging.StreamHandler()]
    if log_file:
        log_path = config.paths.log_dir / log_file
        handlers.append(logging.FileHandler(log_path, encoding='utf-8'))
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        force=True
    )
    return logging.getLogger(__name__)

def save_config_to_file(config_dict: Dict[str, Any], file_path: Path):
    """설정을 파일로 저장"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

def validate_audio_files(file_paths: List[str]) -> Tuple[List[str], List[str]]:
    """오디오 파일 경로 검증"""
    valid_files = []
    invalid_files = []
    
    for file_path in file_paths:
        # 파일 존재 확인
        if not os.path.exists(file_path):
            invalid_files.append(f"{file_path} (파일 없음)")
            continue
        
        # 확장자 확인
        _, ext = os.path.splitext(file_path.lower())
        if ext not in config.data.supported_formats:
            invalid_files.append(f"{file_path} (지원되지 않는 형식)")
            continue
        
        # 파일 크기 확인
        if os.path.getsize(file_path) == 0:
            invalid_files.append(f"{file_path} (빈 파일)")
            continue
        
        valid_files.append(file_path)
    
    return valid_files, invalid_files

class CheckpointManager:
    """체크포인트 관리 클래스"""
    
    def __init__(self, save_best_only: bool = False):
        self.checkpoint_dir = config.paths.checkpoint_dir
        self.save_best_only = save_best_only
        self.best_score = -float('inf')
    
    def save(self, epoch: int, model, optimizer, score: float, is_best: bool = False, extra_info: Dict = None):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'score': score,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'model': config.model.__dict__,
                'training': config.training.__dict__
            }
        }
        
        if extra_info:
            checkpoint.update(extra_info)
        
        # 최신 체크포인트 저장
        if not self.save_best_only:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
        
        # 최고 성능 체크포인트 저장
        if is_best or score > self.best_score:
            self.best_score = score
            best_path = self.checkpoint_dir / 'best_checkpoint.pt'
            torch.save(checkpoint, best_path)
            print(f"최고 성능 체크포인트 저장: {best_path} (score: {score:.4f})")
    
    def load(self, checkpoint_path: Path, model, optimizer=None):
        """체크포인트 로드"""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint.get('epoch', 0), checkpoint.get('score', 0.0)