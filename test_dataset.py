#!/usr/bin/env python3
"""
Large 데이터셋에 대한 모델 테스트 스크립트
학습된 모델을 로드하여 Classification Report를 생성합니다.
"""

import os
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
import warnings
warnings.filterwarnings('ignore')
import json

# 프로젝트 모듈 import
from config import Config
from data_utils import (
    build_large_corpus_index, 
    split_large_dataset, 
    EMOTION_LABELS
)
from torch.utils.data import DataLoader
from datasets import collate_fn, EmotionDataset
from data_utils import balance_by_undersampling_majority
from Wav2Vec2_seq_clf import custom_Wav2Vec2ForEmotionClassification, custom_Wav2Vec2ForEmotionClassification_Text
from transformers import Wav2Vec2Config, Wav2Vec2Processor


def load_model_and_processor(checkpoint_path: str, config: Config, device: torch.device, num_speakers: int = 500):
    """
    학습된 모델과 프로세서 로드 - create_model_and_processor 기반
    - Hugging Face safetensors (.safetensors)
    - PyTorch checkpoint (.pth, .pt)
    - Hugging Face 모델 디렉토리
    """
    print(f"📂 모델 로드 중: {checkpoint_path}")
    
    # 설정 수정
    model_config = Wav2Vec2Config.from_pretrained(
        config.model.model_name,
        num_labels=config.model.num_labels,
        label2id=config.model.label2id,
        id2label=config.model.id2label,
        finetuning_task="emotion_classification"
    )
    model_config.num_speakers = num_speakers  # 화자 수 설정
    
    if config.char_vocab:
        model_config.char_vocab_size = len(config.char_vocab)
    
    # 모델 초기화 (사전 훈련된 가중치 없이)
    model = custom_Wav2Vec2ForEmotionClassification.from_pretrained(
        config.model.model_name,
        config=model_config,
        ignore_mismatched_sizes=True
    )
    
    # 프로세서 로드
    processor = Wav2Vec2Processor.from_pretrained(config.model.model_name)
    
    # 디바이스로 이동
    model = model.to(device)
    
    # 체크포인트가 제공된 경우 가중치 로드
    if checkpoint_path and os.path.exists(checkpoint_path):
        # 경로가 디렉토리인지 파일인지 확인
        if os.path.isdir(checkpoint_path):
            # Hugging Face 모델 디렉토리인 경우
            try:
                print("🤗 Hugging Face 모델 디렉토리에서 가중치 로드 중...")
                
                # 새로운 모델을 해당 디렉토리에서 로드
                loaded_model = custom_Wav2Vec2ForEmotionClassification.from_pretrained(
                    checkpoint_path,
                    config=model_config,
                    ignore_mismatched_sizes=True,
                    local_files_only=True
                )
                
                model = loaded_model.to(device)
                print("✅ Hugging Face 모델 디렉토리에서 로드 완료")
                
            except Exception as e:
                print(f"❌ Hugging Face 디렉토리 로드 실패: {e}")
                print("🔄 기본 가중치로 계속 진행합니다.")
                
        elif checkpoint_path.endswith('.safetensors'):
            # safetensors 파일인 경우
            try:
                from safetensors.torch import load_file
                print("🔒 SafeTensors 파일에서 로드 중...")
                
                state_dict = load_file(checkpoint_path, device=str(device))
                
                # 키 이름 확인 및 매핑
                if any(key.startswith('wav2vec2.') for key in state_dict.keys()):
                    # 이미 올바른 키 구조인 경우
                    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                else:
                    # 키 구조 변경이 필요한 경우
                    mapped_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('model.'):
                            new_key = key.replace('model.', '')
                            mapped_state_dict[new_key] = value
                        else:
                            mapped_state_dict[key] = value
                    missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)
                
                if missing_keys:
                    print(f"⚠️  누락된 키: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
                if unexpected_keys:
                    print(f"⚠️  예상치 못한 키: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
                    
                print("✅ SafeTensors 파일에서 로드 완료")
                
            except ImportError:
                print("❌ safetensors 라이브러리가 설치되지 않았습니다. pip install safetensors")
                print("🔄 기본 가중치로 계속 진행합니다.")
            except Exception as e:
                print(f"❌ SafeTensors 로드 실패: {e}")
                print("🔄 기본 가중치로 계속 진행합니다.")
                
        else:
            # PyTorch 체크포인트 파일인 경우 (.pth, .pt)
            try:
                print("⚡ PyTorch 체크포인트에서 로드 중...")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # 다양한 체크포인트 형식 지원
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    epoch_info = checkpoint.get('epoch', 'unknown')
                    print(f"📊 체크포인트 정보: epoch {epoch_info}")
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif isinstance(checkpoint, dict) and any('wav2vec2' in key for key in checkpoint.keys()):
                    # 직접 state_dict인 경우
                    state_dict = checkpoint
                else:
                    print("❌ 지원하지 않는 체크포인트 형식입니다.")
                    print(f"사용 가능한 키: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'dict가 아님'}")
                    print("🔄 기본 가중치로 계속 진행합니다.")
                    state_dict = None
                
                if state_dict is not None:
                    # 키 이름 매핑 (필요시)
                    mapped_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('module.'):  # DataParallel 사용 시
                            new_key = key.replace('module.', '')
                            mapped_state_dict[new_key] = value
                        else:
                            mapped_state_dict[key] = value
                    
                    # 모델에 로드
                    missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)
                    
                    if missing_keys:
                        print(f"⚠️  누락된 키: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
                    if unexpected_keys:
                        print(f"⚠️  예상치 못한 키: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
                        
                    print("✅ PyTorch 체크포인트에서 로드 완료")
                
            except Exception as e:
                print(f"❌ PyTorch 체크포인트 로드 실패: {e}")
                print("🔄 기본 가중치로 계속 진행합니다.")
    else:
        print("📦 체크포인트가 없습니다. 사전 훈련된 가중치를 사용합니다.")
    
    model.eval()
    print(f"📱 모델이 {device}에 로드되었습니다.")
    return model, processor


def evaluate_model(model, test_loader, device, emotion_labels):
    """모델 평가 및 예측 결과 수집"""
    model.eval()
    all_predictions = []
    all_true_labels = []
    all_probabilities = []
    
    print("🔍 모델 평가 중...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # 배치 데이터를 device로 이동
            input_values = batch['input_values'].to(device)
            labels = batch['labels'].to(device)
            
            # 모델 예측
            outputs = model(input_values)
            # logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            logits = outputs.get("emotion_logits")
            
            # 예측 결과 수집
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_true_labels), np.array(all_probabilities)


def generate_classification_report(y_true, y_pred, emotion_labels, save_dir=None):
    """Classification Report 생성 및 저장"""
    
    # Classification Report 생성
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=emotion_labels,
        digits=4,
        output_dict=True
    )
    
    # 텍스트 형태 리포트
    text_report = classification_report(
        y_true, 
        y_pred, 
        target_names=emotion_labels,
        digits=4
    )
    
    print("📊 Classification Report:")
    print("=" * 60)
    print(text_report)
    print("=" * 60)
    
    # 저장
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # JSON 형태로 저장
        with open(os.path.join(save_dir, 'classification_report.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 텍스트 형태로 저장
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        print(f"💾 Classification Report 저장 완료: {save_dir}")
    
    return report

def plot_confusion_matrix(y_true, y_pred, emotion_labels, save_dir=None):
    """Confusion Matrix 시각화"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=emotion_labels,
        yticklabels=emotion_labels
    )
    plt.title('Confusion Matrix - Large Dataset Test')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        print(f"💾 Confusion Matrix 저장 완료: {save_dir}/confusion_matrix.png")
    
    plt.show()
    
    return cm


def main():
    parser = argparse.ArgumentParser(description='Large 데이터셋 모델 테스트')
    parser.add_argument('--checkpoint', type=str, 
                       help='모델 체크포인트 경로 (없으면 데이터만 확인)')
    parser.add_argument('--config', type=str, default='config.py',
                       help='설정 파일 경로')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--data_only', action='store_true',
                       help='데이터 로드 테스트만 실행')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = Config()
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 사용 디바이스: {device}")
    
    # Large 데이터셋 로드
    print("📊 Large 데이터셋 로드 중...")
    large_index = build_large_corpus_index(
        str(config.paths.large_data_dir), 
        with_sentence=True
    )
    index = balance_by_undersampling_majority(large_index)
    if not large_index:
        print("❌ Large 데이터셋을 찾을 수 없습니다.")
        return
    
    # 데이터 분할 (테스트 세트만 필요)
    print("🔄 데이터 분할 중...")
    (train_paths, train_labels, train_sentences), \
    (val_paths, val_labels, val_sentences), \
    (test_paths, test_labels, test_sentences) = split_large_dataset(
        index,
        val_speaker_ratio=0.2,
        test_speaker_ratio=0.2,
        val_content_ratio=0.2,
        test_content_ratio=0.2,
        with_sentence=True
    )
    
    print(f"📋 테스트 데이터: {len(test_paths)}개 샘플")
    
    # 데이터만 확인하고 종료
    if args.data_only or not args.checkpoint:
        print("✅ 데이터 로드 테스트 완료!")
        print("💡 모델 테스트를 원하면 --checkpoint 옵션을 사용하세요.")
        return
    
    # 체크포인트 파일 존재 확인
    if not os.path.exists(args.checkpoint):
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {args.checkpoint}")
        return
    
    # 모델과 프로세서 로드
    model, processor = load_model_and_processor(args.checkpoint, config, device)
    
    if model is None:
        print("❌ 모델 로드 실패")
        return
    
    # 데이터셋 생성
    test_dataset = EmotionDataset(test_paths, test_labels, processor, is_training=False)

    # 데이터 로더 생성
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )
    
    # 모델 평가
    y_pred, y_true, probabilities = evaluate_model(model, test_loader, device, EMOTION_LABELS)
    
    # Classification Report 생성
    report = generate_classification_report(y_true, y_pred, EMOTION_LABELS, args.output_dir)
    
    # Confusion Matrix 생성
    cm = plot_confusion_matrix(y_true, y_pred, EMOTION_LABELS, args.output_dir)
    
    # 추가 통계 정보
    print("\n📈 추가 통계 정보:")
    print(f"전체 테스트 샘플: {len(y_true)}")
    print(f"정확도: {(y_pred == y_true).sum() / len(y_true):.4f}")
    print(f"report : {report}")
    print(f"confusion matrix : {cm}")
    
    # 클래스별 샘플 수
    from collections import Counter
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    
    print("\n📊 클래스별 분포:")
    for i, emotion in enumerate(EMOTION_LABELS):
        print(f"{emotion}: 실제 {true_counts[i]}, 예측 {pred_counts[i]}")
    
    print(f"\n✅ 테스트 완료! 결과는 {args.output_dir}에 저장되었습니다.")


if __name__ == "__main__":
    main()
