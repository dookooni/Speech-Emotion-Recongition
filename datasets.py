from torch.utils.data import Dataset
from typing import List, Dict
from data_utils import *
from audiomentations import Compose, PitchShift, TimeStretch, AddGaussianNoise, Shift, Gain, RoomSimulator, HighPassFilter, LowPassFilter
import json
import sys
import torch
import librosa
from pathlib import Path

from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
    Wav2Vec2Config,
    AutoTokenizer
)

from config import config

# Use the shared config's vocabulary data
CHAR2ID = config.char2id
CHAR_VOCAB = config.char_vocab  
ID2CHAR = config.id2char

SER_ROOT = Path(__file__).parent

# Audio Augmentation -> audiomentaions 오류 지속 발생 .. ?
AUGMENTATION = Compose([
    RoomSimulator(p=0.20 * 0.7),
    HighPassFilter(min_cutoff_freq=60, 
                   max_cutoff_freq=120, 
                   p=0.15 * 0.7),
    LowPassFilter(min_cutoff_freq=3500, 
                  max_cutoff_freq=6000, 
                  p=0.15 * 0.7),

    AddGaussianNoise(min_amplitude=0.001, 
                     max_amplitude=0.006,
                     p=0.35 * 0.7),

    Gain(min_gain_in_db=-2.0,
         max_gain_in_db= 2.0, 
         p=0.35 * 0.7),

    Shift(min_shift=-0.03,
          max_shift= 0.03, p=0.35 * 0.7),

    PitchShift(min_semitones=-1, max_semitones=1, p=0.20 * 0.7),
    TimeStretch(min_rate=0.98,
                max_rate=1.02, 
                p=0.15 * 0.7),
])


def preprocess_audio(file_path: str, processor: Wav2Vec2Processor, is_training: bool = False) -> Optional[torch.Tensor]:
    """오디오 전처리"""
    try:
        # 오디오 로드
        audio, sr = librosa.load(file_path, sr=config.model.sample_rate)

        if is_training:
            audio = AUGMENTATION(samples=audio, sample_rate=sr)
       
        # 길이 조정
        target_length = int(config.model.sample_rate * config.model.max_duration)
        if len(audio) > target_length:
            start_idx = np.random.randint(0, len(audio) - target_length + 1)
            audio = audio[start_idx:start_idx + target_length]
        elif len(audio) < target_length:
            pad_length = target_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')

        inputs = processor(audio, sampling_rate=config.model.sample_rate, return_tensors="pt", padding=True)
        return inputs.input_values.squeeze(0)
    except Exception as e:
        print(f"오디오 전처리 오류: {file_path}, {e}")
        return None

class EmotionDataset(Dataset):
    def __init__(self, audio_paths: List[str], labels: List[str], processor: Wav2Vec2Processor, is_training: bool = True, config=None):
        if config is None:
            from config import config as global_config
            config = global_config
            
        self.data_dir = "/data/ghdrnjs/SER/large/large"
        self.audio_paths = audio_paths
        self.labels = labels
        self.processor = processor
        self.config = config
        self.encoded_labels = [self.config.model.label2id[label] for label in labels]
        self.is_training = is_training

        with open(SER_ROOT / "script.json", "r", encoding="utf-8") as f:
            self.text_json = json.load(f)

        self.spk2id = build_speaker_mapping(audio_paths, self.data_dir)        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        emotion_label = self.encoded_labels[idx]
        file_number = extract_number_from_filename(audio_path, type="content")
        
        content_text = ""
        if file_number is not None and str(file_number) in self.text_json:
            content_text = self.text_json[str(file_number)]
        
        input_values = preprocess_audio(audio_path, self.processor, self.is_training)
        if input_values is None:
            input_values = torch.zeros(int(config.model.sample_rate * config.model.max_duration))
        
        spk_idx_tensor = None
        if self.spk2id is not None:
            spk_str = extract_speaker_id(audio_path, self.data_dir)
            if spk_str in self.spk2id:
                spk_idx = self.spk2id[spk_str]
                spk_idx_tensor = torch.tensor(spk_idx, dtype=torch.long)

        return {
            'input_values': input_values,
            'emotion_labels': torch.tensor(emotion_label, dtype=torch.long),
            'content_text': content_text,
            'speaker_id': spk_idx_tensor
        }


class EmotionDataset_Text(Dataset):
    def __init__(self, audio_paths: List[str], labels: List[str], sentences: List[str], processor: Wav2Vec2Processor, is_training: bool = True, config=None):
        if config is None:
            from config import config as global_config
            config = global_config
            
        self.data_dir = "/data/ghdrnjs/SER/large/large"
        self.audio_paths = audio_paths
        self.labels = labels
        self.processor = processor
        self.config = config
        self.encoded_labels = [self.config.model.label2id[label] for label in labels]
        self.transcripts = sentences
        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        self.is_training = is_training

        with open(SER_ROOT / "script.json", "r", encoding="utf-8") as f:
            self.text_json = json.load(f)

        self.spk2id = build_speaker_mapping(audio_paths, self.data_dir) 

    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        emotion_label = self.encoded_labels[idx]
        text = self.transcripts[idx]
        file_number = extract_number_from_filename(audio_path, type="content")
        
        content_text = ""
        if file_number is not None and str(file_number) in self.text_json:
            content_text = self.text_json[str(file_number)]
        
        input_values = preprocess_audio(audio_path, self.processor, self.is_training)
        if input_values is None:
            input_values = torch.zeros(int(config.model.sample_rate * config.model.max_duration))
        
        spk_idx_tensor = None
        if self.spk2id is not None:
            spk_str = extract_speaker_id(audio_path, self.data_dir)
            if spk_str in self.spk2id:
                spk_idx = self.spk2id[spk_str]
                spk_idx_tensor = torch.tensor(spk_idx, dtype=torch.long)

        tokenized_text = self.tokenizer(
            text,
            padding='max_length',    
            truncation=True,         
            max_length=30,          
            return_tensors="pt" 
        )

        return {
            'input_values': input_values,
            'emotion_labels': torch.tensor(emotion_label, dtype=torch.long),
            'content_text': content_text,
            "input_ids": tokenized_text['input_ids'].squeeze(), 
            "text_attention_mask": tokenized_text['attention_mask'].squeeze(),
            'speaker_id': spk_idx_tensor
        }


def collate_fn(batch: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
    input_values = [item['input_values'] for item in batch]
    emotion_labels = [item['emotion_labels'] for item in batch]
    content_texts = [item['content_text'] for item in batch]
    spk_list = [item.get('speaker_id', None) for item in batch]
    
    padded_input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True, padding_value=0.0)
    
    tokenized_contents = []
    content_lengths = []
    for text in content_texts:
        ids = [CHAR2ID.get(char, CHAR2ID['<unk>']) for char in text]
        tokenized_contents.append(torch.tensor(ids, dtype=torch.long))
        content_lengths.append(len(ids))

    padded_content_labels = torch.nn.utils.rnn.pad_sequence(
        tokenized_contents, 
        batch_first=True, 
        padding_value=CHAR2ID['<pad>']
    )

    # attention_mask for audio
    attention_mask = torch.ones_like(padded_input_values, dtype=torch.long)
    for i, seq in enumerate(input_values):
        attention_mask[i, len(seq):] = 0

    if all((s is not None) and isinstance(s, torch.Tensor) for s in spk_list):
        # 각 요소가 0-dim long tensor라면 stack -> (B,)
        speaker_ids = torch.stack(spk_list)            # shape: (B,)
        speaker_ids = speaker_ids.view(-1).long()      # 보정
    else:
        speaker_ids = None

    return {
        'input_values': padded_input_values,
        'attention_mask': attention_mask,
        'labels': torch.stack(emotion_labels),
        'content_labels': padded_content_labels,
        'content_labels_lengths': torch.tensor(content_lengths, dtype=torch.long),
        'speaker_ids': speaker_ids,
    }        



def collate_fn_text(batch: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
    input_values = [item['input_values'] for item in batch]
    emotion_labels = [item['emotion_labels'] for item in batch]
    content_texts = [item['content_text'] for item in batch]
    spk_list = [item.get('speaker_id', None) for item in batch]
    input_ids = [item['input_ids'] for item in batch]
    text_attention_masks = [item['text_attention_mask'] for item in batch]
    
    padded_input_values = torch.nn.utils.rnn.pad_sequence(
        input_values, 
        batch_first=True, 
        padding_value=0.0
    )

    # attention_mask for audio
    attention_mask = torch.ones_like(padded_input_values, dtype=torch.long)
    for i, seq in enumerate(input_values):
        attention_mask[i, len(seq):] = 0

    if all((s is not None) and isinstance(s, torch.Tensor) for s in spk_list):
        # 각 요소가 0-dim long tensor라면 stack -> (B,)
        speaker_ids = torch.stack(spk_list)            # shape: (B,)
        speaker_ids = speaker_ids.view(-1).long()      # 보정
    else:
        speaker_ids = None

    return {
        'input_values': padded_input_values,
        'attention_mask': attention_mask,
        'labels': torch.stack(emotion_labels),
        'speaker_ids': speaker_ids,
        'input_ids': torch.stack(input_ids),
        'text_attention_mask': torch.stack(text_attention_masks),
    }       