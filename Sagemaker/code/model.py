import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Function
from transformers import Wav2Vec2ForSequenceClassification
import json


class custom_Wav2Vec2ForEmotionClassification(Wav2Vec2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.adversary = ContentAdversary(
            input_size=config.hidden_size, 
            num_chars=config.char_vocab_size 
        )
        self.speaker_adversary = SpeakerAdversary(
            input_size = config.hidden_size,
            num_speakers = config.num_speakers
        )
        self.pooler = AttentivePool(config.hidden_size)
        self.stats_projector = nn.Linear(2 * config.hidden_size,  # 2D -> D
                                   config.hidden_size)

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        content_labels=None, 
        content_labels_lengths=None,
        adv_lambda=1.0,
        speaker_ids=None,
        class_weights = None,
    ):

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state

        # 1. 감정 분류
        pooled_output = self.pooler(hidden_states)
        pooled_output = self.stats_projector(pooled_output)
        pooled_output = self.projector(pooled_output)
        emotion_logits = self.classifier(pooled_output)

        # 2. 내용 분류 (적대자)
        spk_logits = None
        loss = None
        if labels is not None and content_labels is not None:
            # 3. 손실 계산
            loss_emotion_fct = nn.CrossEntropyLoss(class_weights=class_weights)
            loss_emotion = loss_emotion_fct(emotion_logits.view(-1, self.config.num_labels), labels.view(-1))
            loss = loss_emotion

        if adv_lambda > 0 and speaker_ids is not None:
            spk_logits = self.speaker_adversary(hidden_states, adv_lambda)
            loss_spk = F.cross_entropy(spk_logits, speaker_ids)
            loss = loss + adv_lambda * loss_spk if loss is not None else adv_lambda * loss_spk

        return {
            "loss": loss, 
            "emotion_logits": emotion_logits, 
        }

class GradientReversalLayer(Function):
    """
    Gradient Reversal Layer (GRL)
    - Forward pass: 항등 함수 (입력을 그대로 통과)
    - Backward pass: 그래디언트의 부호를 반전시킴
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 그래디언트를 lambda 값만큼 스케일링하고 부호를 반전
        return (grad_output.neg() * ctx.lambda_), None

def grad_reverse(x, lambda_=1.0):
    return GradientReversalLayer.apply(x, lambda_)


class ContentAdversary(nn.Module):
    """
    음성 특징(hidden_state)을 보고 원래 텍스트(문자)를 예측하는 적대자 모델
    """
    def __init__(self, input_size: int, num_chars: int, hidden_size: int = 256):
        super().__init__()
        self.grl = GradientReversalLayer.apply
        self.layer_norm = nn.LayerNorm(input_size)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_chars)

    def forward(self, hidden_states, lambda_):
        # hidden_states: (batch, seq_len, 1024)
        
        # 1. Gradient Reversal Layer 통과
        reversed_hidden_states = self.grl(hidden_states, lambda_)
        
        # 2. 간단한 분류기
        x = self.layer_norm(reversed_hidden_states)
        x = self.linear1(x)
        x = self.relu(x)
        logits = self.linear2(x) # (batch, seq_len, num_chars)
        
        return logits

class SpeakerAdversary(nn.Module):
    def __init__(self, input_size, num_speakers, hidden=256):
        super().__init__()
        self.grl = GradientReversalLayer.apply
        self.net = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden), nn.ReLU(),
            nn.Linear(hidden, num_speakers)
        )
    def forward(self, hidden_states, lambda_):
        # frame-level -> utterance-level pooling
        x = hidden_states.mean(dim=1)
        x = self.grl(x, lambda_)
        return self.net(x)

class AttentivePool(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.att = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):  # x: (B,T,D)
        w = self.att(x).squeeze(-1)              # (B,T)
        w = torch.softmax(w, dim=1)
        mean = (x * w.unsqueeze(-1)).sum(dim=1)
        std  = torch.sqrt(((x-mean.unsqueeze(1))**2 * w.unsqueeze(-1)).sum(dim=1) + 1e-5)
        return torch.cat([mean, std], dim=-1)    # (B,2D)