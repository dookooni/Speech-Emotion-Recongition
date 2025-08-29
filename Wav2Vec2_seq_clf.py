import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import Wav2Vec2ForSequenceClassification, AutoModel
from speechbrain.pretrained import EncoderClassifier
from adversary import ContentAdversary, SpeakerAdversary, AttentivePool
import json

from config import config

# Use the shared config's vocabulary data
CHAR2ID = config.char2id
CHAR_VOCAB = config.char_vocab  
ID2CHAR = config.id2char
CTC_BLANK_ID = CHAR2ID["<ctc_blank>"]

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
        
        # 2 계층 분류기 추가
        # self.classifier = nn.Sequential(
        #     nn.Linear(config.classifier_proj_size, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(128, config.num_labels)
        # )

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
        
        if speaker_ids is not None:
            print(f"✅ Speaker IDs received! Shape: {speaker_ids.shape}, Device: {speaker_ids.device}, Speaker IDs: {speaker_ids}")
        else:
            print("❌ Warning: speaker_ids is None. Speaker adversary will be skipped.")

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state

        # 1. 감정 분류
        # pooled_output = torch.mean(hidden_states, dim=1)
        # if hasattr(self, 'projector'):
        #     pooled_output = self.projector(pooled_output)
        pooled_output = self.pooler(hidden_states)
        pooled_output = self.stats_projector(pooled_output)
        pooled_output = self.projector(pooled_output)
        emotion_logits = self.classifier(pooled_output)

        # 2. 내용 분류 (적대자)
        spk_logits = None
        loss = None
        if labels is not None and class_weights is not None:
            # 3. 손실 계산
            loss_emotion_fct = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
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
    
class custom_Wav2Vec2ForEmotionClassification_Text(Wav2Vec2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.speaker_adversary = SpeakerAdversary(
            input_size = config.hidden_size,
            num_speakers = config.num_speakers
        )
        self.pooler = AttentivePool(config.hidden_size)
        self.stats_projector = nn.Linear(2 * config.hidden_size,  # 2D -> D
                                         config.hidden_size)
        self.nlp = AutoModel.from_pretrained("klue/bert-base")

        audio_dim = config.classifier_proj_size
        text_dim = self.nlp.config.hidden_size
        combined_dim = audio_dim + text_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, config.num_labels)
        )
        

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        adv_lambda=1.0,
        speaker_ids=None,
        class_weights = None,
        input_ids=None,
        text_attention_mask=None,
    ):
        
        if speaker_ids is not None:
            print(f"✅ Speaker IDs received! Shape: {speaker_ids.shape}, Device: {speaker_ids.device}, Speaker IDs: {speaker_ids}")
        else:
            print("❌ Warning: speaker_ids is None. Speaker adversary will be skipped.")

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
        audio_features = self.projector(pooled_output)
        # emotion_logits = self.classifier(audio_features)

        if input_ids is not None:
            nlp_outputs = self.nlp(
                input_ids=input_ids,
                attention_mask=text_attention_mask
            )
            # [CLS] 토큰의 출력을 문장 전체의 특징으로 사용
            text_features = nlp_outputs.last_hidden_state[:, 0, :]
        else:
            # 텍스트 입력이 없으면 0으로 채워진 텐서 생성 (에러 방지)
            text_features = torch.zeros_like(audio_features)
        combined_features = torch.cat((audio_features, text_features), dim=1)
        emotion_logits = self.classifier(combined_features)

        # 2. 내용 분류 (적대자)
        spk_logits = None
        loss = None
        if labels is not None and class_weights is not None:
            # 3. 손실 계산
            loss_emotion_fct = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
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


class Wav2Vec2_CE(Wav2Vec2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
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

        # 2. 내용 분류
        loss = None
        if labels is not None:
            loss_emotion_fct = nn.CrossEntropyLoss()
            loss_emotion = loss_emotion_fct(
                emotion_logits.view(-1, self.config.num_labels), labels.view(-1)
            )
            loss = loss_emotion

        return {
            "loss": loss, 
            "emotion_logits": emotion_logits, 
        }

class HybridEmotionModel(nn.Module):
    def __init__(self, w2v2_model_name, prosody_model_name, num_labels, freeze_w2v2=True, freeze_prosody=True):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 🧠 1. 내용 전문가 (Content Expert): Wav2Vec2
        self.w2v2 = Wav2Vec2ForSequenceClassification.from_pretrained(w2v2_model_name)
        self.pooler = AttentivePool(self.w2v2.config.hidden_size)
        
        # 🎶 2. 프로소디 전문가 (Prosody Expert): Speaker Recognition Model
        self.prosody_expert = EncoderClassifier.from_hparams(source=prosody_model_name, savedir=f"pretrained_models/{prosody_model_name.split('/')[-1]},",)


        # --- 모델 동결(Freeze) 설정 ---
        if freeze_w2v2:
            for param in self.w2v2.parameters():
                param.requires_grad = False
        if freeze_prosody:
            for param in self.prosody_expert.parameters():
                param.requires_grad = False

        # 🚀 3. 최종 분류기 (Final Classifier)
        # Wav2Vec2의 AttentivePool 출력 차원 + 프로소디 모델의 출력 차원
        # Wav2Vec2-Large hidden_size: 1024 -> AttentivePool 출력: 2048
        # ECAPA-TDNN 출력: 192
        context_dim = self.w2v2.config.hidden_size * 2 
        prosody_dim = self.prosody_expert.mods.embedding_model.fc.conv.out_channels 
        
        self.classifier = nn.Sequential(
            nn.Linear(context_dim + prosody_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_values, attention_mask=None, labels=None, class_weights=None):
        input_values = input_values.to(self.device)

        # 1. 내용 특징 추출
        context_outputs = self.w2v2(input_values, attention_mask=attention_mask, 
                                    output_hidden_states=True)
        last_hidden_state = context_outputs.hidden_states[-1]
        context_features = self.pooler(last_hidden_state)
        
        # 2. 프로소디 특징 추출
        # prosody_features = self.prosody_expert.encode_batch(input_values).squeeze(1)
        with torch.no_grad(): # 프로소디 모델이 동결(freeze) 상태이므로 그래디언트 계산 비활성화
            feats = self.prosody_expert.mods.compute_features(input_values)
            prosody_features = self.prosody_expert.mods.embedding_model(feats).squeeze(1)

        
        # 3. 두 특징을 결합
        combined_features = torch.cat([context_features, prosody_features], dim=-1)
        
        # 4. 최종 분류
        logits = self.classifier(combined_features)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits, labels)
            
        return {"loss": loss, 
                "logits": logits}
