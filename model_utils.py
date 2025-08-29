def enable_last_k_blocks(model, last_k: int = 4):
    for p in model.wav2vec2.parameters():
        p.requires_grad = False
    
    layers = model.wav2vec2.encoder.layers
    num_layers = len(layers)
    for i in range(num_layers - last_k, num_layers):
        for p in layers[i].parameters():
            p.requires_grad = True
    
    for name, module in model.named_modules():
        if any(k in name for k in ["classifier", "adversary", "speaker_adversary", "pooler", "stats_projector", "projector"]):
            for p in module.parameters():
                p.requires_grad = True

def enable_hybrid_last_k_blocks(model, last_k: int = 4):
    """하이브리드 모델의 내용 전문가(Wav2Vec2)의 마지막 K개 블록만 학습 가능하도록 설정합니다."""
    for param in model.w2v2.parameters():
        param.requires_grad = False
    
    layers_to_unfreeze = model.w2v2.wav2vec2.encoder.layers[-last_k:]
    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True
            
    for module in [model.pooler, model.classifier]:
        for param in module.parameters():
            param.requires_grad = True
            
    print(f"✅ Hybrid Model: Wav2Vec2의 마지막 {last_k}개 block, Pooler, Classifier만 학습합니다.")