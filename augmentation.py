from audiomentations import Compose, PitchShift, TimeStretch, AddGaussianNoise, Shift, Gain
from config import config

def get_augmentation_pipeline(scale: float = 1.0):
    """데이터 증강 파이프라인 생성
    
    Args:
        scale: 증강 강도 조절 (0.0 ~ 1.0)
    """
    if not config.data.data_augmentation:
        return None
    
    return Compose([
        AddGaussianNoise(min_amplitude=0.001 * scale, max_amplitude=0.015 * scale, p=0.3 * scale),
        Gain(min_gain_in_db=-12 * scale, max_gain_in_db=12 * scale, p=0.2 * scale),
        PitchShift(min_semitones=-4 * scale, max_semitones=4 * scale, p=0.3 * scale),
        TimeStretch(
            min_rate=0.98 if scale < 1 else 0.97,
            max_rate=1.02 if scale < 1 else 1.03, 
            p=0.15 * scale
        ),
    ])