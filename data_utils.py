import numpy as np
from typing import Tuple, List, Optional, Literal, Dict, Any, Union, overload
from collections import Counter
import re
import os
import random
from tqdm import tqdm

EMOTION_LABELS = ["Anxious", "Dry", "Kind", "Other"]

# script.txt 감정 매핑
SCRIPT_EMOTION_MAPPING = {
    "NEUTRAL": "Other",
    "ANXIOUS": "Anxious", 
    "KIND": "Kind",
    "DRY": "Dry"
}

def extract_korean_and_punct(text: str) -> str:
    """
    주어진 문자열에서 한글, 쉼표, 온점 등만 추출 (특수마커 제거)
    예: '목이 마르다.|||HL 다||M 이루었다.' -> '목이 마르다. 다 이루었다.'
    """
    # 한글, 쉼표, 온점, 공백만 남기기
    import re
    # 특수마커 제거: '|||HL', '||M' 등
    text = re.sub(r'\|{2,}\w*', '', text)
    # 한글, 쉼표, 온점, 공백만 남기기
    filtered = re.sub(r'[^가-힣.,?\s]', '', text)
    # 공백 정리
    filtered = re.sub(r'\s+', ' ', filtered).strip()
    return filtered

@overload
def parse_script_file(script_file_path: str, with_sentence: Literal[False] = False) -> Dict[str, str]: ...

@overload  
def parse_script_file(script_file_path: str, with_sentence: Literal[True] = True) -> Tuple[Dict[str, str], Dict[str, str]]: ...

def parse_script_file(script_file_path: str, with_sentence: bool = False) -> Union[Dict[str, str], Tuple[Dict[str, str], Dict[str, str]]]:
    """
    script.txt 파일을 파싱하여 파일명 -> 감정 매핑 딕셔너리 반환
    
    Args:
        script_file_path: script.txt 파일 경로
        with_sentence: True시 문장 매핑도 함께 반환
        
    Returns:
        with_sentence=False: Dict[파일명(확장자 제외), 감정]
        with_sentence=True: (emotion_mapping, sentence_mapping)
    """
    emotion_mapping = {}
    sentence_mapping = {}  # 파일명 -> 문장 매핑
    
    try:
        with open(script_file_path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip() for line in f]
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue
                # F0002_000001 NEUTRAL #지문 형태에서 파일명과 감정만 추출
                if re.match(r'^[FM]\d+_\d+\s+\w+', line):
                    parts = line.split()
                    if len(parts) >= 2:
                        filename = parts[0]
                        emotion_raw = parts[1]
                        emotion = SCRIPT_EMOTION_MAPPING.get(emotion_raw, "Other")
                        emotion_mapping[filename] = emotion
                        
                        # with_sentence=True일 때만 문장 추출
                        if with_sentence and i + 1 < len(lines):
                            orig_line = lines[i + 1].strip()
                            clean_sentence = extract_korean_and_punct(orig_line)
                            sentence_mapping[filename] = clean_sentence
                    i += 2  # 다음 블록으로 이동
                else:
                    i += 1
    except Exception as e:
        print(f"❌ script.txt 파싱 오류: {e}")

    if with_sentence:
        return emotion_mapping, sentence_mapping
    return emotion_mapping

def build_large_corpus_index(data_dir: str,
                            accept_exts={'.wav', '.flac'},
                            max_samples_per_class: Optional[int] = None,
                            with_sentence: bool = False) -> List[Dict[str, Any]]:
    """
    large 데이터셋 전용 인덱스 생성 함수
    /data/ghdrnjs/SER/large/large/F0001,F0002,M0001,M0002... 구조
    각 화자 폴더 안에 script.txt와 wav 파일들이 존재
    """
    index = []
    emotion_counts = {emotion: 0 for emotion in EMOTION_LABELS}
    
    # 화자 폴더들 스캔 (F0001~F0004, M0001~M0004 등)
    speaker_folders = sorted([d for d in os.listdir(data_dir) 
                             if os.path.isdir(os.path.join(data_dir, d)) 
                             and re.match(r'^[FM]\d+$', d)])
    
    if not speaker_folders:
        print(f"❌ 화자 폴더를 찾을 수 없습니다: {data_dir}")
        return []
    
    print(f"� 발견된 화자 폴더: {speaker_folders}")
    
    for speaker in tqdm(speaker_folders, desc="Large 데이터셋 인덱스 구축"):
        speaker_dir = os.path.join(data_dir, speaker)
        
        # 각 화자 폴더 내의 script.txt 파싱
        script_file_path = os.path.join(speaker_dir, "script.txt")
        if not os.path.exists(script_file_path):
            print(f"⚠️ script.txt를 찾을 수 없습니다: {script_file_path}")
            continue
        
        print(f"📖 {speaker} script.txt 파싱 중...")
        if not with_sentence:
            emotion_mapping = parse_script_file(script_file_path)
            sentence_mapping = {}
        else:
            emotion_mapping, sentence_mapping = parse_script_file(script_file_path, with_sentence=True)

        # 해당 화자 폴더에서 wav 파일들 스캔
        wav_files = []
        for root, _, files in os.walk(speaker_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in accept_exts):
                    wav_files.append(os.path.join(root, file))
        
        for audio_path in wav_files:
            # 파일명에서 확장자 제거 (F0002_000001.wav -> F0002_000001)
            filename_no_ext = os.path.splitext(os.path.basename(audio_path))[0]
            
            # script.txt에서 감정 정보 조회
            emotion = emotion_mapping.get(filename_no_ext, "Other")
            
            # 클래스별 최대 샘플 수 제한
            if max_samples_per_class and emotion_counts[emotion] >= max_samples_per_class:
                continue
            
            # 컨텐츠 ID 추출 (F0002_000001 -> 000001)
            content_match = re.search(r'_(\d+)$', filename_no_ext)
            content_id = int(content_match.group(1)) if content_match else 0
            
            # 인덱스 항목 생성
            item = {
                "path": audio_path,
                "emotion": emotion,
                "speaker": speaker,  # 화자 폴더명 사용
                "content_id": content_id,
                "source": "large"
            }
            
            # with_sentence=True일 때 문장 정보 추가
            if with_sentence:
                sentence = sentence_mapping.get(filename_no_ext, "")
                item["sentence"] = sentence
                
            index.append(item)
            emotion_counts[emotion] += 1
    
    print(f"✅ Large 데이터셋 인덱스 완료 - 총 {len(index)}개 샘플")
    print(f"📊 감정별 분포: {dict(emotion_counts)}")
    print(f"👥 화자별 분포: {Counter([item['speaker'] for item in index])}")
    
    return index

def balance_large_dataset(index: List[Dict[str, Any]], 
                         balance_ratio: float = 0.3) -> List[Dict[str, Any]]:
    """
    Large 데이터셋의 클래스 불균형 해결
    Other 클래스가 많으므로 비율 조정
    
    Args:
        index: build_large_corpus_index 결과
        balance_ratio: Other 클래스 대비 다른 클래스들의 비율 (0.3 = Other의 30% 수준)
    """
    emotion_groups = {emotion: [] for emotion in EMOTION_LABELS}
    
    # 감정별로 샘플 그룹핑
    for item in index:
        emotion_groups[item["emotion"]].append(item)
    
    print(f"🎯 클래스 균형 조정 (Other 대비 비율: {balance_ratio})")
    
    # Other 클래스 개수를 기준으로 다른 클래스들 개수 결정
    other_count = len(emotion_groups["Other"])
    target_other_count = other_count  # Other는 그대로 유지하거나 필요시 조정
    target_non_other_count = int(other_count * balance_ratio)
    
    balanced_index = []
    
    for emotion, samples in emotion_groups.items():
        if emotion == "Other":
            # Other는 전체 또는 조정된 수만큼 사용
            selected = samples[:target_other_count] if len(samples) > target_other_count else samples
        else:
            # 다른 감정들은 balance_ratio에 따라 조정
            if len(samples) >= target_non_other_count:
                selected = random.sample(samples, target_non_other_count)
            else:
                selected = samples  # 샘플이 부족하면 모두 사용
        
        balanced_index.extend(selected)
        print(f"  {emotion}: {len(selected)}개 (원본: {len(samples)}개)")
    
    return balanced_index

def balance_by_undersampling_majority(index: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    데이터셋의 클래스 불균형을 해결합니다.
    다수 클래스인 'Other'를 소수 클래스들의 평균 개수에 맞춰 언더샘플링합니다.
    
    Args:
        index: 원본 데이터 인덱스
    """
    emotion_groups = {emotion: [] for emotion in EMOTION_LABELS}
    for item in index:
        emotion_groups[item["emotion"]].append(item)

    # 'Other'를 제외한 소수 클래스들의 평균 샘플 수를 계산
    non_other_counts = [len(samples) for emotion, samples in emotion_groups.items() if emotion != "Other"]
    if not non_other_counts:
        return index # 'Other' 외에 클래스가 없으면 원본 반환
        
    target_count = int(sum(non_other_counts) / len(non_other_counts))
    
    print(f"🎯 클래스 균형 조정 (언더샘플링)")
    print(f"   'Other' 클래스를 다른 클래스 평균 개수인 {target_count}개로 조정합니다.")

    balanced_index = []
    
    # 'Other' 클래스를 목표 개수만큼 랜덤 샘플링
    if 'Other' in emotion_groups and len(emotion_groups['Other']) > target_count:
        other_samples = random.sample(emotion_groups['Other'], target_count)
        balanced_index.extend(other_samples)
        print(f"  Other: {len(other_samples)}개 (원본: {len(emotion_groups['Other'])}개)")
    else:
        # 'Other'가 없거나 이미 목표치보다 적으면 그대로 사용
        balanced_index.extend(emotion_groups.get('Other', []))

    # 'Other'가 아닌 클래스들은 모두 사용
    for emotion, samples in emotion_groups.items():
        if emotion != "Other":
            balanced_index.extend(samples)
            print(f"  {emotion}: {len(samples)}개 (원본: {len(samples)}개)")
            
    random.shuffle(balanced_index) # 데이터 순서 섞기
    return balanced_index

@overload
def split_large_dataset(
    index: List[Dict[str, Any]],
    val_speaker_ratio: float = 0.2,
    test_speaker_ratio: float = 0.2,
    val_content_ratio: float = 0.2,
    test_content_ratio: float = 0.2,
    seed: int = 42,
    with_sentence: Literal[False] = False,
) -> Tuple[Tuple[List[str], List[str]],
           Tuple[List[str], List[str]],
           Tuple[List[str], List[str]]]: ...

@overload
def split_large_dataset(
    index: List[Dict[str, Any]],
    val_speaker_ratio: float = 0.2,
    test_speaker_ratio: float = 0.2,
    val_content_ratio: float = 0.2,
    test_content_ratio: float = 0.2,
    seed: int = 42,
    with_sentence: Literal[True] = True,
) -> Tuple[Tuple[List[str], List[str], List[str]],
           Tuple[List[str], List[str], List[str]],
           Tuple[List[str], List[str], List[str]]]: ...

def split_large_dataset(
    index: List[Dict[str, Any]],
    val_speaker_ratio: float = 0.2,
    test_speaker_ratio: float = 0.2,
    val_content_ratio: float = 0.2,
    test_content_ratio: float = 0.2,
    seed: int = 42,
    with_sentence: bool = False,
) -> Union[Tuple[Tuple[List[str], List[str]],
                 Tuple[List[str], List[str]],
                 Tuple[List[str], List[str]]],
           Tuple[Tuple[List[str], List[str], List[str]],
                 Tuple[List[str], List[str], List[str]],
                 Tuple[List[str], List[str], List[str]]]]:
    """
    Large 데이터셋 전용 화자/스크립트 불교차 분할
    
    Args:
        index: build_large_corpus_index() 결과
        val_speaker_ratio: validation용 화자 비율
        test_speaker_ratio: test용 화자 비율  
        val_content_ratio: validation용 스크립트 비율
        test_content_ratio: test용 스크립트 비율
        seed: 랜덤 시드
        with_sentence: True시 문장 정보도 함께 반환
        
    Returns:
        with_sentence=False: ((train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels))
        with_sentence=True: ((train_paths, train_labels, train_sentences), (val_paths, val_labels, val_sentences), (test_paths, test_labels, test_sentences))
    """
    rng = random.Random(seed)
    
    # 전체 화자 및 스크립트 ID 목록
    all_speakers = sorted(set([item["speaker"] for item in index]))
    all_contents = sorted(set([item["content_id"] for item in index]))
    
    print(f"📊 전체 화자: {len(all_speakers)}명 {all_speakers}")
    print(f"📝 전체 스크립트: {len(all_contents)}개")
    
    # 화자 분할
    speakers = all_speakers[:]
    rng.shuffle(speakers)
    n_val_spk = max(1, int(len(speakers) * val_speaker_ratio))
    n_test_spk = max(1, int(len(speakers) * test_speaker_ratio))
    
    val_speakers = set(speakers[:n_val_spk])
    test_speakers = set(speakers[n_val_spk:n_val_spk+n_test_spk])
    train_speakers = set(speakers[n_val_spk+n_test_spk:])
    
    # 스크립트 ID 분할
    contents = all_contents[:]
    rng.shuffle(contents)
    n_val_content = max(1, int(len(contents) * val_content_ratio))
    n_test_content = max(1, int(len(contents) * test_content_ratio))
    
    val_contents = set(contents[:n_val_content])
    test_contents = set(contents[n_val_content:n_val_content+n_test_content])
    train_contents = set(contents[n_val_content+n_test_content:])
    
    # 화자와 스크립트 모두 불교차인 샘플만 선택
    train_items = [item for item in index 
                   if item["speaker"] in train_speakers and item["content_id"] in train_contents]
    val_items = [item for item in index 
                 if item["speaker"] in val_speakers and item["content_id"] in val_contents]
    test_items = [item for item in index 
                  if item["speaker"] in test_speakers and item["content_id"] in test_contents]
    
    # 결과 출력
    def summarize_large(name, items, speakers_set, contents_set):
        spks = sorted(set([item["speaker"] for item in items]))
        cids = sorted(set([item["content_id"] for item in items]))
        emo_cnt = Counter([item["emotion"] for item in items])
        print(f"\n[{name}]")
        print(f"  샘플: {len(items)}개")
        print(f"  화자: {len(spks)}명 - {spks}")
        print(f"  스크립트: {len(cids)}개 (예시: {cids[:10]})")
        print(f"  감정분포: {dict(emo_cnt)}")
    
    summarize_large("TRAIN", train_items, train_speakers, train_contents)
    summarize_large("VAL", val_items, val_speakers, val_contents)
    summarize_large("TEST", test_items, test_speakers, test_contents)
    
    # 불교차 검증
    assert set([item["speaker"] for item in train_items]).isdisjoint(
        set([item["speaker"] for item in val_items + test_items])), "Train 화자가 Val/Test와 겹칩니다."
    assert set([item["speaker"] for item in val_items]).isdisjoint(
        set([item["speaker"] for item in test_items])), "Val 화자가 Test와 겹칩니다."
    assert set([item["content_id"] for item in train_items]).isdisjoint(
        set([item["content_id"] for item in val_items + test_items])), "Train 스크립트가 Val/Test와 겹칩니다."
    assert set([item["content_id"] for item in val_items]).isdisjoint(
        set([item["content_id"] for item in test_items])), "Val 스크립트가 Test와 겹칩니다."
    
    print("✅ 화자 및 스크립트 불교차 검증 완료!")
    
    # 최종 리스트 변환
    if with_sentence:
        def to_xy_with_sentence(items):
            paths = [item["path"] for item in items]
            emotions = [item["emotion"] for item in items]
            sentences = [item.get("sentence", "") for item in items]
            return paths, emotions, sentences
        
        return to_xy_with_sentence(train_items), to_xy_with_sentence(val_items), to_xy_with_sentence(test_items)
    else:
        def to_xy(items):
            return [item["path"] for item in items], [item["emotion"] for item in items]
        
        return to_xy(train_items), to_xy(val_items), to_xy(test_items)

def simple_augmentation(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """간단한 오디오 증강 (NumPy 2.x 호환)"""
    if np.random.random() < 0.3:  # 30% 확률로 노이즈 추가
        noise = np.random.normal(0, 0.005, audio.shape)
        audio = audio + noise
    
    if np.random.random() < 0.3:  # 30% 확률로 볼륨 조정
        volume_factor = np.random.uniform(0.8, 1.2)
        audio = audio * volume_factor
    
    return audio





def extract_number_from_filename(filename: str, type: Literal['content', 'emotion'] = 'emotion') -> Optional[int]:
    try:
        if type == "content":
            # 파일명에서 마지막 숫자 그룹 전체를 추출 (예: F2001_000123.wav -> 123)
            match = re.search(r'_(\d+)\.wav$', os.path.basename(filename))
            if match:
                return int(match.group(1))
            return None
        else:
            # F..._...xxxD.wav 에서 마지막 숫자 D를 추출
            match = re.search(r'_(\d+)\.wav$', os.path.basename(filename))
            if match:
                return int(match.group(1)) % 10
            return None
    except (ValueError, AttributeError):
        return None



def get_emotion_from_filename(filename: str) -> Optional[str]:
    """파일명에서 번호를 추출하여 감정 라벨 반환"""
    file_num = extract_number_from_filename(filename, type="content")
    if file_num is None:
        return None
        
    if 21 <= file_num <= 30:
        return "Anxious"
    elif 31 <= file_num <= 40:
        return "Kind"
    elif 141 <= file_num <= 150:
        return "Dry"
    else:
        return "Other"


# 데이터 전체를 스캔해서 (경로, 감정, 화자, 스크립트ID) 인덱스 생성
def build_corpus_index(data_dir: str,
                       accept_exts={'.wav', '.flac'},
                       require_emotion=True,
                       max_samples_per_class=None) -> List[Dict[str, Any]]:
    """
    return: [{"path": p, "emotion": e, "speaker": s, "content_id": c}, ...]
    max_samples_per_class: 클래스당 최대 샘플 수 (None이면 제한 없음)
    """
    index = []
    emotion_counts = {emotion: 0 for emotion in EMOTION_LABELS}  # 클래스별 카운트
    
    speakers = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d))])
    print(f"📁 화자 폴더 수: {len(speakers)}")

    for spk in tqdm(speakers, desc="인덱스 구축"):
        spk_dir = os.path.join(data_dir, spk)
        # 하위 디렉토리를 재귀적으로 탐색 (감정별 폴더/단일 폴더 둘 다 대응)
        for root, _, files in os.walk(spk_dir):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in accept_exts:
                    continue
                path = os.path.join(root, fn)

                # 감정 라벨
                emo = infer_emotion_from_path(path)
                if require_emotion and emo not in EMOTION_LABELS:
                    # Other 감정도 포함하도록 수정
                    emo = "Other"
                
                # 클래스별 최대 샘플 수 제한
                if max_samples_per_class and emotion_counts[emo] >= max_samples_per_class:
                    continue

                # 스크립트(대화) ID: 파일명에서 추출 (기존 규칙 그대로)
                cid = extract_number_from_filename(fn, type="content")
                if cid is None:
                    # 스크립트 ID 없으면 제외(불교차 조건을 보장하기 위해)
                    continue

                index.append({
                    "path": path,
                    "emotion": emo,
                    "speaker": spk,
                    "content_id": cid
                })
                emotion_counts[emo] += 1
    
    print(f"✅ 인덱스 샘플 수: {len(index)}")
    print(f"📊 클래스별 분포: {dict(emotion_counts)}")
    return index


def split_speaker_and_content(
    index: List[Dict[str, Any]],
    val_content_ratio: float = 0.2,
    test_content_ratio: float = 0.2,
    val_speaker_ratio: float = 0.2,
    test_speaker_ratio: float = 0.2,
    seed: int = 42,
    fixed_val_content_ids: Optional[List[int]] = None,
    fixed_test_content_ids: Optional[List[int]] = None,
) -> Tuple[Tuple[List[str], List[str]],
           Tuple[List[str], List[str]],
           Tuple[List[str], List[str]]]:
    """
    index: build_corpus_index() 반환 리스트
    반환: ((train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels))
    """
    rng = random.Random(seed)

    # 전체 스크립트 ID, 화자 목록
    all_contents = sorted(set([it["content_id"] for it in index]))
    all_speakers = sorted(set([it["speaker"] for it in index]))

    # --- 2-1) 스크립트(대화) 불교차 세트 만들기
    if fixed_val_content_ids is not None and fixed_test_content_ids is not None:
        val_contents = set(fixed_val_content_ids)
        test_contents = set(fixed_test_content_ids)
        train_contents = set(all_contents) - val_contents - test_contents
    else:
        contents = all_contents[:]
        rng.shuffle(contents)
        n_val = max(1, int(len(contents) * val_content_ratio))
        n_test = max(1, int(len(contents) * test_content_ratio))
        val_contents = set(contents[:n_val])
        test_contents = set(contents[n_val:n_val+n_test])
        train_contents = set(contents[n_val+n_test:])

    # --- 2-2) 화자 불교차 세트 만들기
    speakers = all_speakers[:]
    rng.shuffle(speakers)
    n_val_spk = max(1, int(len(speakers) * val_speaker_ratio))
    n_test_spk = max(1, int(len(speakers) * test_speaker_ratio))
    val_speakers = set(speakers[:n_val_spk])
    test_speakers = set(speakers[n_val_spk:n_val_spk+n_test_spk])
    train_speakers = set(speakers[n_val_spk+n_test_spk:])

    # --- 2-3) 교집합 제거: 두 조건(화자 세트, 스크립트 세트)을 동시에 만족하는 샘플만 채택
    train_items = [it for it in index
                   if it["speaker"] in train_speakers and it["content_id"] in train_contents]
    val_items   = [it for it in index
                   if it["speaker"] in val_speakers and it["content_id"] in val_contents]
    test_items  = [it for it in index
                   if it["speaker"] in test_speakers and it["content_id"] in test_contents]

    # --- 2-4) 점검 출력
    def summarize(name, items):
        spks = sorted(set([it["speaker"] for it in items]))
        cids = sorted(set([it["content_id"] for it in items]))
        emo_cnt = Counter([it["emotion"] for it in items])
        print(f"\n[{name}] 샘플: {len(items)}, 화자: {len(spks)}, 스크립트ID: {len(cids)}")
        print(f"  감정분포: {dict(emo_cnt)}")
        print(f"  예시 화자(최대 10): {spks[:10]}")
        print(f"  예시 스크립트ID(최대 20): {cids[:20]}")

    summarize("TRAIN", train_items)
    summarize("VAL",   val_items)
    summarize("TEST",  test_items)

    # --- 2-5) 교차 검증: 화자/스크립트 불교차 여부 확인
    assert set([it["speaker"] for it in train_items]).isdisjoint(set([it["speaker"] for it in val_items + test_items])), \
        "Train 화자가 Val/Test와 겹칩니다."
    assert set([it["speaker"] for it in val_items]).isdisjoint(set([it["speaker"] for it in test_items])), \
        "Val 화자가 Test와 겹칩니다."
    assert set([it["content_id"] for it in train_items]).isdisjoint(set([it["content_id"] for it in val_items + test_items])), \
        "Train 스크립트ID가 Val/Test와 겹칩니다."
    assert set([it["content_id"] for it in val_items]).isdisjoint(set([it["content_id"] for it in test_items])), \
        "Val 스크립트ID가 Test와 겹칩니다."

    # --- 2-6) 최종 리스트 변환
    def to_xy(items):
        return [it["path"] for it in items], [it["emotion"] for it in items]

    return to_xy(train_items), to_xy(val_items), to_xy(test_items)


def load_dataset_subset(data_dir: str, max_per_class: int) -> Tuple[List[str], List[str]]:
    audio_paths = []
    labels = []
    emotion_counts = {label: 0 for label in EMOTION_LABELS}
    
    person_folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    print(f"📁 발견된 person 폴더: {len(person_folders)}개")
    
    for person_folder in tqdm(person_folders, desc="데이터셋 로딩 중"):
        wav_path = os.path.join(data_dir, person_folder, "wav_48000")
        if not os.path.exists(wav_path):
            continue
        
        for audio_file in os.listdir(wav_path):
            if not audio_file.lower().endswith('.wav'):
                continue
            
            emotion_label = get_emotion_from_filename(audio_file)
            if emotion_label and emotion_counts[emotion_label] < max_per_class:
                audio_paths.append(os.path.join(wav_path, audio_file))
                labels.append(emotion_label)
                emotion_counts[emotion_label] += 1
        
        if all(count >= max_per_class for count in emotion_counts.values()):
            break

    print(f"\n📊 로드된 데이터 분포 (빠른 테스트용):")
    for emotion, count in emotion_counts.items():
        percentage = (count / len(audio_paths) * 100) if len(audio_paths) > 0 else 0
        print(f"  {emotion}: {count}개 ({percentage:.1f}%)")
            
    return audio_paths, labels


# (필수) 화자 ID 추출: data_dir 바로 아래 1단계 폴더명이 화자
def extract_speaker_id(audio_path: str, data_dir: str) -> str:
    rel = os.path.relpath(audio_path, data_dir)
    spk = rel.split(os.sep)[0]
    return spk



def build_speaker_mapping(train_paths, data_dir):
    train_speakers = sorted({extract_speaker_id(p, data_dir) for p in train_paths})
    spk2id = {spk: i for i, spk in enumerate(train_speakers)}
    return spk2id



# (선택) 경로에서 감정 라벨 추론 (폴더명에 Anxious/Kind/Dry가 있으면 그걸 사용)
def infer_emotion_from_path(audio_path: str) -> Optional[str]:
    parts = os.path.normpath(audio_path).split(os.sep)
    for p in reversed(parts):
        if p in EMOTION_LABELS:
            return p
    # 폴더명에 없으면 파일명 규칙으로 추론 (기존 함수)
    return get_emotion_from_filename(os.path.basename(audio_path))





