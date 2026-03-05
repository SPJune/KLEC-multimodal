from glob import glob
import hydra
import json
import librosa
import logging
import numpy as np
import os
import re
import unicodedata
from omegaconf import DictConfig
import soundfile as sf
import torch
import csv

from silero_metric import calculate_error_cnt, calculate_error_rate
from transformers import WhisperForConditionalGeneration, WhisperProcessor

_SINO_KOREAN_DIGITS = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
_SINO_KOREAN_UNITS = ["", "십", "백", "천"]

def _int_to_sino_korean(num: int) -> str:
    if num == 0:
        return "영"
    if num < 0:
        return "마이너스" + _int_to_sino_korean(-num)

    parts = []
    unit_idx = 0
    while num > 0 and unit_idx < len(_SINO_KOREAN_UNITS):
        digit = num % 10
        if digit != 0:
            unit = _SINO_KOREAN_UNITS[unit_idx]
            if unit_idx == 0:
                parts.append(_SINO_KOREAN_DIGITS[digit])
            else:

                parts.append((("" if digit == 1 else _SINO_KOREAN_DIGITS[digit]) + unit))
        num //= 10
        unit_idx += 1
    return "".join(reversed(parts))

# Match standalone 1~4 digit numbers (not part of a longer digit sequence).
_RE_1TO4DIGIT = re.compile(r"(?<!\d)\d{1,4}(?!\d)")

def replace_1to4digit_numbers_with_korean(text: str) -> str:
    """
    ASR 결과 내 1~4자리 숫자(연도/수량 등)를 한글(한자어) 읽기로 변환.
    예: "2014" -> "이천십사", "12" -> "십이", "7" -> "칠"
    - 앞/뒤가 숫자인 경우(더 긴 숫자열의 일부)는 변환하지 않음
    - 선행 0이 있는 경우(예: 0012)는 그대로 둠
    """
    if not text:
        return text

    def _repl(m: re.Match) -> str:
        s = m.group(0)
        if len(s) > 1 and s[0] == "0":
            return s
        n = int(s)
        if n < 0 or n > 9999:
            return s
        return _int_to_sino_korean(n)

    return _RE_1TO4DIGIT.sub(_repl, text)

# Backward-compatible alias (older name used in earlier code/tests).
replace_4digit_numbers_with_korean = replace_1to4digit_numbers_with_korean


_LATIN_TO_KOREAN_LETTER = {
    "a": "에이",
    "b": "비",
    "c": "씨",
    "d": "디",
    "e": "이",
    "f": "에프",
    "g": "지",
    "h": "에이치",
    "i": "아이",
    "j": "제이",
    "k": "케이",
    "l": "엘",
    "m": "엠",
    "n": "엔",
    "o": "오",
    "p": "피",
    "q": "큐",
    "r": "알",
    "s": "에스",
    "t": "티",
    "u": "유",
    "v": "브이",
    "w": "더블유",
    "x": "엑스",
    "y": "와이",
    "z": "제트",
}

# Replace only single-letter tokens (avoid touching words/emails/urls like "abc", "a@b.com", etc.)
# Note: we intentionally do NOT treat '-' as a boundary to avoid changing hyphenated words like "x-ray".

_HANGUL_BOUNDARY = (
    r"\uAC00-\uD7A3"  # Hangul Syllables
    r"\u1100-\u11FF"  # Hangul Jamo
    r"\u3130-\u318F"  # Hangul Compatibility Jamo
    r"\uA960-\uA97F"  # Hangul Jamo Extended-A
    r"\uD7B0-\uD7FF"  # Hangul Jamo Extended-B
)
_RE_SINGLE_LATIN = re.compile(
    rf"(^|[\s\(\[\{{\"'“‘,.:;!?/\\{_HANGUL_BOUNDARY}])"
    rf"([A-Za-z])"
    rf"(?=$|[\s\)\]\}}\"'”’,.:;!?/\\{_HANGUL_BOUNDARY}])"
)

def replace_single_latin_letters_with_korean(text: str) -> str:
    if not text:
        return text

    def _repl(m: re.Match) -> str:
        prefix = m.group(1)
        ch = m.group(2)
        rep = _LATIN_TO_KOREAN_LETTER.get(ch.lower())
        return prefix + (rep if rep is not None else ch)

    return _RE_SINGLE_LATIN.sub(_repl, text)

def replace_percent_with_korean(text: str) -> str:
    """
    Replace percent symbols with Korean word.
    Examples:
      "50%" -> "50퍼센트"
      "50％" -> "50퍼센트"
    """
    if not text:
        return text
    return text.replace("％", "퍼센트").replace("%", "퍼센트")

def limit_repeated_syllables_or_words(text: str, max_repeat: int = 3) -> str:
    """
    ASR prediction 후처리:
    - 같은 어절(공백으로 구분되는 토큰)이 연속으로 max_repeat 초과 반복되면 max_repeat까지만 유지
    - 같은 음절(유니코드 문자)이 연속으로 max_repeat 초과 반복되면 max_repeat까지만 유지

    예)
      "안녕 안녕 안녕 안녕" -> "안녕 안녕 안녕"
      "하하하하하" -> "하하하"
    """
    if not text:
        return text
    if max_repeat < 1:
        return ""

    text = unicodedata.normalize("NFC", text)

    # 1) Word-level: keep whitespace, limit consecutive identical non-space tokens.
    parts = re.findall(r"\s+|\S+", text)
    limited_parts = []
    prev_token = None
    prev_count = 0
    for p in parts:
        if p.isspace():
            limited_parts.append(p)
            continue
        if p == prev_token:
            prev_count += 1
        else:
            prev_token = p
            prev_count = 1
        if prev_count <= max_repeat:
            limited_parts.append(p)
    text = "".join(limited_parts)

    # 2) Syllable-level (character-level) within each non-space token.
    parts = re.findall(r"\s+|\S+", text)
    out_parts = []
    for p in parts:
        if p.isspace():
            out_parts.append(p)
            continue
        chars = []
        prev_ch = None
        ch_count = 0
        for ch in p:
            if ch == prev_ch:
                ch_count += 1
            else:
                prev_ch = ch
                ch_count = 1
            if ch_count <= max_repeat:
                chars.append(ch)
        out_parts.append("".join(chars))
    return "".join(out_parts).strip()

# Allow Hangul syllables + jamo blocks + whitespace.
# This avoids flagging NFD/decomposed Hangul like "반..." as non-Korean.
_RE_NON_KO_OR_SPACE = re.compile(
    r"[^\uAC00-\uD7A3"      # Hangul Syllables
    r"\u1100-\u11FF"       # Hangul Jamo
    r"\u3130-\u318F"       # Hangul Compatibility Jamo
    r"\uA960-\uA97F"       # Hangul Jamo Extended-A
    r"\uD7B0-\uD7FF"       # Hangul Jamo Extended-B
    r"\s]"
)

def has_non_korean_or_space(text: str) -> bool:
    """
    True if text contains anything other than Hangul syllables(가-힣) or whitespace.
    """
    if not text:
        return False
    text = unicodedata.normalize("NFC", text)
    return _RE_NON_KO_OR_SPACE.search(text) is not None

def load_json(json_dir):
    with open(json_dir) as f:
        json_file = json.load(f)
    return json_file['text']

def load_text_pair_npz(npz_path: str) -> str:
    """
    silent_speech_dataset/*/{sess}/data/text/text_pair_*.npz 로부터 GT 텍스트 로드.
    확인된 포맷: keys ['text1','text2', ...], text2가 문장.
    """
    z = np.load(npz_path, allow_pickle=True)
    if "text2" not in z.files:
        raise KeyError(f"text2 not found in {npz_path}: keys={z.files}")
    # numpy scalar string
    return str(z["text2"])

def save_json(write_dir, data):
    with open(write_dir, 'w') as f:
        json.dump(data, f)

def _try_paths(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def find_gt_info_json(data_path: str, data_split: str, sess: str, idx: str):
    """
    GT text json을 여러 체계에 대해 탐색.
    - legacy: silent_speech_dataset/{split}/voiced_parallel_data/{sess}/{idx}_info.json
    - global: silent_speech_dataset/voiced_parallel_data/{sess}/{idx}_info.json
    - new-ish: silent_speech_dataset/voiced/{sess}/data/info/{idx}_info.json
    """
    base = os.path.join(data_path, "silent_speech_dataset")
    idx_i = None
    try:
        idx_i = int(idx)
    except Exception:
        idx_i = None

    cands = []
    # legacy split-based
    cands.append(os.path.join(base, str(data_split), "voiced_parallel_data", sess, f"{idx}_info.json"))
    if idx_i is not None:
        cands.append(os.path.join(base, str(data_split), "voiced_parallel_data", sess, f"{idx_i:04d}_info.json"))
    # global voiced_parallel_data
    cands.append(os.path.join(base, "voiced_parallel_data", sess, f"{idx}_info.json"))
    if idx_i is not None:
        cands.append(os.path.join(base, "voiced_parallel_data", sess, f"{idx_i:04d}_info.json"))
    # voiced session info dir
    cands.append(os.path.join(base, "voiced", sess, "data", "info", f"{idx}_info.json"))
    if idx_i is not None:
        cands.append(os.path.join(base, "voiced", sess, "data", "info", f"{idx_i:04d}_info.json"))

    found = _try_paths(cands)
    if found is not None:
        return found

    # last resort: glob around session
    for pat in [
        os.path.join(base, "**", sess, f"*{idx}*info.json"),
    ]:
        ms = glob(pat, recursive=True)
        if len(ms) > 0:
            ms.sort()
            return ms[0]
    return None

def find_gt_text_pair_npz(data_path: str, data_type: str, sess: str, idx: str):
    """
    GT 텍스트를 text_pair_*.npz에서 찾는다.
    예) /data2/ai_champion/silent_speech_dataset/silent/3-3/data/text/text_pair_3_3_3914.wav_*.npz
    """
    base = os.path.join(data_path, "silent_speech_dataset")


    patterns = [
        os.path.join(base, data_type, sess, "data", "text", f"*_{idx}.wav_*.npz"),
    ]
    try:
        idx_i = int(idx)
        patterns.append(os.path.join(base, data_type, sess, "data", "text", f"*_{idx_i:04d}.wav_*.npz"))
    except Exception:
        pass

    for pat in patterns:
        ms = glob(pat)
        if len(ms) > 0:
            ms.sort()
            return ms[0]
    return None

def load_gt_text(cfg: DictConfig, sess: str, idx: str, data_type: str):
    """
    우선순위:
    1) text_pair npz (새 체계)
    2) legacy info json
    """
    npz_path = find_gt_text_pair_npz(cfg.data_path, data_type, sess, idx)
    if npz_path is not None:
        return load_text_pair_npz(npz_path)
    json_path = find_gt_info_json(cfg.data_path, cfg.data_split, sess, idx)
    if json_path is not None:
        return load_json(json_path)
    return None

def iter_eval_wavs(cfg: DictConfig, data_dir: str):
    """
    가능한 경우 preprocess/emg_split.csv를 기준으로 평가 대상 wav 목록을 만든다.
    그렇지 않으면 data_dir 아래 wav를 재귀적으로 모두 평가한다.
    """
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    split_csv = os.path.join(project_dir, "preprocess", "emg_split.csv")

    # 1) prefer emg_split.csv (new system)
    if os.path.exists(split_csv):
        rows = []
        with open(split_csv, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if str(r.get("split", "")).strip() != str(cfg.data_split):
                    continue
                rows.append(r)

        # optional filter by cfg.data_type (only if it matches actual dt set)
        want_dt = None
        if hasattr(cfg, "data_type") and cfg.data_type is not None and str(cfg.data_type).strip() != "":
            want_dt = str(cfg.data_type).strip()
        dt_set = set(str(r.get("data_type", "")).strip() for r in rows)
        if want_dt is not None and want_dt in dt_set:
            rows = [r for r in rows if str(r.get("data_type", "")).strip() == want_dt]

        for r in rows:
            sess = str(r["session"])
            dt = str(r["data_type"])
            idx = str(r["data_num"])
            p = os.path.join(data_dir, dt, f"{sess}_{idx}.wav")
            if os.path.exists(p):
                yield p
            else:
                # fallback: try to find any matching wav under data_dir
                ms = glob(os.path.join(data_dir, "**", f"{sess}_{idx}.wav"), recursive=True)
                if len(ms) > 0:
                    ms.sort()
                    yield ms[0]
        return

    # 2) fallback: recursive glob
    for p in sorted(glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)):
        yield p

@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg:DictConfig):
    data_dir = os.path.join(cfg.exp_path, cfg.wav_dir)
    os.makedirs(data_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logger_name = 'asr_whisper.log'
    file_handler = logging.FileHandler(os.path.join(data_dir, logger_name), mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    sr16 = 16000
    path_list = list(iter_eval_wavs(cfg, data_dir))
    processor = WhisperProcessor.from_pretrained('openai/whisper-medium')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-medium').to('cuda')
    use_jiwer = bool(getattr(cfg, "use_jiwer", False))
    if use_jiwer:
        try:
            import jiwer  # type: ignore
        except Exception as e:
            raise ImportError(
                "cfg.use_jiwer=True but 'jiwer' is not available. Install jiwer or set use_jiwer=False."
            ) from e
        predictions = []
        targets = []
        results = None
    else:
        predictions = None
        targets = None
        results = []
    invalid_postprocessed_preds = []
    skipped = 0
    evaluated = 0
    max_eval = getattr(cfg, "max_eval", None)
    try:
        max_eval = int(max_eval) if max_eval is not None else None
    except Exception:
        max_eval = None
    for i, path in enumerate(path_list):
        audio, sr = sf.read(path)
        if sr != sr16:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sr16)
        sess, idx = path.split('/')[-1].split('_')
        idx = idx.split('.')[0]

        data_type = os.path.basename(os.path.dirname(path))
        info_text = load_gt_text(cfg, sess, idx, data_type)
        if info_text is None:
            logger.info(f"[WARN] missing GT text for {sess} {idx} (type={data_type}, wav={path}) -> skip")
            skipped += 1
            continue
        target_text = processor.tokenizer._normalize(info_text)
        # Metric stability: use decomposed Hangul (NFD) consistently for CER/WER.
        # This prevents NFC vs NFD mismatches from inflating edit distance.
        target_text_eval = unicodedata.normalize("NFD", target_text or "")
        input_features = processor(audio, sampling_rate=sr16, return_tensors='pt').input_features
        with torch.no_grad():
            predicted_ids = model.generate(input_features.to('cuda'), language="korean", task="transcribe")[0]

        transcription = processor.decode(predicted_ids)
        prediction = processor.tokenizer._normalize(transcription)

        # Post-process: convert 1~4 digit numbers to Korean reading.
        prediction = replace_1to4digit_numbers_with_korean(prediction)
        prediction = replace_single_latin_letters_with_korean(prediction)
        prediction = replace_percent_with_korean(prediction)
        prediction = limit_repeated_syllables_or_words(prediction, max_repeat=3)
        prediction_eval = unicodedata.normalize("NFD", prediction or "")
        if has_non_korean_or_space(prediction):
            invalid_postprocessed_preds.append((i, sess, idx, prediction))

        result = calculate_error_cnt(prediction_eval, target_text_eval)
        wer0 = result[-2]/result[-1]
        cer0 = result[0]/result[1]
        if use_jiwer:
            # accumulate for jiwer metrics
            targets.append(target_text_eval)
            predictions.append(prediction_eval)
        else:
            results.append(result)
        evaluated += 1

        logger.info(f'trgt{i} {sess} {idx}: {target_text}')
        logger.info(f'pred{i} {sess} {idx}: {prediction}')
        logger.info(f'{i} sample cer: {cer0*100:.2f}, wer: {wer0*100:.2f}')
        if max_eval is not None and evaluated >= max_eval:
            logger.info(f"[INFO] Reached max_eval={max_eval}, stopping early.")
            break
    if evaluated == 0:
        logger.info("[WARN] No samples evaluated (all skipped).")
        logger.info(f"[INFO] skipped {skipped} samples due to missing GT")
        return
    if use_jiwer:
        # NOTE: jiwer is primarily designed for space-delimited word error rate.
        # For CER we compute jiwer.wer on character-tokenized transforms.
        transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])

        class SentencesToListOfChars(jiwer.AbstractTransform):
            def process_string(self, s: str):
                return list(s)

            def process_list(self, inp):
                chars = []
                for sentence in inp:
                    chars.extend(self.process_string(sentence))
                return chars

        wer_transform = jiwer.Compose([transformation, jiwer.SentencesToListOfWords()])
        cer_transform = jiwer.Compose([transformation, SentencesToListOfChars()])
        cer_wo_space_transform = jiwer.Compose([
            transformation,
            jiwer.RemoveWhiteSpace(replace_by_space=False),
            SentencesToListOfChars(),
        ])

        cer = jiwer.wer(targets, predictions, truth_transform=cer_transform, hypothesis_transform=cer_transform)
        cer_wo_space = jiwer.wer(
            targets,
            predictions,
            truth_transform=cer_wo_space_transform,
            hypothesis_transform=cer_wo_space_transform,
        )
        wer = jiwer.wer(targets, predictions, truth_transform=wer_transform, hypothesis_transform=wer_transform)
    else:
        cer, cer_wo_space, wer = calculate_error_rate(results)
    result_str = f"CER: {cer*100:.2f}%, CER without space: {cer_wo_space*100:.2f}%, WER: {wer*100:.2f}%"
    if skipped > 0:
        logger.info(f"[INFO] skipped {skipped} samples due to missing GT")

    if len(invalid_postprocessed_preds) > 0:
        msg = (
            f"[WARN] Found {len(invalid_postprocessed_preds)} predictions containing non-Korean/non-space "
            f"characters after postprocess. Printing offending sentences below."
        )
        logger.warning(msg)
        print(msg)
        for j, sess, idx, pred in invalid_postprocessed_preds:
            line = f"[WARN] pred{j} {sess} {idx}: {pred}"
            logger.warning(line)
            print(line)

    # Print final summary at the very end (after warnings).
    logger.info(result_str)
    print(result_str)

if __name__ == '__main__':
    main()
