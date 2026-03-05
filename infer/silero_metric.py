import Levenshtein as Lev
from typing import List
from silero_utils import normalize_text
import unicodedata

def cer(prediction, target):
    """
    Computes the Letter Error Rate, defined as the edit distance.
    Arguments:
        prediction (string): space-separated sentence
        target (string): space-separated sentence
        lang (string): language
    """
    # prediction, target, = prediction.replace(' ', ''), target.replace(' ', '')
    return Lev.distance(prediction, target), len(target)

def cer_wo_space(prediction, target):
    # remove space and enter
    prediction = prediction.replace(' ', '').replace('\n', '')
    target = target.replace(' ', '').replace('\n', '')
    return Lev.distance(prediction, target), len(target)

def wer(prediction, target):
    """
    Computes the Word Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to words.
    Arguments:
        prediction (string): space-separated sentence
        target (string): space-separated sentence
        lang (string): language
    """
    # build mapping of words to integers
    b = set(prediction.split() + target.split())
    word2char = dict(zip(b, range(len(b))))

    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    prediction = [chr(word2char[w]) for w in prediction.split()]
    target = [chr(word2char[w]) for w in target.split()]

    return Lev.distance(''.join(prediction), ''.join(target)), len(target)

def calculate_error_cnt(prediction, target):
    # Ensure consistent Hangul representation for edit-distance based CER.
    # We use NFD (decomposed jamo) so CER is measured at jamo-level consistently.
    prediction = unicodedata.normalize("NFD", prediction or "")
    target = unicodedata.normalize("NFD", target or "")

    def normalize_text_unicode(text: str) -> str:
        """
        유니코드(한글 포함) 보존용 정규화.
        - 문자/숫자(Unicode category L*, N*)와 공백만 남김
        - 기타(구두점/기호 등)는 제거
        - 공백은 단일 스페이스로 정리
        """
        text = unicodedata.normalize("NFD", text or "")
        text = text.lower().replace("-", " ")
        kept = []
        last_space = False
        for ch in text:
            if ch.isspace():
                if not last_space:
                    kept.append(" ")
                last_space = True
                continue
            cat = unicodedata.category(ch)
            if cat and (cat[0] == "L" or cat[0] == "N"):
                kept.append(ch)
                last_space = False
            else:
                # drop punctuation/symbols/marks etc.
                pass
        return "".join(kept).strip()



    pred_basic = normalize_text(prediction)
    tgt_basic = normalize_text(target)
    pred_uni = normalize_text_unicode(prediction)
    tgt_uni = normalize_text_unicode(target)

    # choose normalization that preserves more target signal
    if len(tgt_basic) >= max(2, len(tgt_uni) // 2):
        prediction = pred_basic
        target = tgt_basic
    else:
        prediction = pred_uni
        target = tgt_uni

    char_err_cnt, char_target_cnt = cer(prediction, target)
    char_wo_space_err_cnt, char_wo_space_target_cnt = cer_wo_space(prediction, target)
    word_err_cnt, word_target_cnt = wer(prediction, target)
    if char_target_cnt == 0:
        char_target_cnt = 1
    if char_wo_space_target_cnt == 0:
        char_wo_space_target_cnt = 1
    if word_target_cnt == 0:
        word_target_cnt = 1

    return char_err_cnt, char_target_cnt, char_wo_space_err_cnt, char_wo_space_target_cnt, word_err_cnt, word_target_cnt

def calculate_error_rate(results: List):
    char_err_tot_cnt = 0
    char_target_tot_cnt = 0
    char_wo_space_err_tot_cnt = 0
    char_wo_space_target_tot_cnt = 0
    word_err_tot_cnt = 0
    word_target_tot_cnt = 0
    
    for char_err_cnt, char_target_cnt, char_wo_space_err_cnt, char_wo_space_target_cnt, word_err_cnt, word_target_cnt in results:
        char_err_tot_cnt += char_err_cnt
        char_target_tot_cnt += char_target_cnt
        char_wo_space_err_tot_cnt+=char_wo_space_err_cnt
        char_wo_space_target_tot_cnt+=char_wo_space_target_cnt
        word_err_tot_cnt += word_err_cnt
        word_target_tot_cnt += word_target_cnt

    char_err_rate = char_err_tot_cnt / char_target_tot_cnt
    char_wo_space_err_rate = char_wo_space_err_tot_cnt / char_wo_space_target_tot_cnt
    word_err_rate = word_err_tot_cnt / word_target_tot_cnt

    return char_err_rate, char_wo_space_err_rate, word_err_rate
