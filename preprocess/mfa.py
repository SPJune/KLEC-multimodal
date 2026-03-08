import os
import glob
import argparse
import subprocess
import tempfile
import shutil
from typing import Optional, Tuple
import numpy as np


VOICED_BASE = "/data/silent_speech_dataset/voiced"
SILENT_BASE = "/data/silent_speech_dataset/silent"
PHONEME_SET_PATH = "/data/silent_speech_dataset/phoneme_set.json"


MFA_ACOUSTIC_MODEL = "korean_mfa"
MFA_DICTIONARY = "korean_mfa"
MFA_G2P_MODEL = "korean_mfa"
MFA_COMMAND_HISTORY = os.path.expanduser("~/Documents/MFA/command_history.yaml")


ALL_PHONEMES = set()


def clean_mfa_history():
    if os.path.exists(MFA_COMMAND_HISTORY):
        try:
            os.remove(MFA_COMMAND_HISTORY)
        except OSError:
            pass


def find_audio_file(sess: str, num: int) -> Optional[str]:
    pattern = os.path.join(VOICED_BASE, sess, 'data/audio', f'*_{num:04d}.*.flac')
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return None


def find_text_file(sess: str, num: int) -> Optional[str]:
    pattern = os.path.join(SILENT_BASE, sess, 'data/text', f'*_{num:04d}.*.npz')
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return None


def get_text_from_npz(npz_path: str) -> str:
    import re
    data = np.load(npz_path, allow_pickle=True)
    text = str(data['text2'])

    text = re.sub(r'[^\w\s가-힣a-zA-Z0-9]', '', text)
    text = text.strip()
    return text


def get_textgrid_output_path(sess: str, num: int) -> str:
    output_dir = os.path.join(VOICED_BASE, sess, 'data/textgrid')
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f'tg_{num:04d}.TextGrid')


def generate_dictionary_with_g2p(text: str, dict_path: str, verbose: bool = True) -> Tuple[bool, dict]:
    global ALL_PHONEMES
    

    words_path = dict_path + ".words"
    words = set(text.split())
    
    with open(words_path, 'w', encoding='utf-8') as f:
        for word in words:
            f.write(word + '\n')
    

    clean_mfa_history()
    

    g2p_cmd = [
        'mfa', 'g2p',
        words_path,
        MFA_G2P_MODEL,
        dict_path,
        '--clean',
        '--overwrite',
        '--dictionary_path', MFA_DICTIONARY
    ]
    
    result = subprocess.run(g2p_cmd, capture_output=True, text=True)
    

    if os.path.exists(words_path):
        os.remove(words_path)
    
    if result.returncode != 0:
        print(f"G2P 오류: {result.stderr}")
        return False, {}
    

    word_phonemes = {}
    if os.path.exists(dict_path):
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    word = parts[0]

                    phoneme_parts = parts[-1].split()
                    word_phonemes[word] = phoneme_parts

                    ALL_PHONEMES.update(phoneme_parts)
        

        if verbose:
            print("G2P 생성 음소:")
            for word, phonemes in sorted(word_phonemes.items()):
                print(f"  {word}: {' '.join(phonemes)}")
    
    return os.path.exists(dict_path), word_phonemes


def extract_phonemes_from_textgrid(textgrid_path: str) -> set:
    global ALL_PHONEMES
    phonemes = set()
    
    if not os.path.exists(textgrid_path):
        return phonemes
    
    with open(textgrid_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    import re
    


    phones_match = re.search(r'name = "phones".*?(?=item \[|$)', content, re.DOTALL)
    if phones_match:
        phones_section = phones_match.group(0)

        matches = re.findall(r'text = "([^"]*)"', phones_section)
        for match in matches:

            if match and match not in ('', 'spn', 'sil'):
                phonemes.add(match)
    

    ALL_PHONEMES.update(phonemes)
    
    return phonemes


def textgrid_has_spn(textgrid_path: str) -> bool:
    if not os.path.exists(textgrid_path):
        return False
    
    with open(textgrid_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    import re
    phones_match = re.search(r'name = "phones".*?(?=item \[|$)', content, re.DOTALL)
    if phones_match:
        phones_section = phones_match.group(0)
        matches = re.findall(r'text = "([^"]*)"', phones_section)
        return 'spn' in matches
    
    return False


def save_phoneme_set(path: str = None):
    import json
    
    if path is None:
        path = PHONEME_SET_PATH
    
    phoneme_list = sorted(list(ALL_PHONEMES))
    
    data = {
        'phonemes': phoneme_list,
        'count': len(phoneme_list)
    }
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n음소 집합 저장 완료: {path}")
    print(f"총 {len(phoneme_list)}개 음소: {', '.join(phoneme_list[:20])}{'...' if len(phoneme_list) > 20 else ''}")


def load_phoneme_set(path: str = None) -> list:
    import json
    
    if path is None:
        path = PHONEME_SET_PATH
    
    if not os.path.exists(path):
        print(f"음소 집합 파일이 없습니다: {path}")
        return []
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('phonemes', [])


def run_mfa_alignment(
    audio_path: str,
    text: str,
    output_path: str,
    overwrite: bool = True,
    use_g2p: bool = True
) -> bool:
    if os.path.exists(output_path) and not overwrite:
        print(f"파일이 이미 존재함 (건너뜀): {output_path}")
        return True
    

    temp_dir = tempfile.mkdtemp(prefix='mfa_')
    
    try:

        base_name = "utterance"
        

        wav_path = os.path.join(temp_dir, f"{base_name}.wav")
        
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', audio_path,
            '-ar', '16000', '-ac', '1',
            wav_path
        ]
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ffmpeg 오류: {result.stderr}")
            return False
        

        txt_path = os.path.join(temp_dir, f"{base_name}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        

        mfa_output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(mfa_output_dir, exist_ok=True)
        

        if use_g2p:
            custom_dict_path = os.path.join(temp_dir, 'custom.dict')
            print("G2P로 발음 사전 생성 중...")
            success, _ = generate_dictionary_with_g2p(text, custom_dict_path)
            if success:
                dictionary = custom_dict_path
                print(f"G2P 사전 생성 완료")
            else:
                print("G2P 실패, 기본 사전 사용")
                dictionary = MFA_DICTIONARY
        else:
            dictionary = MFA_DICTIONARY
        

        clean_mfa_history()
        
        mfa_cmd = [
            'mfa', 'align',
            temp_dir,
            dictionary,
            MFA_ACOUSTIC_MODEL,
            mfa_output_dir,
            '--clean',
            '--overwrite',
            '--single_speaker'
        ]
        
        print(f"MFA 정렬 실행 중...")
        result = subprocess.run(mfa_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"MFA 오류: {result.stderr}")
            print(f"MFA 출력: {result.stdout}")
            return False
        

        generated_tg = os.path.join(mfa_output_dir, f"{base_name}.TextGrid")
        
        if os.path.exists(generated_tg):

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            

            if os.path.exists(output_path):
                os.remove(output_path)
                print(f"기존 파일 덮어쓰기: {output_path}")
            
            shutil.copy(generated_tg, output_path)
            print(f"TextGrid 생성 완료: {output_path}")
            

            phonemes = extract_phonemes_from_textgrid(output_path)
            if phonemes:
                print(f"추출된 음소 ({len(phonemes)}개): {' '.join(sorted(phonemes))}")
            
            return True
        else:
            print(f"TextGrid 파일이 생성되지 않음: {generated_tg}")

            if os.path.exists(mfa_output_dir):
                print(f"출력 디렉토리 내용: {os.listdir(mfa_output_dir)}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"명령 실행 오류: {e}")
        print(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def process_single(sess: str, num: int, overwrite: bool = True) -> bool:
    """단일 파일 처리"""
    print(f"\n{'='*60}")
    print(f"처리 중: sess={sess}, num={num:04d}")
    print(f"{'='*60}")
    

    audio_path = find_audio_file(sess, num)
    if not audio_path:
        print(f"오디오 파일을 찾을 수 없음: sess={sess}, num={num:04d}")
        return False
    print(f"오디오 파일: {audio_path}")
    
    text_path = find_text_file(sess, num)
    if not text_path:
        print(f"텍스트 파일을 찾을 수 없음: sess={sess}, num={num:04d}")
        return False
    print(f"텍스트 파일: {text_path}")
    

    text = get_text_from_npz(text_path)
    print(f"텍스트: {text}")
    

    output_path = get_textgrid_output_path(sess, num)
    print(f"출력 경로: {output_path}")
    

    success = run_mfa_alignment(audio_path, text, output_path, overwrite)
    
    return success


def process_session(sess: str, overwrite: bool = True) -> Tuple[int, int]:
    """세션의 모든 파일 처리"""
    print(f"\n{'#'*60}")
    print(f"세션 처리 시작: {sess}")
    print(f"{'#'*60}")
    

    text_pattern = os.path.join(SILENT_BASE, sess, 'data/text', '*.npz')
    text_files = glob.glob(text_pattern)
    
    if not text_files:
        print(f"텍스트 파일이 없음: {sess}")
        return 0, 0
    
    success_count = 0
    fail_count = 0
    
    for text_path in sorted(text_files):

        filename = os.path.basename(text_path)

        import re
        match = re.search(r'_(\d{4})\.', filename)
        if match:
            num = int(match.group(1))
            if process_single(sess, num, overwrite):
                success_count += 1
            else:
                fail_count += 1
        else:
            print(f"파일명에서 번호를 추출할 수 없음: {filename}")
            fail_count += 1
    
    print(f"\n세션 {sess} 완료: 성공 {success_count}, 실패 {fail_count}")
    

    if success_count > 0:
        save_phoneme_set()
    
    return success_count, fail_count


def list_sessions() -> list:
    """사용 가능한 세션 목록 반환"""
    sessions = []
    if os.path.exists(VOICED_BASE):
        for item in os.listdir(VOICED_BASE):
            if os.path.isdir(os.path.join(VOICED_BASE, item)):
                sessions.append(item)
    return sorted(sessions)


def process_all_sessions(overwrite: bool = True) -> Tuple[int, int]:
    """모든 세션의 모든 파일 처리"""
    sessions = list_sessions()
    
    if not sessions:
        print("처리할 세션이 없습니다.")
        return 0, 0
    
    print(f"\n{'#'*60}")
    print(f"전체 세션 처리 시작: 총 {len(sessions)}개 세션")
    print(f"{'#'*60}")
    
    total_success = 0
    total_fail = 0
    
    for i, sess in enumerate(sessions, 1):
        print(f"\n[{i}/{len(sessions)}] 세션 처리 중: {sess}")
        success, fail = process_session(sess, overwrite)
        total_success += success
        total_fail += fail
    
    print(f"\n{'#'*60}")
    print(f"전체 처리 완료!")
    print(f"총 성공: {total_success}, 총 실패: {total_fail}")
    print(f"{'#'*60}")
    

    if total_success > 0:
        save_phoneme_set()
    
    return total_success, total_fail


def fix_spn_errors() -> Tuple[int, int, int]:
    """
    모든 세션의 TextGrid 파일을 검사하여 spn이 포함된 파일만 G2P + MFA 재실행
    
    Returns:
        (검사 수, 수정 성공 수, 수정 실패 수)
    """
    import re
    
    sessions = list_sessions()
    if not sessions:
        print("처리할 세션이 없습니다.")
        return 0, 0, 0
    
    total_checked = 0
    total_fixed = 0
    total_failed = 0
    total_skipped = 0
    
    print(f"\n{'#'*60}")
    print(f"spn 오류 수정 시작: 총 {len(sessions)}개 세션 검사")
    print(f"{'#'*60}")
    
    for sess in sessions:
        tg_dir = os.path.join(VOICED_BASE, sess, 'data/textgrid')
        if not os.path.exists(tg_dir):
            continue
        
        tg_pattern = os.path.join(tg_dir, 'tg_*.TextGrid')
        tg_files = sorted(glob.glob(tg_pattern))
        
        if not tg_files:
            continue
        
        sess_spn_count = 0
        
        for tg_path in tg_files:
            total_checked += 1
            
            if not textgrid_has_spn(tg_path):

                extract_phonemes_from_textgrid(tg_path)
                total_skipped += 1
                continue
            


            filename = os.path.basename(tg_path)
            match = re.search(r'tg_(\d{4})\.TextGrid', filename)
            if not match:
                print(f"파일명에서 번호를 추출할 수 없음: {filename}")
                total_failed += 1
                continue
            
            num = int(match.group(1))
            sess_spn_count += 1
            
            print(f"\n[spn 발견] sess={sess}, num={num:04d} → 재처리")
            
            if process_single(sess, num, overwrite=True):

                output_path = get_textgrid_output_path(sess, num)
                if textgrid_has_spn(output_path):
                    print(f"  경고: 재처리 후에도 spn 남아있음 (사전에 없는 단어일 수 있음)")
                    total_failed += 1
                else:
                    print(f"  수정 완료!")
                    total_fixed += 1
            else:
                total_failed += 1
        
        if sess_spn_count > 0:
            print(f"\n세션 {sess}: {len(tg_files)}개 검사, {sess_spn_count}개 spn 발견")
    
    print(f"\n{'#'*60}")
    print(f"spn 오류 수정 완료!")
    print(f"총 검사: {total_checked}, 정상: {total_skipped}, 수정 성공: {total_fixed}, 수정 실패: {total_failed}")
    print(f"{'#'*60}")
    

    if total_fixed > 0 or total_skipped > 0:
        save_phoneme_set()
    
    return total_checked, total_fixed, total_failed


def main():
    parser = argparse.ArgumentParser(description='Montreal Forced Alignment 전처리')
    parser.add_argument('--sess', type=str, default=None, help='세션 ID (예: 1-1), "all"로 모든 세션 처리')
    parser.add_argument('--num', type=int, default=None, help='파일 번호 (예: 0)')
    parser.add_argument('--all', action='store_true', help='세션의 모든 파일 처리')
    parser.add_argument('--no-overwrite', action='store_true', help='기존 파일 덮어쓰지 않음')
    parser.add_argument('--list-sessions', action='store_true', help='사용 가능한 세션 목록 출력')
    parser.add_argument('--show-phonemes', action='store_true', help='저장된 음소 집합 출력')
    parser.add_argument('--modify_error', action='store_true', help='모든 세션에서 spn 포함 TextGrid를 G2P+MFA로 재처리')
    
    args = parser.parse_args()
    
    if args.show_phonemes:
        phonemes = load_phoneme_set()
        if phonemes:
            print(f"저장된 음소 집합 ({len(phonemes)}개):")
            for i, p in enumerate(phonemes, 1):
                print(f"  {i:3d}. {p}")
        return
    
    if args.list_sessions:
        sessions = list_sessions()
        print("사용 가능한 세션:")
        for sess in sessions:
            print(f"  - {sess}")
        print(f"\n총 {len(sessions)}개 세션")
        return
    
    if args.modify_error:
        fix_spn_errors()
        return
    
    overwrite = not args.no_overwrite
    

    if args.sess == "all":
        process_all_sessions(overwrite)
        return
    
    if args.sess is None:
        print("--sess 옵션을 지정해주세요. (특정 세션 ID 또는 'all')")
        parser.print_help()
        return
    
    if args.all:

        success, fail = process_session(args.sess, overwrite)
        print(f"\n최종 결과: 성공 {success}, 실패 {fail}")
    elif args.num is not None:

        success = process_single(args.sess, args.num, overwrite)
        if success:
            save_phoneme_set()
            print("\n처리 완료!")
        else:
            print("\n처리 실패!")
    else:
        print("--num 또는 --all 옵션을 지정해주세요.")
        parser.print_help()


if __name__ == "__main__":
    main()
