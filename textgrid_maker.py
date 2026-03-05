"""
Korean TextGrid Generator
한국어 오디오 파일에 대한 TextGrid 파일을 생성하는 도구

주요 기능:
1. 한국어 텍스트를 음소로 변환 (G2P)
2. Montreal Forced Aligner를 사용한 음성-텍스트 정렬
3. TextGrid 파일 생성
"""

import os
import re
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional
import glob
import numpy as np
import librosa
import soundfile as sf
from jamo import h2j, j2hcj
import unicodedata
from korean_phonetics import AdvancedKoreanG2P


DATA_PATH = "/data2/ai_champion/silent_speech_dataset/voiced/1-1/data/audio"
SAMPLE = "audio_1_1_1040.wav_20250814_201705.flac"

class KoreanG2P:
    """한국어 Grapheme-to-Phoneme 변환기"""
    
    def __init__(self):

        self.consonant_map = {
            'ㄱ': 'g', 'ㄲ': 'kk', 'ㄴ': 'n', 'ㄷ': 'd', 'ㄸ': 'tt',
            'ㄹ': 'r', 'ㅁ': 'm', 'ㅂ': 'b', 'ㅃ': 'pp', 'ㅅ': 's',
            'ㅆ': 'ss', 'ㅇ': 'ng', 'ㅈ': 'j', 'ㅉ': 'jj', 'ㅊ': 'ch',
            'ㅋ': 'k', 'ㅌ': 't', 'ㅍ': 'p', 'ㅎ': 'h'
        }
        
        self.vowel_map = {
            'ㅏ': 'a', 'ㅐ': 'ae', 'ㅑ': 'ya', 'ㅒ': 'yae', 'ㅓ': 'eo',
            'ㅔ': 'e', 'ㅕ': 'yeo', 'ㅖ': 'ye', 'ㅗ': 'o', 'ㅘ': 'wa',
            'ㅙ': 'wae', 'ㅚ': 'oe', 'ㅛ': 'yo', 'ㅜ': 'u', 'ㅝ': 'wo',
            'ㅞ': 'we', 'ㅟ': 'wi', 'ㅠ': 'yu', 'ㅡ': 'eu', 'ㅢ': 'ui',
            'ㅣ': 'i'
        }
        

        self.final_consonant_map = {
            'ㄱ': 'k', 'ㄲ': 'k', 'ㄳ': 'k', 'ㄴ': 'n', 'ㄵ': 'n',
            'ㄶ': 'n', 'ㄷ': 't', 'ㄹ': 'l', 'ㄺ': 'k', 'ㄻ': 'm',
            'ㄼ': 'l', 'ㄽ': 'l', 'ㄾ': 'l', 'ㄿ': 'p', 'ㅀ': 'l',
            'ㅁ': 'm', 'ㅂ': 'p', 'ㅄ': 'p', 'ㅅ': 't', 'ㅆ': 't',
            'ㅇ': 'ng', 'ㅈ': 't', 'ㅊ': 't', 'ㅋ': 'k', 'ㅌ': 't',
            'ㅍ': 'p', 'ㅎ': 't'
        }
    
    def is_korean(self, char: str) -> bool:
        """한글 문자인지 확인"""
        return '가' <= char <= '힣'
    
    
    def syllable_to_phonemes(self, syllable: str) -> List[str]:
        """한 음절을 음소로 변환"""
        if not self.is_korean(syllable):
            return [syllable]
        

        decomposed = j2hcj(h2j(syllable))
        phonemes = []
        
        for i, jamo in enumerate(decomposed):
            if jamo in self.consonant_map:
                if i == 0:
                    phonemes.append(self.consonant_map[jamo])
                else:
                    phonemes.append(self.final_consonant_map.get(jamo, self.consonant_map[jamo]))
            elif jamo in self.vowel_map:
                phonemes.append(self.vowel_map[jamo])
        
        return phonemes
    
    def text_to_phonemes(self, text: str) -> List[str]:
        """텍스트를 음소 리스트로 변환"""

        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'[^\w\s가-힣]', '', text)
        
        phonemes = ['sil']
        words = text.split()
        
        for word in words:
            word_phonemes = []
            for char in word:
                if self.is_korean(char):
                    word_phonemes.extend(self.syllable_to_phonemes(char))
                elif char.isalpha():
                    word_phonemes.append(char.lower())
            
            if word_phonemes:
                phonemes.extend(word_phonemes)
                phonemes.append('sil')
        

        '''
        if phonemes and phonemes[-1] == 'sil':
            phonemes.pop()
        '''
        
        return phonemes

class TextGridGenerator:
    """TextGrid 파일 생성기"""
    
    def __init__(self, mfa_path: Optional[str] = None, use_advanced_g2p: bool = True):
        if use_advanced_g2p:
            self.g2p = AdvancedKoreanG2P()
        else:
            self.g2p = KoreanG2P()
        self.mfa_path = mfa_path or "mfa"
        self.use_advanced_g2p = use_advanced_g2p
    
    def prepare_audio(self, audio_path: str, output_path: str) -> str:
        """오디오 파일을 MFA 호환 형식으로 변환"""

        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        

        output_wav = output_path.replace('.flac', '.wav').replace('.mp3', '.wav')
        sf.write(output_wav, y, sr)
        
        return output_wav
    
    def create_pronunciation_dict(self, text: str, dict_path: str):
        """발음 사전 생성"""
        words = set(text.split())
        
        with open(dict_path, 'w', encoding='utf-8') as f:
            for word in words:
                if word.strip():
                    if self.use_advanced_g2p:
                        phonemes = self.g2p.text_to_phonemes(word, apply_rules=True)
                    else:
                        phonemes = self.g2p.text_to_phonemes(word)
                    if phonemes:
                        f.write(f"{word}\t{' '.join(phonemes)}\n")
    
    def run_mfa_alignment(self, audio_dir: str, text_file: str, dict_file: str, output_dir: str) -> bool:
        """MFA를 사용한 강제 정렬 실행"""
        try:
            cmd = [
                self.mfa_path, "align",
                audio_dir,
                dict_file,
                "korean_mfa",
                output_dir,
                "--clean"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"MFA 정렬 실패: {e}")
            print(f"Error output: {e.stderr}")
            return False
    
    def create_simple_textgrid(self, audio_path: str, text: str, output_path: str) -> bool:
        """간단한 TextGrid 생성 (MFA 없이, 균등 분할)"""
        try:

            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            

            words = text.split()
            if not words:
                return False
            

            word_duration = duration / len(words)
            

            textgrid_content = self._create_textgrid_content(words, word_duration, duration)
            

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(textgrid_content)
            
            return True
        except Exception as e:
            print(f"간단한 TextGrid 생성 실패: {e}")
            return False
    
    def _create_textgrid_content(self, words: List[str], word_duration: float, total_duration: float) -> str:
        """TextGrid 파일 내용 생성"""
        content = []
        content.append('File type = "ooTextFile"')
        content.append('Object class = "TextGrid"')
        content.append('')
        content.append('xmin = 0.0')
        content.append(f'xmax = {total_duration:.3f}')
        content.append('tiers? <exists>')
        content.append('size = 2')
        content.append('item []:')
        
        # Words tier
        content.append('\titem [1]:')
        content.append('\t\tclass = "IntervalTier"')
        content.append('\t\tname = "words"')
        content.append('\t\txmin = 0.0')
        content.append(f'\t\txmax = {total_duration:.3f}')
        content.append(f'\t\tintervals: size = {len(words)}')
        
        for i, word in enumerate(words):
            start_time = i * word_duration
            end_time = (i + 1) * word_duration
            content.append(f'\t\t\tintervals [{i+1}]:')
            content.append(f'\t\t\t\txmin = {start_time:.3f}')
            content.append(f'\t\t\t\txmax = {end_time:.3f}')
            content.append(f'\t\t\t\ttext = "{word}"')
        
        # Phones tier (simplified)
        if self.use_advanced_g2p:
            phonemes = self.g2p.text_to_phonemes(' '.join(words), apply_rules=True)
        else:
            phonemes = self.g2p.text_to_phonemes(' '.join(words))
        
        content.append('\titem [2]:')
        content.append('\t\tclass = "IntervalTier"')
        content.append('\t\tname = "phones"')
        content.append('\t\txmin = 0.0')
        content.append(f'\t\txmax = {total_duration:.3f}')
        content.append(f'\t\tintervals: size = {len(phonemes)}')
        
        if phonemes:
            phone_duration = total_duration / len(phonemes)
            for i, phoneme in enumerate(phonemes):
                start_time = i * phone_duration
                end_time = (i + 1) * phone_duration
                content.append(f'\t\t\tintervals [{i+1}]:')
                content.append(f'\t\t\t\txmin = {start_time:.3f}')
                content.append(f'\t\t\t\txmax = {end_time:.3f}')
                content.append(f'\t\t\t\ttext = "{phoneme}"')
        
        return '\n'.join(content)
    
    def generate_textgrid(self, audio_path: str, text: str, output_path: str, 
                         use_mfa: bool = True, fallback_to_simple: bool = True) -> bool:
        """오디오와 텍스트로부터 TextGrid 생성"""
        if use_mfa:

            success = self._generate_textgrid_with_mfa(audio_path, text, output_path)
            if success:
                return True
            elif not fallback_to_simple:
                return False
        

        print("MFA 정렬 실패 또는 비활성화. 간단한 TextGrid 생성을 시도합니다.")
        return self.create_simple_textgrid(audio_path, text, output_path)
    
    def _generate_textgrid_with_mfa(self, audio_path: str, text: str, output_path: str) -> bool:
        """MFA를 사용한 TextGrid 생성"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            

            audio_name = Path(audio_path).stem
            prepared_audio = self.prepare_audio(audio_path, str(temp_path / f"{audio_name}.wav"))
            

            text_file = temp_path / f"{audio_name}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text.strip())
            

            dict_file = temp_path / "pronunciation.dict"
            self.create_pronunciation_dict(text, str(dict_file))
            

            mfa_input_dir = temp_path / "mfa_input"
            mfa_output_dir = temp_path / "mfa_output"
            mfa_input_dir.mkdir()
            mfa_output_dir.mkdir()
            

            shutil.copy(prepared_audio, mfa_input_dir)
            shutil.copy(text_file, mfa_input_dir)
            

            success = self.run_mfa_alignment(
                str(mfa_input_dir),
                str(text_file),
                str(dict_file),
                str(mfa_output_dir)
            )
            
            if success:

                textgrid_file = mfa_output_dir / f"{audio_name}.TextGrid"
                if textgrid_file.exists():
                    shutil.copy(textgrid_file, output_path)
                    return True
            
            return False

def main():
    """메인 실행 함수"""
    generator = TextGridGenerator()
    

    base_path = "/data2/ai_champion/silent_speech_dataset/voiced"
    audio_pattern = os.path.join(base_path, "*", "data", "audio", "*_[0-9][0-9][0-9][0-9].*.flac")
    
    print("한국어 TextGrid 생성 시작...")
    print(f"검색 패턴: {audio_pattern}")
    

    audio_files = glob.glob(audio_pattern)
    
    if not audio_files:
        print("해당 패턴에 맞는 오디오 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(audio_files)}개의 오디오 파일을 찾았습니다.")
    

    success_count = 0
    fail_count = 0
    
    for audio_file in audio_files:
        try:

            filename = os.path.basename(audio_file)
            match = re.search(r'_(\d{4})\.', filename)
            
            if not match:
                print(f"경고: 파일명에서 4자리 숫자를 찾을 수 없습니다: {audio_file}")
                fail_count += 1
                continue
            
            number = match.group(1)
            

            # /data2/ai_champion/silent_speech_dataset/voiced/*/data/textgrid
            audio_dir = os.path.dirname(audio_file)

            textgrid_dir = audio_dir.replace('/audio', '/textgrid')
            

            os.makedirs(textgrid_dir, exist_ok=True)
            

            output_file = os.path.join(textgrid_dir, f"tg_{number}.TextGrid")
            

            '''
            if os.path.exists(output_file):
                print(f"스킵 (이미 존재): {output_file}")
                continue
            '''
            
            print(f"\n처리 중: {audio_file}")
            print(f"출력: {output_file}")
            



            # -> /data2/ai_champion/silent_speech_dataset/silent/1-1/data/text
            audio_dir = os.path.dirname(audio_file)
            text_dir = audio_dir.replace('/voiced/', '/silent/').replace('/audio', '/text')
            

            npz_pattern = os.path.join(text_dir, f"text_*_{number}.*.npz")
            npz_files = glob.glob(npz_pattern)
            

            if not npz_files:
                print(f"\n❌ 오류: npz 파일을 찾을 수 없습니다!")
                print(f"   오디오 파일: {audio_file}")
                print(f"   검색 패턴: {npz_pattern}")
                print(f"   텍스트 디렉토리: {text_dir}")
                raise FileNotFoundError(f"npz 파일을 찾을 수 없습니다: {npz_pattern}")
            

            npz_file = npz_files[0]
            try:
                npz_data = np.load(npz_file, allow_pickle=True)
                if 'text2' not in npz_data:
                    print(f"\n❌ 오류: npz 파일에 'text2' 키가 없습니다!")
                    print(f"   오디오 파일: {audio_file}")
                    print(f"   npz 파일: {npz_file}")
                    print(f"   사용 가능한 키: {list(npz_data.keys())}")
                    raise KeyError(f"npz 파일에 'text2' 키가 없습니다: {npz_file}")
                
                sample_text = str(npz_data['text2']).strip()
                if not sample_text:
                    print(f"\n❌ 오류: 'text2' 값이 비어있습니다!")
                    print(f"   오디오 파일: {audio_file}")
                    print(f"   npz 파일: {npz_file}")
                    raise ValueError(f"'text2' 값이 비어있습니다: {npz_file}")
                
                print(f"텍스트 로드 성공: {npz_file}")
            except (KeyError, ValueError) as e:

                raise
            except Exception as e:
                print(f"\n❌ 오류: npz 파일 로드 실패!")
                print(f"   오디오 파일: {audio_file}")
                print(f"   npz 파일: {npz_file}")
                print(f"   오류 내용: {e}")
                raise RuntimeError(f"npz 파일 로드 실패 ({npz_file}): {e}") from e
            

            success = generator.generate_textgrid(audio_file, sample_text, output_file, use_mfa=False)
            
            if success:
                print(f"✓ 성공: {output_file}")
                success_count += 1
            else:
                print(f"✗ 실패: {audio_file}")
                fail_count += 1
                
        except (FileNotFoundError, KeyError, ValueError, RuntimeError) as e:

            print(f"\n프로그램을 종료합니다.")
            sys.exit(1)
        except Exception as e:
            print(f"✗ 오류 발생 ({audio_file}): {e}")
            fail_count += 1
    
    print(f"\n=== 처리 완료 ===")
    print(f"성공: {success_count}개")
    print(f"실패: {fail_count}개")
    print(f"전체: {len(audio_files)}개")

if __name__ == "__main__":
    main()
