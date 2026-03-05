import soundfile as sf
import numpy as np
from resampy import resample
import os, re
from typing import Dict
from numpy.typing import NDArray

def read_audio(path: str, target_sr: int = 16000)->NDArray[np.float32]:

    audio, sr = sf.read(path)

    if audio.ndim>1: 
        audio = choose_channel(audio)
    if sr != target_sr:
        audio = resample(audio, sr, target_sr)
    
    return audio

def choose_channel(audio: NDArray[np.float32])->NDArray[np.float32]:
    audio = np.transpose(audio)
    squared_audio = np.square(audio)
    squared_audio = np.sum(squared_audio, axis=1)
    audio = audio[np.argmax(squared_audio)]

    return audio

def prepare_grapheme(grapheme_path:str)->Dict[str, str]:
    with open(grapheme_path, 'r') as f:
        lines = f.readlines()

    grapheme_dict = {}
    for line in lines:
        splits = line.strip().split('|')
        normalized_grapheme = normalize_text(splits[1])
        grapheme_path = os.path.sep.join(splits[0].split('/')[1:])
        grapheme_dict[grapheme_path] = normalized_grapheme
        
    return grapheme_dict


def normalize_text(text:str)->str:
    text = text.lower()
    text = text.replace('-', ' ')
    text = re.sub(r'[^a-z ]+', r'', text)
    text = text.strip()

    return text
