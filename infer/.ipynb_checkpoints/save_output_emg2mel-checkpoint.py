from glob import glob
import hydra
import numpy as np
import os
from omegaconf import DictConfig, OmegaConf
import pickle
from scipy.io.wavfile import write
import sys
import torch
from tqdm import tqdm

from save_output_gt import extract_filename

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from modules import Vocoder

@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg:DictConfig):
    device = 'cuda'
    sr22 = 22050
    vocoder = Vocoder(device=device, half=False)
    output_dir = os.path.join(cfg.exp_path, cfg.encoder, 'direct', cfg.data_split, cfg.data_type)
    os.makedirs(output_dir, exist_ok=True)
    normalizer_file = os.path.join(cfg.data_path, f'preprocessed/target_feature/mspec/sr22050/normalizer.pkl')
    feat_norm, emg_norm = pickle.load(open(normalizer_file,'rb'))
    path_list = glob(os.path.join(cfg.data_path, f'preprocessed/est_feature/mspec/sr22050/{cfg.encoder}/{cfg.data_split}/{cfg.data_type}/*/*feat.npy'))
    path_list.sort()
    for i, path in enumerate(tqdm(path_list)):
        sess, idx = extract_filename(path)
        feat = np.load(path)
        audio = vocoder(torch.tensor(feat).to(device)).cpu().numpy()
        m = np.max(np.abs(audio))
        audio32 = (audio/m).astype(np.float32)
        file_path = os.path.join(output_dir, f'{sess}_{idx}.wav')
        write(file_path, sr22, audio32)


if __name__ == '__main__':
    main()
