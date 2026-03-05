from glob import glob
import hydra
import numpy as np
import os
from omegaconf import DictConfig
from scipy.io.wavfile import write
import sys
import torch
from tqdm import tqdm
import json

from save_output_gt import extract_filename

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from modules import Vocoder, HIFIGAN_VOCODER_CKPT_DEFAULT

@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg:DictConfig):
    device = 'cuda'

    sr = 16000
    try:
        config_file = os.path.join(os.path.split(HIFIGAN_VOCODER_CKPT_DEFAULT)[0], "config.json")
        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                hps = json.load(f)
            if isinstance(hps, dict) and "sampling_rate" in hps:
                sr = int(hps["sampling_rate"])
    except Exception:
        pass
    vocoder = Vocoder(device=device, half=False)


    #feat_encoder = str(getattr(cfg, "feat_encoder", cfg.encoder))
    feat_encoder = cfg.encoder



    target = str(getattr(cfg.feature, "target", "mspec")) if hasattr(cfg, "feature") else "mspec"
    root_candidates = sorted(
        glob(os.path.join(cfg.data_path, "preprocessed", "est_feature", target, "sr*", feat_encoder, cfg.data_split))
    )
    # prefer sr16000 for this script
    root = None
    for r in root_candidates:
        if os.path.basename(os.path.dirname(os.path.dirname(r))) == "sr16000":
            root = r
            break
    if root is None and len(root_candidates) > 0:
        root = root_candidates[0]
    if root is None:
        # fallback to legacy hard-coded location
        root = os.path.join(cfg.data_path, "preprocessed", "est_feature", "mspec", "sr16000", feat_encoder, cfg.data_split)

    path_list = []

    # 1) try cfg.data_type first (backward compatible)
    cfg_dt = None
    if hasattr(cfg, "data_type") and cfg.data_type is not None and str(cfg.data_type).strip() != "":
        cfg_dt = str(cfg.data_type)
        path_list.extend(glob(os.path.join(root, cfg_dt, "*", "*feat.npy")))

    # 2) if nothing found, auto-detect all data_type dirs under root
    if len(path_list) == 0 and os.path.isdir(root):
        for dt in sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]):
            path_list.extend(glob(os.path.join(root, dt, "*", "*feat.npy")))

    path_list = sorted(set(path_list))

    # backward-compat fallback (old hard-coded path)
    if len(path_list) == 0:
        legacy = os.path.join(cfg.data_path, f"preprocessed/est_feature/mspec/sr16000/{feat_encoder}/{cfg.data_split}/{cfg.data_type}/*/*feat.npy")
        path_list = sorted(glob(legacy))

    for path in tqdm(path_list):
        sess, idx = extract_filename(path)
        # .../{encoder}/{split}/{data_type}/{sess}/{idx}_feat.npy
        dt = path.split("/")[-3]
        output_dir = os.path.join(cfg.exp_path, cfg.encoder, "direct", cfg.data_split, dt)
        os.makedirs(output_dir, exist_ok=True)

        feat = np.load(path).astype(np.float32)
        audio = vocoder(torch.from_numpy(feat).to(device=device, dtype=torch.float32)).cpu().numpy()
        m = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
        if m > 0:
            audio32 = (audio / m).astype(np.float32)
        else:
            audio32 = audio.astype(np.float32)
        file_path = os.path.join(output_dir, f"{sess}_{idx}.wav")
        write(file_path, sr, audio32)


if __name__ == '__main__':
    main()
