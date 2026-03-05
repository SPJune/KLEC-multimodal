from glob import glob
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
import pickle
import sys
import torch
from tqdm import tqdm

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from modules import EMGEncoder
from loader import apply_to_all, subsample, notch_harmonics, remove_drift
from utils import extract_path_info

def permute_emg_channels(x, channel_indices):

    x_perm = x.clone()
    batch_size, time_steps, _ = x.shape

    for ch in channel_indices:
        for b in range(batch_size):
            perm = torch.randperm(time_steps)
            x_perm[b, :, ch] = x_perm[b, perm, ch]
    
    return x_perm
    
def get_best_ckpt(exp_name):
    pattern = f"/data1/marg/spjune/silent_speech/{exp_name}/epoch=*val_loss=*.ckpt"
    ckpt_files = glob(pattern)
    
    best_loss = float('inf')
    best_ckpt = None
    
    for ckpt_file in ckpt_files:
        epoch_loss_str = ckpt_file.split('val_loss=')[1].split('.ckpt')[0]
        try:
            loss = float(epoch_loss_str)
        except ValueError:
            continue
        if loss < best_loss:
            best_loss = loss
            best_ckpt = ckpt_file
    return best_ckpt

def get_adjacent_paths(path):
    base_dir, file_name = os.path.split(path)
    index = int(file_name.split('_')[0])
    before = os.path.join(base_dir, f"{index - 1}_emg.npy")
    after = os.path.join(base_dir, f"{index + 1}_emg.npy")
    return before, after

def load_emg(path, frame_rate):
    conv_ds = round(100/frame_rate)*8 # 8 for 66.7~200, 16 for 40~66.6
    resample_rate = int(frame_rate*conv_ds)
    raw_emg = np.load(path)
    before, after = get_adjacent_paths(path)
    if os.path.exists(before):
        raw_emg_before = np.load(before)
    else:
        raw_emg_before = np.zeros([0,raw_emg.shape[1]])
    if os.path.exists(after):
        raw_emg_after = np.load(after)
    else:
        raw_emg_after = np.zeros([0,raw_emg.shape[1]])

    x = np.concatenate([raw_emg_before, raw_emg, raw_emg_after], 0)
    x = apply_to_all(notch_harmonics, x, 60, 1000)
    x = apply_to_all(remove_drift, x, 1000)
    x = x[raw_emg_before.shape[0]:x.shape[0]-raw_emg_after.shape[0],:]
    raw_emg = apply_to_all(subsample, x, resample_rate, 1000)

    raw_emg = raw_emg / 20
    raw_emg = 50*np.tanh(raw_emg/50.)
    raw_emg = raw_emg.astype(np.float32)

    return raw_emg

@hydra.main(version_base=None, config_path="../configs", config_name="emg2feat")
def main(cfg:DictConfig):
    device = 'cuda'
    if cfg.ckpt_epoch == None or cfg.ckpt_epoch == 'last':
        checkpoint_path = os.path.join(cfg.exp_path, cfg.exp_name, 'last.ckpt')
    elif cfg.ckpt_epoch == 'best':
        checkpoint_path = get_best_ckpt(cfg.exp_name)

    else:
        checkpoint_path = glob(os.path.join(cfg.exp_path, cfg.exp_name, f'epoch={cfg.ckpt_epoch:02d}*'))[0]
    print(checkpoint_path)
    emg_enc = EMGEncoder.load_from_checkpoint(checkpoint_path).to(device)
    emg_enc.eval()
    feature = emg_enc.hparams.feature_config
    path_list = glob(os.path.join(cfg.data_path, 'silent_speech_dataset', f'{cfg.split}/*/*/*_emg.npy'))
    sub_option = f'{feature.sub_option}{feature[feature.sub_option]}'
    base_dir = os.path.join(cfg.data_path, 'preprocessed', 'target_feature', feature.target, sub_option)
    if feature.normalize:
        feat_norm, _ = pickle.load(open(os.path.join(base_dir, 'normalizer.pkl'),'rb'))

    for path in tqdm(path_list):
        data_split, data_type, sess, idx = extract_path_info(path)
        emg = load_emg(path, feature.frame_rate)
        emg = torch.tensor(emg).unsqueeze(0).to(device)
        if len(cfg.permute_channel) > 0:
            emg = permute_emg_channels(emg, cfg.permute_channel)
        f0_tensor, uv_tensor = None, None
        with torch.no_grad():
            feat, ph, _, _ = emg_enc(emg)
        path_save = os.path.join(cfg.data_path, 'preprocessed/est_feature', feature.target, sub_option, cfg.exp_name, data_split, data_type, sess, f'{idx}_feat.npy')
        path_ph = os.path.join(cfg.data_path, 'preprocessed/est_feature', feature.target, sub_option, cfg.exp_name, data_split, data_type, sess, f'{idx}_ph.npy')
        feat = feat[0].cpu().numpy()
        ph = ph[0].cpu().numpy()
        if feature.normalize:
            feat = feat_norm.inverse(feat)
        path_save_dir, _ = os.path.split(path_save)
        os.makedirs(path_save_dir, exist_ok=True)
        np.save(path_save, feat)
        np.save(path_ph, ph)

if __name__ == '__main__':
    main()
