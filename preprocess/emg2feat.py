from glob import glob
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
import pandas as pd
import pickle
import sys
import torch
from tqdm import tqdm
import re

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from modules import EMGEncoder
from loader import apply_to_all, subsample, notch_harmonics, remove_drift

def permute_emg_channels(x: torch.Tensor, channel_indices):
    """
    채널별로 time 차원을 랜덤 permute.
    x: (B, T, C)
    """
    if len(channel_indices) == 0:
        return x
    x_perm = x.clone()
    batch_size, time_steps, _ = x.shape
    for ch in channel_indices:
        for b in range(batch_size):
            perm = torch.randperm(time_steps, device=x.device)
            x_perm[b, :, ch] = x_perm[b, perm, ch]
    return x_perm
    
def _extract_metric_from_ckpt_path(ckpt_path: str, metric_name: str):
    """
    PL ModelCheckpoint 파일명에서 metric 값을 파싱.
    예) epoch=200-val_phone_accuracy=0.8123.ckpt
    """
    m = re.search(rf"{re.escape(metric_name)}=([-+0-9.eE]+)", ckpt_path)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _extract_metric_from_newstyle_ckpt_path(ckpt_path: str):
    """
    New-style ckpt name: "{epoch:02d}-{score:.4f}.ckpt"
    Returns (epoch:int, score:float) or (None, None) if not matched.
    """
    base = os.path.basename(str(ckpt_path))
    m = re.match(r"^(\d+)-([-+0-9.eE]+)\.ckpt$", base)
    if m is None:
        return None, None
    try:
        ep = int(m.group(1))
        score = float(m.group(2))
        return ep, score
    except ValueError:
        return None, None


def get_best_ckpt(exp_name, exp_path, metric_name: str = "val_phone_accuracy"):
    """
    exp_path/exp_name 아래의 체크포인트 중 metric_name 기준으로 best를 선택.
    기본은 val_phone_accuracy 최대값.

    구버전( val_loss 기반 파일명 )만 존재하는 경우에는 val_loss 최소값으로 fallback.
    """
    # 0) prefer new-style ckpts: "{epoch}-{score}.ckpt" where score is val_phone_accuracy_best
    # (saved by our current ModelCheckpoint naming policy)
    pattern_new = f"{exp_path}/{exp_name}/[0-9]*-*.ckpt"
    ckpt_files = [p for p in glob(pattern_new) if os.path.basename(p) != "last.ckpt" and "epoch=" not in os.path.basename(p)]
    best_score = float("-inf")
    best_ckpt = None
    for ckpt_file in ckpt_files:
        _, score = _extract_metric_from_newstyle_ckpt_path(ckpt_file)
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best_ckpt = ckpt_file
    if best_ckpt is not None:
        return best_ckpt

    # 1) prefer phone accuracy ckpts (maximize) - legacy filename contains metric key
    pattern_acc = f"{exp_path}/{exp_name}/epoch=*{metric_name}=*.ckpt"
    ckpt_files = glob(pattern_acc)
    best_score = float("-inf")
    best_ckpt = None

    for ckpt_file in ckpt_files:
        score = _extract_metric_from_ckpt_path(ckpt_file, metric_name)
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best_ckpt = ckpt_file

    if best_ckpt is not None:
        return best_ckpt

    # 2) fallback: legacy val_loss ckpts (minimize)
    legacy_metric = "val_loss"
    pattern_loss = f"{exp_path}/{exp_name}/epoch=*{legacy_metric}=*.ckpt"
    ckpt_files = glob(pattern_loss)
    best_loss = float("inf")
    best_ckpt = None

    for ckpt_file in ckpt_files:
        loss = _extract_metric_from_ckpt_path(ckpt_file, legacy_metric)
        if loss is None:
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

def _trim_time_series_torch(x: torch.Tensor, trim_left: int, trim_right: int | None = None, *, time_dim: int = 1) -> torch.Tensor:
    """
    시계열 텐서의 양끝을 자른다.
    - x: (B, T, ...) 또는 (T, ...)
    - trim_left/right: frame 단위
    - time_dim: time 차원 인덱스 (기본: (B,T,...) 가정으로 1)
    길이가 부족하면 빈 텐서(T=0)로 만든다.
    """
    if trim_right is None:
        trim_right = trim_left
    trim_left = int(max(0, trim_left))
    trim_right = int(max(0, trim_right))
    if trim_left == 0 and trim_right == 0:
        return x

    t = int(x.shape[time_dim])
    if t <= trim_left + trim_right:
        # empty along time dimension
        slc = [slice(None)] * x.ndim
        slc[time_dim] = slice(0, 0)
        return x[tuple(slc)]

    start = trim_left
    end = t - trim_right if trim_right > 0 else t
    slc = [slice(None)] * x.ndim
    slc[time_dim] = slice(start, end)
    return x[tuple(slc)]

def _center_trim_to_length_torch(x: torch.Tensor, target_len: int, *, time_dim: int = 1) -> torch.Tensor:
    """
    중앙 기준(center crop)으로 target_len(프레임)로 자른다.
    - 잘리는 길이는 좌/우 동일하게(홀수면 뒤쪽을 1프레임 더 자름)
    - x의 길이가 target_len 이하이면 그대로 반환
    """
    target_len = int(max(0, target_len))
    t = int(x.shape[time_dim])
    if t <= target_len:
        return x
    diff = t - target_len
    trim_left = diff // 2
    trim_right = diff - trim_left
    return _trim_time_series_torch(x, trim_left, trim_right, time_dim=time_dim)

def _align_emg_video(
    emg: torch.Tensor,
    video_feat: torch.Tensor,
    *,
    frame_rate: float,
    conv_ds: int = 4,
    video_fps: float = 25.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    EMG(=frame_rate*conv_ds Hz)와 video feature(=video_fps Hz)를 시간 기준으로 정렬한다.
    - 기준 time base: speech frame (=frame_rate Hz)로 환산한 길이
      emg_eq_len   = T_emg // conv_ds
      video_eq_len = T_vid * k  (k = frame_rate/video_fps 가 정수일 때)
    - 중앙 기준으로 trim하며, k가 정수일 때는 길이를 k의 배수로 내림해서 정확히 정렬 가능하게 함.
    """
    fr = float(frame_rate)
    vf = float(video_fps)
    conv_ds = int(conv_ds)
    if conv_ds <= 0:
        raise ValueError(f"conv_ds must be positive, got {conv_ds}")
    if fr <= 0 or vf <= 0:
        raise ValueError(f"frame_rate/video_fps must be positive, got frame_rate={fr}, video_fps={vf}")

    # shapes: (B, T, C)
    t_emg = int(emg.shape[1])
    t_vid = int(video_feat.shape[1])

    emg_eq_len = int(t_emg // conv_ds)

    ratio = fr / vf
    nearest = int(round(ratio))
    k = nearest if (nearest >= 1 and abs(ratio - float(nearest)) < 1e-6) else None

    if k is not None:
        video_eq_len = int(t_vid * k)
    else:
        # fallback: best-effort (shouldn't happen for 100/25)
        video_eq_len = int(np.floor(t_vid * fr / vf))

    min_eq_raw = int(max(0, min(emg_eq_len, video_eq_len)))
    if k is not None and k > 1:
        min_eq = (min_eq_raw // k) * k
    else:
        min_eq = min_eq_raw

    target_emg_len = int(min_eq * conv_ds)
    if k is not None:
        target_vid_len = int(min_eq // k)
    else:
        target_vid_len = int(np.floor(min_eq * vf / fr))

    emg = _center_trim_to_length_torch(emg, target_emg_len, time_dim=1)
    video_feat = _center_trim_to_length_torch(video_feat, target_vid_len, time_dim=1)
    return emg, video_feat

def load_emg(path, frame_rate):
    """
    loader.py의 EMGDataset.load_emg와 동일한 전처리:
    - 원본 250Hz(npz['data'] 또는 npy)을 notch+drift 제거 후
    - (frame_rate * 4) Hz로 리샘플(모델 conv stride=2 두 번(/4)로 frame_rate 정렬)
    """
    old_freq = 250
    conv_ds = 4
    resample_rate = float(frame_rate) * float(conv_ds)

    arr = np.load(path, allow_pickle=False)
    if isinstance(arr, np.lib.npyio.NpzFile):
        raw_emg = arr["data"]
    else:
        raw_emg = arr

    x = raw_emg
    x = apply_to_all(notch_harmonics, x, 60, old_freq)
    x = apply_to_all(remove_drift, x, old_freq)
    raw_emg = apply_to_all(subsample, x, resample_rate, old_freq)

    raw_emg = raw_emg / 20.0
    raw_emg = 50.0 * np.tanh(raw_emg / 50.0)

    return raw_emg.astype(np.float32)

def _parse_split_filter(split_value):
    if split_value is None:
        return None
    if isinstance(split_value, (list, tuple)):
        return set(str(s) for s in split_value)
    s = str(split_value).strip()
    if s == "*" or s == "":
        return None
    # allow comma-separated list: "train,valid"
    return set(x.strip() for x in s.split(",") if x.strip())

def _find_video_feature(data_path: str, data_type: str, session: str, data_num: int):
    """
    silent_speech_dataset/{data_type}/{session}/data/video_features 아래에서 *_<num>*.pth 찾기.
    """
    vf_dir = os.path.join(data_path, "silent_speech_dataset", data_type, session, "data", "video_features")
    if not os.path.isdir(vf_dir):
        return None
    patterns = [
        os.path.join(vf_dir, f"*_{data_num}*.pth"),
        os.path.join(vf_dir, f"*_{data_num:04d}*.pth"),
    ]
    matches = []
    for pat in patterns:
        matches.extend(glob(pat))
    matches = sorted(set(matches))
    if len(matches) == 0:
        return None
    return matches[0]

def _load_video_feature(path: str):
    if path is None or (not os.path.exists(path)):
        return None
    feat = torch.load(path, map_location="cpu")
    if isinstance(feat, np.ndarray):
        feat = torch.from_numpy(feat)
    if not isinstance(feat, torch.Tensor):
        raise TypeError(f"Unsupported video feature type at {path}: {type(feat)}")
    # expect (T, 1024) float
    if feat.ndim == 3 and feat.shape[0] == 1:
        feat = feat[0]
    return feat.to(dtype=torch.float32)

@hydra.main(version_base=None, config_path="../configs", config_name="emg2feat")
def main(cfg:DictConfig):
    device = 'cuda'
    if cfg.ckpt_epoch == None or cfg.ckpt_epoch == 'last':
        checkpoint_path = os.path.join(cfg.exp_path, cfg.exp_name, 'last.ckpt')
    elif cfg.ckpt_epoch == 'best':
        checkpoint_path = get_best_ckpt(cfg.exp_name, cfg.exp_path)

    else:
        # Support both legacy "epoch=XX*.ckpt" and new "{XX}-*.ckpt"
        try:
            ep = int(cfg.ckpt_epoch)
        except Exception:
            ep = None
        if ep is None:
            # user provided a raw glob or filename
            pat = os.path.join(cfg.exp_path, cfg.exp_name, str(cfg.ckpt_epoch))
            matches = sorted(glob(pat))
            if len(matches) == 0:
                raise FileNotFoundError(f"No checkpoint matches pattern: {pat}")
            checkpoint_path = matches[0]
        else:
            matches = sorted(glob(os.path.join(cfg.exp_path, cfg.exp_name, f'epoch={ep:02d}*')))
            if len(matches) == 0:
                matches = sorted(glob(os.path.join(cfg.exp_path, cfg.exp_name, f'{ep:02d}-*.ckpt')))
            if len(matches) == 0:
                raise FileNotFoundError(f"Checkpoint not found for ckpt_epoch={ep} under {cfg.exp_path}/{cfg.exp_name}")
            checkpoint_path = matches[0]
    print(checkpoint_path)
    emg_enc = EMGEncoder.load_from_checkpoint(checkpoint_path).to(device)
    emg_enc.eval()

    #save_name = str(getattr(cfg, "save_name", cfg.exp_name))
    save_name = cfg.exp_name
    feature = emg_enc.hparams.feature_config
    sub_option = f'{feature.sub_option}{feature[feature.sub_option]}'
    base_dir = os.path.join(cfg.data_path, 'preprocessed', 'target_feature', feature.target, sub_option)
    if feature.normalize:
        feat_norm, _ = pickle.load(open(os.path.join(base_dir, 'normalizer.pkl'),'rb'))

    # split list from preprocess/emg_split.csv (new format)
    split_filter = _parse_split_filter(cfg.split)
    split_csv = os.path.join(project_dir, "preprocess", "emg_split.csv")
    df = pd.read_csv(split_csv)
    if split_filter is not None:
        df = df[df["split"].astype(str).isin(split_filter)]
    rows = df.to_dict("records")

    need_video = bool(getattr(emg_enc, "use_video", False)) and str(getattr(emg_enc, "modality", "both")) in ("video_only", "both")

    edge_trim_sec = float(getattr(cfg, "edge_trim_sec", 0.2))
    conv_ds = 4
    video_fps = 25.0

    for row in tqdm(rows):
        data_split = str(row["split"])
        data_type = str(row["data_type"])
        sess = str(row["session"])
        idx = int(row["data_num"])

        emg_path = os.path.join(cfg.data_path, str(row["path"]))
        emg = load_emg(emg_path, feature.frame_rate)
        emg = torch.from_numpy(emg).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1, T_emg, C)
        if len(cfg.permute_channel) > 0:
            emg = permute_emg_channels(emg, cfg.permute_channel)
        video_feat = None
        video_padding_mask = None
        if need_video:
            vf_path = _find_video_feature(cfg.data_path, data_type, sess, idx)
            vf = _load_video_feature(vf_path)
            missing_vf = vf is None
            if vf is None:

                T_emg = int(emg.shape[1])
                T_out = max(1, int(np.ceil(T_emg / 4.0)))      # ~frame_rate
                T_vid = max(1, int(np.ceil(T_out / 4.0)))      # 25Hz
                vf = torch.zeros((T_vid, 1024), dtype=torch.float32)
            if vf.ndim != 2 or vf.shape[1] != 1024:
                raise ValueError(f"video feature has unexpected shape {tuple(vf.shape)} for {vf_path}")
            video_feat = vf.unsqueeze(0).to(device=device, dtype=torch.float32)  # (1, T_vid, 1024)
            # True = padding (V2SFlow convention)
            if missing_vf:
                video_padding_mask = torch.ones((1, video_feat.shape[1]), device=device, dtype=torch.bool)
            else:
                video_padding_mask = torch.zeros((1, video_feat.shape[1]), device=device, dtype=torch.bool)

            # ------------------------------------------------------------

            # ------------------------------------------------------------
            emg, video_feat = _align_emg_video(
                emg,
                video_feat,
                frame_rate=float(feature.frame_rate),
                conv_ds=conv_ds,
                video_fps=video_fps,
            )

            if video_padding_mask is not None:
                video_padding_mask = _center_trim_to_length_torch(video_padding_mask, int(video_feat.shape[1]), time_dim=1)

        # ------------------------------------------------------------

        # ------------------------------------------------------------
        if edge_trim_sec > 0:
            trim_emg = int(round(float(feature.frame_rate) * float(conv_ds) * edge_trim_sec))
            emg = _trim_time_series_torch(emg, trim_emg, time_dim=1)
            if need_video and video_feat is not None:
                trim_video = int(round(float(video_fps) * edge_trim_sec))
                video_feat = _trim_time_series_torch(video_feat, trim_video, time_dim=1)
                if video_padding_mask is not None:
                    video_padding_mask = _trim_time_series_torch(video_padding_mask, trim_video, time_dim=1)

        save_modality = str(getattr(cfg, "modality", "emg")).strip().lower()
        if save_modality not in ("emg", "video"):
            raise ValueError(f"cfg.modality must be 'emg' or 'video', got: {save_modality}")

        with torch.no_grad():
            feat, ph, feat_vid, ph_vid = emg_enc(
                emg,
                video_feat=video_feat,
                video_padding_mask=video_padding_mask,
            )

        if save_modality == "video":
            if feat_vid is None or ph_vid is None:
                raise ValueError(
                    "Requested modality=video, but model did not return feat_vid/ph_vid. "
                    "This requires a checkpoint trained with modality=both and fusion_method='ab' (and aux heads enabled)."
                )
            feat = feat_vid
            ph = ph_vid
        path_save = os.path.join(cfg.data_path, 'preprocessed/est_feature', feature.target, sub_option, save_name, data_split, data_type, sess, f'{idx}_feat.npy')
        path_ph = os.path.join(cfg.data_path, 'preprocessed/est_feature', feature.target, sub_option, save_name, data_split, data_type, sess, f'{idx}_ph.npy')
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
