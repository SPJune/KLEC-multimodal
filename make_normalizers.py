import os
import sys
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class WelfordState:
    count: int
    mean: np.ndarray
    m2: np.ndarray


def _welford_init(dim: int) -> WelfordState:
    mean = np.zeros((dim,), dtype=np.float64)
    m2 = np.zeros((dim,), dtype=np.float64)
    return WelfordState(count=0, mean=mean, m2=m2)


def _welford_update(state: WelfordState, x: np.ndarray) -> WelfordState:
    """
    Update Welford state with a batch x of shape (N, D).
    Uses the parallel (batch) update formulation for efficiency.
    """
    if x.size == 0:
        return state
    if x.ndim != 2:
        raise ValueError(f"Expected x with shape (N, D), got {x.shape}")

    x = x.astype(np.float64, copy=False)
    batch_count = int(x.shape[0])
    batch_mean = x.mean(axis=0)
    batch_m2 = ((x - batch_mean) ** 2).sum(axis=0)

    if state.count == 0:
        return WelfordState(count=batch_count, mean=batch_mean, m2=batch_m2)

    total = state.count + batch_count
    delta = batch_mean - state.mean
    new_mean = state.mean + delta * (batch_count / total)
    new_m2 = state.m2 + batch_m2 + (delta**2) * (state.count * batch_count / total)
    return WelfordState(count=total, mean=new_mean, m2=new_m2)


def _finalize(state: WelfordState, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    if state.count <= 0:
        raise ValueError("No samples were accumulated; cannot finalize normalizer.")
    var = state.m2 / float(state.count)
    std = np.sqrt(var)
    std = np.maximum(std, eps)
    return state.mean.astype(np.float32), std.astype(np.float32)


def _resolve_preprocessed_dir(cfg) -> str:
    """
    Resolve:
      {data_path}/preprocessed/target_feature/{feature.target}/{sub_option}{value}
    """
    feature = cfg.feature
    sub_option = str(feature.sub_option)
    if sub_option not in feature:
        raise KeyError(f"cfg.feature has no key '{sub_option}'. cfg.feature={feature}")
    value = feature[sub_option]
    return os.path.join(
        str(cfg.data_path),
        "preprocessed",
        "target_feature",
        str(feature.target),
        f"{sub_option}{value}",
    )


def _load_cfg(config_name: str = "config"):
    # Use Hydra compose without changing working directory.
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    project_dir = os.path.abspath(os.path.dirname(__file__))
    config_dir = os.path.join(project_dir, "configs")
    # In case this function is called multiple times in a process.
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
    return cfg


def main(max_items: Optional[int] = None):
    project_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(project_dir)

    from data_utils import FeatureNormalizer
    from loader import EMGDataset

    cfg = _load_cfg("config")
    preprocessed_dir = _resolve_preprocessed_dir(cfg)
    normalizer_pkl_path = os.path.join(preprocessed_dir, "normalizer.pkl")
    normalizer_npz_path = os.path.join(preprocessed_dir, "normalizer.npz")

    frame_rate = float(cfg.feature.frame_rate)
    dataset = EMGDataset(
        preprocessed_dir,
        target=str(cfg.feature.target),
        data_split="train",
        frame_rate=frame_rate,
        target_sec=None,
        normalize=False,
        seq_len=None,
    )

    # ---- feature normalizer (prefer existing normalizer.npz from speech2feat.py) ----
    if os.path.exists(normalizer_npz_path):
        npz = np.load(normalizer_npz_path)
        feat_mean = np.array(npz["mean"], dtype=np.float32)
        feat_std = np.array(npz["std"], dtype=np.float32)
    else:
        # Fallback: compute from dataset speech features
        feat_state = _welford_init(dim=int(cfg.feature.dim))
        for i in range(len(dataset)):
            if max_items is not None and i >= max_items:
                break
            speech_feature, _, _, _, _, _ = dataset[i]
            sf = speech_feature.detach().cpu().numpy()
            feat_state = _welford_update(feat_state, sf.reshape(-1, sf.shape[-1]))
        feat_mean, feat_std = _finalize(feat_state)
        os.makedirs(preprocessed_dir, exist_ok=True)
        np.savez(normalizer_npz_path, mean=feat_mean, std=feat_std)

    feat_norm = FeatureNormalizer(feat_mean, feat_std)

    # ---- emg normalizer (compute from dataset EMG) ----
    emg_state: Optional[WelfordState] = None
    for i in range(len(dataset)):
        if max_items is not None and i >= max_items:
            break

        _, emg, _, silent, voiced_emg, _ = dataset[i]
        emg_np = emg.detach().cpu().numpy()
        if emg_state is None:
            emg_state = _welford_init(dim=emg_np.shape[-1])
        emg_state = _welford_update(emg_state, emg_np.reshape(-1, emg_np.shape[-1]))

        # Silent sample has paired voiced_emg (from voiced session); include it as well.
        if bool(silent) and voiced_emg is not None:
            v_np = voiced_emg.detach().cpu().numpy()
            if v_np.size > 0:
                emg_state = _welford_update(emg_state, v_np.reshape(-1, v_np.shape[-1]))

    if emg_state is None:
        raise RuntimeError("Failed to accumulate EMG stats (empty dataset?)")
    emg_mean, emg_std = _finalize(emg_state)
    emg_norm = FeatureNormalizer(emg_mean, emg_std)

    os.makedirs(preprocessed_dir, exist_ok=True)
    with open(normalizer_pkl_path, "wb") as f:
        pickle.dump((feat_norm, emg_norm), f)

    print(f"[ok] wrote: {normalizer_pkl_path}")
    print(f"[info] feature_dir: {preprocessed_dir}")
    print(f"[info] feat: mean/std shape = {feat_mean.shape} / {feat_std.shape}")
    print(f"[info] emg : mean/std shape = {emg_mean.shape} / {emg_std.shape}")


if __name__ == "__main__":
    # Optional speed knob: export MAX_ITEMS=200 to limit scanning.
    max_items_env = os.environ.get("MAX_ITEMS", "").strip()
    max_items_val = int(max_items_env) if max_items_env else None
    main(max_items=max_items_val)

