#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AV-HuBERT feature extractor (video -> continuous features).

This module uses the local `av_hubert` repo (vendored fairseq + avhubert code) and a checkpoint
to extract per-frame features, typically (T, 1024) for AV-HuBERT Large.

Checkpoint (default):
  /data2/spjune/v2sflow/large_lrs3_iter5.pt

Input video expectations:
  - Preprocessed mouth ROI video (grayscale), 25fps, typically 96x96.
  - We apply the same normalization scheme used by AV-HuBERT:
      1) frames / 255
      2) center-crop to 88x88 (task.cfg.image_crop_size)
      3) (frames - mean) / std, where mean/std come from task.cfg (defaults 0.421/0.165)

Output:
  torch.FloatTensor of shape (T, 1024) on CPU by default.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import cv2
import torch


def _patch_numpy_legacy_aliases():
    # fairseq / older code may use these aliases; NumPy>=1.24 removed them.
    np.float = float  # type: ignore[attr-defined]
    np.int = int  # type: ignore[attr-defined]
    np.bool = bool  # type: ignore[attr-defined]
    np.complex = complex  # type: ignore[attr-defined]


def _patch_omegaconf_legacy_api():
    # older fairseq expects omegaconf._utils.is_primitive_type
    from omegaconf import _utils as oc_utils

    if not hasattr(oc_utils, "is_primitive_type"):
        oc_utils.is_primitive_type = lambda x: x is None or isinstance(x, (bool, int, float, str))  # type: ignore[attr-defined]


def _ensure_avhubert_on_path(project_root: str):
    avhubert_root = os.path.join(project_root, "av_hubert")
    fairseq_root = os.path.join(avhubert_root, "fairseq")
    avhubert_code_root = os.path.join(avhubert_root, "avhubert")

    # Put vendored fairseq first to avoid picking up unrelated fairseq installs.
    for p in [fairseq_root, avhubert_code_root]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_grayscale_frames_uint8(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames read from: {video_path}")

    return np.stack(frames).astype(np.uint8)  # (T,H,W)


def _center_crop(frames: np.ndarray, crop_size: int) -> np.ndarray:
    # frames: (T,H,W)
    t, h, w = frames.shape
    if h < crop_size or w < crop_size:
        raise ValueError(f"Cannot center-crop {crop_size} from frames of size {h}x{w}")
    dh = (h - crop_size) // 2
    dw = (w - crop_size) // 2
    return frames[:, dh : dh + crop_size, dw : dw + crop_size]


@dataclass(frozen=True)
class AVHuBERTExtractorConfig:
    #ckpt_path: str = "/data2/spjune/v2sflow/large_vox_iter5.pt" # LRS3 + Vox Noise augmented
    #ckpt_path: str = "/data2/spjune/v2sflow/large_vox.pt" # LRS3 + Vox
    ckpt_path: str = "/data2/spjune/v2sflow/large_lrs3_iter5.pt" # LRS3
    # output_layer: None -> final layer. Set integer (1-based) to get intermediate layer.
    output_layer: Optional[int] = None
    device: str = "cuda"


class AVHuBERTFeatureExtractor:
    """
    Extract (T,1024) continuous features from a preprocessed mouth ROI video.
    """

    _CACHE: Dict[Tuple[str, str], Tuple[torch.nn.Module, object]] = {}

    def __init__(self, cfg: Optional[AVHuBERTExtractorConfig] = None):
        self.cfg = cfg or AVHuBERTExtractorConfig()

        if self.cfg.device.startswith("cuda") and not torch.cuda.is_available():
            self.cfg = AVHuBERTExtractorConfig(
                ckpt_path=self.cfg.ckpt_path,
                output_layer=self.cfg.output_layer,
                device="cpu",
            )

        _patch_numpy_legacy_aliases()
        _patch_omegaconf_legacy_api()

        # Resolve project root from this file location.
        project_root = os.path.dirname(os.path.abspath(__file__))
        _ensure_avhubert_on_path(project_root)

        self.model, self.task = self._get_or_load(self.cfg.ckpt_path, self.cfg.device)

    @classmethod
    def _get_or_load(cls, ckpt_path: str, device: str):
        key = (ckpt_path, device)
        if key in cls._CACHE:
            return cls._CACHE[key]

        # Import registrations.
        # NOTE: avhubert/hubert.py switches between absolute vs relative imports based on len(sys.argv).
        # We force len(sys.argv)==1 during import to take the absolute-import path (works without package context).
        argv_bak = list(sys.argv)
        try:
            sys.argv = [""]  # ensure DBG=True in avhubert code
            import hubert  # noqa: F401
            import hubert_pretraining  # noqa: F401
        finally:
            sys.argv = argv_bak

        from fairseq import checkpoint_utils

        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        model = models[0].eval()
        if device.startswith("cuda"):
            model = model.cuda()
        else:
            model = model.cpu()

        cls._CACHE[key] = (model, task)
        return model, task

    def _preprocess_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        frames: (T,H,W) uint8 or float in [0,1] or [0,255]
        returns: (T, crop, crop) float32 normalized
        """
        frames_f = frames.astype(np.float32)
        if frames_f.max() > 2.0:
            frames_f = frames_f / 255.0

        crop_size = int(getattr(self.task.cfg, "image_crop_size", 88))
        frames_f = _center_crop(frames_f, crop_size)

        mean = float(getattr(self.task.cfg, "image_mean", 0.421))
        std = float(getattr(self.task.cfg, "image_std", 0.165))
        frames_f = (frames_f - mean) / std
        return frames_f.astype(np.float32)

    @torch.no_grad()
    def extract_from_video_path(self, video_path: str) -> torch.Tensor:
        frames = _load_grayscale_frames_uint8(video_path)
        return self.extract_from_frames(frames)

    @torch.no_grad()
    def extract_from_frames(self, frames: np.ndarray) -> torch.Tensor:
        """
        frames: (T,H,W) grayscale
        returns: (T,1024) float32 on CPU
        """
        frames_f = self._preprocess_frames(frames)  # (T,88,88)
        video = torch.from_numpy(frames_f).unsqueeze(0).unsqueeze(0)  # (1,1,T,H,W)

        T = video.shape[2]
        # AV-HuBERT uses stacked logfbank features. Default: 26 * stack_order_audio (usually 4) = 104.
        stack_order_audio = int(getattr(self.task.cfg, "stack_order_audio", 4))
        audio_feat_dim = 26 * stack_order_audio
        audio = torch.zeros((1, audio_feat_dim, T), dtype=torch.float32)

        if self.cfg.device.startswith("cuda"):
            video = video.cuda(non_blocking=True)
            audio = audio.cuda(non_blocking=True)

        source = {"audio": audio, "video": video}
        feat, _ = self.model.extract_features(
            source=source,
            padding_mask=None,
            mask=False,
            output_layer=self.cfg.output_layer,
            ret_conv=False,
        )
        feat = feat.squeeze(0).to(dtype=torch.float32).contiguous()  # (T,1024)
        return feat.cpu()

    @torch.no_grad()
    def extract_from_preprocessed_video_tensor(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        video_tensor: (T,H,W) torch tensor (float in [0,1] or [0,255])
        returns: (T,1024) float32 on CPU
        """
        frames = video_tensor.detach().cpu().numpy()
        return self.extract_from_frames(frames)

