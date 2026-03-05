#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract video features from *preprocessed* videos and save them for downstream use.

Input:
  /data2/ai_champion/silent_speech_dataset/{data_type}/{sess}/data/video_preprocessed/*.mp4

Output:
  /data2/ai_champion/silent_speech_dataset/{data_type}/{sess}/data/video_features/*.pth

Notes
- V2SFlow expects 25Hz continuous AV-HuBERT-Large features (typically 1024-d per frame).
- This script uses the local `av_hubert` repository + the checkpoint at
  `/data2/spjune/v2sflow/large_lrs3_iter5.pt` to extract features.

Usage examples:
  # Extract all videos in a session
  python preprocess/pp_video2feat.py --data-type voiced --sess 1-1

  # Extract a single sample number (e.g. 0000)
  python preprocess/pp_video2feat.py --data-type voiced --sess 1-1 --num 0

  # Extract ALL videos (all data_type / sessions)
  python preprocess/pp_video2feat.py --all
"""

import os
import sys
import argparse
from glob import glob
from typing import Optional, List, Tuple

import numpy as np
import cv2
import torch
from tqdm import tqdm

# Add project root to import path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from avhubert_feature_extractor import AVHuBERTFeatureExtractor, AVHuBERTExtractorConfig


DEFAULT_BASE_PATH = "/data2/ai_champion/silent_speech_dataset"


def _video_preprocessed_glob(base_path: str, data_type: str, sess: str, num: Optional[int]) -> List[str]:
    video_dir = os.path.join(base_path, data_type, sess, "data", "video_preprocessed")
    if num is None:
        pattern = os.path.join(video_dir, "*.mp4")
    else:
        pattern = os.path.join(video_dir, f"*_{num:04d}.*.mp4")
    return sorted(glob(pattern))


def _video_preprocessed_glob_all(
    base_path: str,
    data_type: Optional[str],
    sess: Optional[str],
    num: Optional[int],
) -> List[str]:
    """
    Scan dataset for:
      {base_path}/{data_type}/{sess}/data/video_preprocessed/*.mp4
    with optional filters.
    """
    if data_type is not None and sess is not None:
        return _video_preprocessed_glob(base_path, data_type, sess, num)

    if data_type is None:
        dt_glob = "*"
    else:
        dt_glob = data_type

    if sess is None:
        sess_glob = "*"
    else:
        sess_glob = sess

    video_dir = os.path.join(base_path, dt_glob, sess_glob, "data", "video_preprocessed")
    if num is None:
        pattern = os.path.join(video_dir, "*.mp4")
    else:
        pattern = os.path.join(video_dir, f"*_{num:04d}.*.mp4")

    return sorted(glob(pattern))


def _feature_out_path_from_video(video_path: str) -> str:
    # /.../data/video_preprocessed/xxx.mp4 -> /.../data/video_features/xxx.pth
    out_path = video_path.replace("/data/video_preprocessed/", "/data/video_features/")
    out_path = os.path.splitext(out_path)[0] + ".pth"
    return out_path


def load_preprocessed_video_as_tensor(video_path: str) -> torch.Tensor:
    """
    Load preprocessed video (96x96 grayscale mouth crop, 25fps) as float tensor in [0,1].
    Returns: (T, 96, 96) float32
    """
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

    video = np.asarray(frames, dtype=np.float32) / 255.0
    return torch.from_numpy(video)  # (T,H,W)


def extract_and_save_one(
    video_path: str,
    extractor: AVHuBERTFeatureExtractor,
    overwrite: bool = False,
) -> Tuple[str, str]:
    out_path = _feature_out_path_from_video(video_path)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    if (not overwrite) and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return video_path, out_path

    # Use AV-HuBERT extractor (keeps preprocessing consistent with checkpoint task cfg)
    feats = extractor.extract_from_video_path(video_path)  # (T, 1024) on CPU
    torch.save(feats, out_path)
    return video_path, out_path


def main():
    parser = argparse.ArgumentParser(description="Preprocessed-video -> feature (.pth) extractor")
    parser.add_argument("--base-path", type=str, default=DEFAULT_BASE_PATH)
    parser.add_argument("--all", action="store_true", help="Process all data_type/sessions under base-path")
    parser.add_argument("--data-type", type=str, default=None, choices=["voiced", "silent"], help="Optional filter")
    parser.add_argument("--sess", type=str, default=None, help="Optional session filter, e.g. 1-1")
    parser.add_argument("--num", type=int, default=None, help="Optional integer sample id, e.g. 0 for 0000")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only print how many files would be processed")
    args = parser.parse_args()

    if not args.all:
        if args.data_type is None or args.sess is None:
            raise SystemExit("Error: provide --all OR both --data-type and --sess")

    videos = _video_preprocessed_glob_all(args.base_path, args.data_type, args.sess, args.num)
    if len(videos) == 0:
        raise SystemExit(
            f"No videos found under {args.base_path}/{{data_type}}/{{sess}}/data/video_preprocessed "
            f"(data_type={args.data_type}, sess={args.sess}, num={args.num})"
        )

    extractor = AVHuBERTFeatureExtractor(
        AVHuBERTExtractorConfig(
            ckpt_path="/data2/spjune/v2sflow/large_vox.pt",
            output_layer=None,
            device=args.device,
        )
    )

    if args.dry_run:
        print(f"[Dry-run] Found {len(videos)} videos.")
        print("Example:")
        for p in videos[:5]:
            print(f"  - {p}")
        print("Done.")
        return

    for vp in tqdm(videos, desc="Extracting video features"):
        in_path, out_path = extract_and_save_one(vp, extractor, overwrite=args.overwrite)
        # lightweight progress log
        if args.num is not None:
            print(f"Saved: {out_path}  (from {os.path.basename(in_path)})")

    print("Done.")


if __name__ == "__main__":
    main()

