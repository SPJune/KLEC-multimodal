import re
import os
import numpy as np
import scipy
import json
import pickle
import pandas as pd
from glob import glob
from fractions import Fraction
from typing import Dict, Tuple, Optional
import torch
import cv2

from data_utils import FeatureNormalizer, phoneme_inventory, read_phonemes, TextTransform

def remove_drift(signal, fs): # for emg signal
    b, a = scipy.signal.butter(3, 2, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)

def notch(signal, freq, sample_frequency):
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)

def notch_harmonics(signal, freq, sample_frequency):
    for harmonic in range(1,3):
        signal = notch(signal, freq*harmonic, sample_frequency)
    return signal

def subsample(signal, new_freq, old_freq):
    """
    Resample a 1D signal from old_freq -> new_freq.

    Notes:
    - Uses polyphase resampling (anti-aliasing) to behave well for downsampling.
    - Output length is forced to round(N * new_freq / old_freq) for stable alignment.
    """
    old_f = float(old_freq)
    new_f = float(new_freq)
    if old_f <= 0 or new_f <= 0:
        raise ValueError(f"Frequencies must be positive (old={old_freq}, new={new_freq})")
    if old_f == new_f:
        return np.asarray(signal)

    # Approximate ratio with a rational for resample_poly(up, down)
    ratio = Fraction(new_f / old_f).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator

    y = scipy.signal.resample_poly(np.asarray(signal), up, down)

    # Force deterministic length for downstream alignment (e.g., //conv_ds with speech frames)
    target_len = int(round(len(signal) * (new_f / old_f)))
    if target_len < 0:
        target_len = 0
    if y.shape[0] > target_len:
        y = y[:target_len]
    elif y.shape[0] < target_len:
        pad_val = float(y[-1]) if y.shape[0] > 0 else 0.0
        y = np.pad(y, (0, target_len - y.shape[0]), mode="constant", constant_values=pad_val)
    return y

def apply_to_all(function, signal_array, *args, **kwargs):
    results = []
    for i in range(signal_array.shape[1]):
        results.append(function(signal_array[:,i], *args, **kwargs))
    return np.stack(results, 1)


class EMGDataset(torch.utils.data.Dataset):
    DEFAULT_EDGE_TRIM_SEC = 0.2

    def __init__(
        self,
        base_dir,
        target,
        data_split,
        frame_rate,
        target_sec=None,
        normalize=False,
        seq_len=None,
        edge_trim_sec: float = DEFAULT_EDGE_TRIM_SEC,
    ):
        self.load_zero_output(base_dir)
        if normalize:
            npz_data = np.load(os.path.join(base_dir, 'normalizer.npz'))
            self.normalizer = FeatureNormalizer(npz_data['mean'], npz_data['std'])
            self.feat_norm, self.emg_norm = pickle.load(open(os.path.join(base_dir, 'normalizer.pkl'),'rb'))
        self.data_split = data_split
        self.preprocessed_path = base_dir
        base_dir = base_dir[:base_dir.find('preprocessed')].rstrip('/')
        self.base_path = os.path.join(base_dir, 'silent_speech_dataset')

        df = pd.read_csv('preprocess/emg_split.csv')
        df = df[df["split"] == data_split]
        if data_split in ["valid", "test"]:
            df = df[df["data_type"] == "silent"]

        paths = df["path"].apply(lambda p: os.path.join(base_dir, p)).tolist()
        sessions = df["session"].tolist()
        data_types = df["data_type"].tolist()
        data_nums = df["data_num"].tolist()

        self.paths = paths
        self.sessions = sessions
        self.data_types = data_types
        self.data_nums = data_nums

        # ------------------------------------------------------------
        # Path indices (avoid per-sample glob)
        # ------------------------------------------------------------
        self._feat_path_index: Dict[Tuple[str, int], str] = {}
        self._voiced_emg_path_index: Dict[Tuple[str, int], str] = {}
        self._video_feat_path_index: Dict[Tuple[str, str, int], str] = {}
        self._build_path_indices()

        self.frame_rate = frame_rate
        # Resample EMG to (frame_rate * 4) Hz, then downsample twice (/4) in the model to match frame_rate.
        self.conv_ds = 4
        # Video features are typically extracted at 25 fps.
        self.video_fps = 25.0
        self.text_transform = TextTransform()
        self.target_sec = target_sec
        self.normalize = normalize
        self.seq_len = seq_len
        self.edge_trim_sec = float(edge_trim_sec)
        if self.target_sec != None:
            self.target_speech_len = int(self.frame_rate*self.target_sec)
            self.target_emg_len = self.target_speech_len * self.conv_ds

        self.target = target

    @staticmethod
    def _trim_time_series(x: torch.Tensor, trim_left: int, trim_right: Optional[int] = None) -> torch.Tensor:
        """
        Trim both ends of a time-series tensor.
        - x: (T, ...) or (T,)
        - trim_left/right: number of frames to trim
        If the sequence is too short, returns an empty tensor (T=0) with the remaining shape preserved.
        """
        if trim_right is None:
            trim_right = trim_left
        trim_left = int(max(0, trim_left))
        trim_right = int(max(0, trim_right))
        if trim_left == 0 and trim_right == 0:
            return x
        t = int(x.shape[0])
        if t <= trim_left + trim_right:
            return x[:0]
        end = t - trim_right if trim_right > 0 else t
        return x[trim_left:end]

    @staticmethod
    def _center_trim_to_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Center-trim a time-series tensor to target_len frames.
        - Trimming is symmetric; if odd, trim one extra frame on the right.
        - If x is already <= target_len, returns x as-is.
        """
        target_len = int(max(0, target_len))
        t = int(x.shape[0])
        if t <= target_len:
            return x
        diff = t - target_len
        trim_left = diff // 2
        trim_right = diff - trim_left  # if odd, trim one more on the right
        return EMGDataset._trim_time_series(x, trim_left, trim_right)

    def _build_path_indices(self):
        """
        Scan directories once at init time to build path indices and avoid per-sample glob calls.
        """
        sess_set = sorted(set(self.sessions))

        # ---- speech feature (.npy) under preprocessed_path/voiced/{sess}/data/audio ----
        for sess in sess_set:
            audio_dir = os.path.join(self.preprocessed_path, "voiced", sess, "data", "audio")
            if not os.path.isdir(audio_dir):
                continue
            for p in glob(os.path.join(audio_dir, "*.npy")):
                base = os.path.basename(p)
                m = re.search(r"_(\d+)\.", base)
                if not m:
                    continue
                num = int(m.group(1))
                self._feat_path_index.setdefault((sess, num), p)

        # ---- voiced emg (.npz) under base_path/voiced/{sess}/data/emg ----
        for sess in sess_set:
            emg_dir = os.path.join(self.base_path, "voiced", sess, "data", "emg")
            if not os.path.isdir(emg_dir):
                continue
            for p in glob(os.path.join(emg_dir, "*.npz")):
                base = os.path.basename(p)
                m = re.search(r"_(\d+)\.", base)
                if not m:
                    continue
                num = int(m.group(1))
                self._voiced_emg_path_index.setdefault((sess, num), p)

        # ---- video feature (.pth) under base_path/{data_type}/{sess}/data/video_features ----
        dt_set = sorted(set(self.data_types))
        for dt in dt_set:
            for sess in sess_set:
                vf_dir = os.path.join(self.base_path, dt, sess, "data", "video_features")
                if not os.path.isdir(vf_dir):
                    continue
                for p in glob(os.path.join(vf_dir, "*.pth")):
                    base = os.path.basename(p)
                    m = re.search(r"_(\d+)\.", base)
                    if not m:
                        continue
                    num = int(m.group(1))
                    self._video_feat_path_index.setdefault((dt, sess, num), p)

    def load_emg(self, path, max_len=None):
        resample_rate = self.frame_rate*self.conv_ds
        raw_emg = np.load(path)['data']
    
        x = raw_emg
        x = apply_to_all(notch_harmonics, x, 60, 250)
        x = apply_to_all(remove_drift, x, 250)
        raw_emg = apply_to_all(subsample, x, resample_rate, 250)
        raw_emg = raw_emg / 20
        raw_emg = 50*np.tanh(raw_emg/50.)
        raw_emg = raw_emg.astype(np.float32)
        return raw_emg

    def load_textgrid(self, path, max_len=None):
        if os.path.exists(path):
            phonemes = read_phonemes(path, max_len, fr=self.frame_rate)
        else:
            phonemes = np.zeros(max_len, dtype=np.int64)+phoneme_inventory.index('sil')
        return phonemes
    
    def load_text(self, path):
        with open(path) as f:
            info = json.load(f)
        return info['text']

    def load_speech_feature(self, path, max_len=None):
        feat = np.load(path)
        feat = feat.astype(np.float32)
        if max_len is not None and feat.shape[0] > max_len:
            feat = feat[:max_len, :]
        return feat

    def find_video_path(self, data_type, sess, num, preprocessed=True):
        """Find video path matching the pattern.
        
        Args:
            data_type: 'voiced' or 'silent'
            sess: session name (e.g., '3-4')
            num: data number
            preprocessed: if True, look in video_preprocessed folder
        
        Returns:
            video file path or None if not found
        """
        if preprocessed:
            video_dir = 'video_preprocessed'
        else:
            video_dir = 'video'
        
        pattern = os.path.join(self.base_path, data_type, sess, f'data/{video_dir}/*_{num:04d}.*.mp4')
        matches = glob(pattern)
        
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            print(f"Warning: Multiple videos found for {data_type}/{sess}/{num}: {matches}")
            return matches[0]
        else:
            return None

    def find_video_feature_path(self, data_type, sess, num):
        """Find video feature path matching the pattern.

        Feature files are expected to be created from preprocessed videos and saved under:
          {base_path}/{data_type}/{sess}/data/video_features/*.pth
        """
        return self._video_feat_path_index.get((data_type, sess, int(num)))

    def load_video_feature(self, path):
        """Load saved video feature (.pth).

        Expected shape: (T, C) torch.Tensor, where C is usually 1024 (AV-HuBERT Large).
        """
        if path is None or not os.path.exists(path):
            return None
        feat = torch.load(path, map_location="cpu")
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat)
        if not isinstance(feat, torch.Tensor):
            raise TypeError(f"Unsupported feature type at {path}: {type(feat)}")
        return feat.to(dtype=torch.float32)

    def load_video(self, path, normalize=True):
        """Load preprocessed video frames from V2SFlow-style preprocessing.
        
        Args:
            path: path to preprocessed video file (96x96 grayscale mouth crop, 25fps)
            normalize: if True, normalize pixel values to [0, 1]
        
        Returns:
            video frames as numpy array (T, H, W) or (T, C, H, W)
        """
        if path is None or not os.path.exists(path):
            return None
        
        cap = cv2.VideoCapture(path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to grayscale if needed
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            return None
        
        video = np.array(frames, dtype=np.float32)  # (T, H, W)
        
        if normalize:
            video = video / 255.0
        
        return video

    def __len__(self):
        return len(self.paths)

    def load_zero_output(self, base_dir):
        path = os.path.join(base_dir, 'zero_output.npy')
        self.zero_output = np.load(path)

    def extract_path(self, sess, num):
        feat_path = self._feat_path_index.get((sess, int(num)))
        if feat_path is None:
            raise FileNotFoundError(
                f"speech feature not found for sess={sess}, num={num} under {self.preprocessed_path}"
            )
        tg_path = os.path.join(self.base_path, "voiced", sess, 'data/textgrid', f'tg_{num:04d}.TextGrid')
        voiced_path = self._voiced_emg_path_index.get((sess, int(num)))
        if voiced_path is None:
            raise FileNotFoundError(
                f"voiced emg not found for sess={sess}, num={num} under {self.base_path}/voiced/{sess}/data/emg"
            )
        return feat_path, tg_path, voiced_path

    def __getitem__(self, i):
        session = self.sessions[i]
        data_type = self.data_types[i]
        data_num = self.data_nums[i]
        emg_path = self.paths[i]
        silent = data_type == 'silent'
        feat_path, tg_path, voiced_path = self.extract_path(session, data_num)

        video_feat_path = self.find_video_feature_path(data_type, session, data_num)

        speech_feature = self.load_speech_feature(feat_path)
        emg_max_len = None
        emg = self.load_emg(emg_path, max_len=emg_max_len)
        voiced_emg = self.load_emg(voiced_path, max_len=emg_max_len) if silent else None

        phonemes = self.load_textgrid(tg_path, max_len=speech_feature.shape[0])

        if self.normalize:
            speech_feature = self.feat_norm.normalize(speech_feature)

        speech_feature = torch.from_numpy(speech_feature)
        emg = torch.from_numpy(emg)
        phonemes = torch.from_numpy(phonemes)
        speech_feature = speech_feature.type(torch.float32)
        emg = emg.type(torch.float32)
        if silent:
            voiced_emg = torch.from_numpy(voiced_emg)
            voiced_emg = voiced_emg.type(torch.float32)

        video_feature = self.load_video_feature(video_feat_path)
        if video_feature is None:
            # Keep return type stable; downstream can decide how to handle missing features.
            video_feature = torch.zeros((0, 1024), dtype=torch.float32)

        # ------------------------------------------------------------
        # Align modality lengths (before edge trimming).
        # - silent=False: align speech/phoneme/emg/video to the shortest duration.
        # - silent=True : align (emg, video) and align (voiced_emg, speech_feature, phoneme) separately.
        # Length comparison is done in speech-frame units; trimming is center-based.
        # ------------------------------------------------------------
        fr = float(self.frame_rate)
        conv_ds = int(self.conv_ds)
        video_fps = float(getattr(self, "video_fps", 25.0))

        video_len = int(video_feature.shape[0])
        has_video = (video_len > 0 and fr > 0 and video_fps > 0)
        # If speech/video ratio is an integer k, floor target length to a multiple of k
        # so video (e.g., 25 Hz) and speech (e.g., 100 Hz) align exactly.
        k_speech_per_video: Optional[int] = None
        if has_video:
            ratio = fr / video_fps
            nearest = int(round(ratio))
            if nearest >= 1 and abs(ratio - float(nearest)) < 1e-6:
                k_speech_per_video = nearest

        if not silent:
            speech_len = int(speech_feature.shape[0])
            phoneme_len = int(phonemes.shape[0])
            emg_eq_len = int(emg.shape[0] // conv_ds) if conv_ds > 0 else 0  # in speech-frame units

            lengths_eq = [speech_len, phoneme_len, emg_eq_len]
            if has_video:
                if k_speech_per_video is not None:
                    video_eq_len = int(video_len * k_speech_per_video)
                else:
                    video_eq_len = int(np.floor(video_len * fr / video_fps))
                lengths_eq.append(video_eq_len)

            min_eq_len_raw = int(max(0, min(lengths_eq))) if len(lengths_eq) > 0 else 0
            # If video is present and the speech/video ratio is an integer k, floor the target
            # length to a multiple of k so it is exactly representable in video frames too.
            if has_video and k_speech_per_video is not None and k_speech_per_video > 1:
                min_eq_len = (min_eq_len_raw // k_speech_per_video) * k_speech_per_video
            else:
                min_eq_len = min_eq_len_raw

            target_speech_len = min_eq_len
            target_phoneme_len = min_eq_len
            target_emg_len = min_eq_len * conv_ds
            if has_video:
                if k_speech_per_video is not None:
                    target_video_len = int(min_eq_len // k_speech_per_video)
                else:
                    target_video_len = int(np.floor(min_eq_len * video_fps / fr))
            else:
                target_video_len = video_len

            speech_feature = self._center_trim_to_length(speech_feature, target_speech_len)
            phonemes = self._center_trim_to_length(phonemes, target_phoneme_len)
            emg = self._center_trim_to_length(emg, target_emg_len)
            video_feature = self._center_trim_to_length(video_feature, target_video_len)

        else:
            # ---- group A: (silent) emg ↔ video ----
            emg_eq_len = int(emg.shape[0] // conv_ds) if conv_ds > 0 else 0
            lengths_a = [emg_eq_len]
            if has_video:
                if k_speech_per_video is not None:
                    video_eq_len = int(video_len * k_speech_per_video)
                else:
                    video_eq_len = int(np.floor(video_len * fr / video_fps))
                lengths_a.append(video_eq_len)
            min_eq_a_raw = int(max(0, min(lengths_a))) if len(lengths_a) > 0 else 0
            if has_video and k_speech_per_video is not None and k_speech_per_video > 1:
                min_eq_a = (min_eq_a_raw // k_speech_per_video) * k_speech_per_video
            else:
                min_eq_a = min_eq_a_raw

            target_emg_len_a = min_eq_a * conv_ds
            if has_video:
                if k_speech_per_video is not None:
                    target_video_len_a = int(min_eq_a // k_speech_per_video)
                else:
                    target_video_len_a = int(np.floor(min_eq_a * video_fps / fr))
            else:
                target_video_len_a = video_len

            emg = self._center_trim_to_length(emg, target_emg_len_a)
            video_feature = self._center_trim_to_length(video_feature, target_video_len_a)

            # ---- group B: voiced_emg ↔ speech_feature ↔ phoneme ----
            speech_len = int(speech_feature.shape[0])
            phoneme_len = int(phonemes.shape[0])
            v_emg_eq_len = int(voiced_emg.shape[0] // conv_ds) if (voiced_emg is not None and conv_ds > 0) else 0

            lengths_b = [speech_len, phoneme_len, v_emg_eq_len]
            min_eq_b = int(max(0, min(lengths_b))) if len(lengths_b) > 0 else 0

            target_speech_len_b = min_eq_b
            target_phoneme_len_b = min_eq_b
            target_voiced_emg_len_b = min_eq_b * conv_ds

            speech_feature = self._center_trim_to_length(speech_feature, target_speech_len_b)
            phonemes = self._center_trim_to_length(phonemes, target_phoneme_len_b)
            if voiced_emg is not None:
                voiced_emg = self._center_trim_to_length(voiced_emg, target_voiced_emg_len_b)

        # ------------------------------------------------------------
        # After alignment, trim all time-series by edge_trim_sec on both ends.
        # ------------------------------------------------------------
        trim_sec = float(self.edge_trim_sec)
        if trim_sec > 0:
            trim_speech = int(round(float(self.frame_rate) * trim_sec))              # speech_feature, phonemes (T @ frame_rate)
            trim_emg = int(round(float(self.frame_rate * self.conv_ds) * trim_sec)) # emg, voiced_emg (T @ frame_rate*conv_ds)
            trim_video = int(round(float(self.video_fps) * trim_sec))               # video_feature (T @ video_fps)

            speech_feature = self._trim_time_series(speech_feature, trim_speech)
            phonemes = self._trim_time_series(phonemes, trim_speech)
            emg = self._trim_time_series(emg, trim_emg)
            if silent and voiced_emg is not None:
                voiced_emg = self._trim_time_series(voiced_emg, trim_emg)
            video_feature = self._trim_time_series(video_feature, trim_video)

        return speech_feature, emg, phonemes, silent, voiced_emg, video_feature

    def collate_to_max_len(self, batch):
        """
        Pad/collate to the maximum length within the batch.
        - If the longest sample exceeds 8 seconds, truncate all samples to 8 seconds.
        - Length unit is speech-feature frames (T).
          - target_lengths: speech/phoneme length (target)
          - est_lengths   : EMG length (T_emg//conv_ds; model output length base)
        - With silent=True, target and est may differ; keep separate caps for target (speech) and est (EMG).
        """
        # (speech_feature, emg, phonemes, silent, voiced_emg, video_feature)
        batch = list(batch)
        batch.sort(key=lambda x: len(x[0]), reverse=True)

        speech_features = [item[0] for item in batch]  # (T, D)
        emg_signals = [item[1] for item in batch]      # (T_emg, C)
        phonemes = [item[2] for item in batch]         # (T,)
        silents = [bool(item[3]) for item in batch]    # bool
        voiced_emgs = [item[4] for item in batch]      # (T_emg, C) or None
        video_features = [item[5] for item in batch]   # (T_vid, 1024)

        # length (in speech-frame units)
        target_lengths_raw = [len(sf) for sf in speech_features]
        est_lengths_raw = [len(emg) // self.conv_ds for emg in emg_signals]

        # batch max length (cap at 8 seconds)
        max_len_cap = max(1, int(round(float(self.frame_rate) * 8.0)))
        # target (speech) max
        max_target_length = min(max(target_lengths_raw), max_len_cap)
        # est (emg) max (speech-frame-equivalent)
        max_est_length = min(max(est_lengths_raw), max_len_cap)
        max_emg_len = int(max_est_length * self.conv_ds)

        sil_id = phoneme_inventory.index('sil')
        pad_vec = torch.tensor(self.zero_output, dtype=torch.float32)  # (D,)

        padded_speech = []
        padded_emg = []
        padded_phonemes = []
        padded_voiced_emg = []

        target_lengths = []
        est_lengths = []

        for sf, emg, ph, silent, v_emg in zip(speech_features, emg_signals, phonemes, silents, voiced_emgs):
            t_len = min(len(sf), max_target_length)
            e_len = min(len(emg) // self.conv_ds, max_est_length)
            target_lengths.append(t_len)
            est_lengths.append(e_len)

            # speech feature pad with dataset-specific "zero output" vector
            sf = sf[:max_target_length]
            if sf.shape[0] < max_target_length:
                pad = pad_vec.expand(max_target_length - sf.shape[0], -1).clone()
                sf = torch.cat([sf, pad], dim=0)
            padded_speech.append(sf)

            # phoneme pad with sil
            ph = ph[:max_target_length]
            if ph.dtype != torch.int64:
                ph = ph.to(dtype=torch.int64)
            if ph.shape[0] < max_target_length:
                pad = torch.full((max_target_length - ph.shape[0],), sil_id, dtype=torch.int64)
                ph = torch.cat([ph, pad], dim=0)
            padded_phonemes.append(ph)

            # emg pad with zeros
            emg = emg[:max_emg_len]
            if emg.shape[0] < max_emg_len:
                pad = torch.zeros((max_emg_len - emg.shape[0], emg.shape[1]), dtype=emg.dtype)
                emg = torch.cat([emg, pad], dim=0)
            padded_emg.append(emg)

            # voiced_emg exists only for silent samples; otherwise keep empty.
            if v_emg is None:
                v_emg = torch.zeros((0, emg.shape[1]), dtype=emg.dtype)
            v_emg = v_emg[:max_emg_len]
            if v_emg.shape[0] < max_emg_len:
                pad = torch.zeros((max_emg_len - v_emg.shape[0], emg.shape[1]), dtype=emg.dtype)
                v_emg = torch.cat([v_emg, pad], dim=0)
            padded_voiced_emg.append(v_emg)

        # video feature collate (25Hz cap @ 8 sec)
        video_fps = float(getattr(self, "video_fps", 25.0))
        max_vid_cap = max(1, int(round(video_fps * 8.0)))
        vid_lengths_raw = [int(v.shape[0]) for v in video_features]
        # Limit video max_len (25 Hz) based on EMG-derived max_est_length (in speech-frame units).
        # If frame_rate/video_fps is an integer k (e.g., 100/25=4), use ceil(max_est_length/k).
        fr = float(self.frame_rate)
        ratio = fr / video_fps if video_fps > 0 else 0.0
        nearest = int(round(ratio)) if ratio > 0 else 0
        k = nearest if (nearest >= 1 and abs(ratio - float(nearest)) < 1e-6) else None
        if k is not None and k > 0:
            emg_based_vid_len = int((max_est_length + k - 1) // k)
        else:
            emg_based_vid_len = int(np.ceil(max_est_length * video_fps / fr)) if (fr > 0 and video_fps > 0) else 0

        max_vid_raw = max(vid_lengths_raw) if len(vid_lengths_raw) > 0 else 0
        if max_vid_raw == 0:
            max_vid_len = 0
        else:
            max_vid_len = min(max_vid_raw, max_vid_cap, emg_based_vid_len)

        padded_video = []
        video_lengths = []
        for vf in video_features:
            vlen = min(int(vf.shape[0]), max_vid_len)
            video_lengths.append(vlen)
            vf = vf[:max_vid_len]
            if vf.shape[0] < max_vid_len:
                # vf may be (0, 1024) for missing
                c = int(vf.shape[1]) if vf.ndim == 2 and vf.shape[1] > 0 else 1024
                pad = torch.zeros((max_vid_len - vf.shape[0], c), dtype=torch.float32)
                vf = torch.cat([vf.to(dtype=torch.float32), pad], dim=0)
            padded_video.append(vf.to(dtype=torch.float32))

        video_lengths_t = torch.tensor(video_lengths, dtype=torch.long)
        target_lengths_t = torch.tensor(target_lengths, dtype=torch.long)
        # Convert video length to speech-frame units (e.g., 25 Hz -> 100 Hz: x4).
        # For silent=True, do not cap to target length to preserve est_length if it differs.
        if k is not None and k > 0:
            video_est_lengths = video_lengths_t * int(k)
        else:
            video_est_lengths = torch.floor(video_lengths_t.to(dtype=torch.float32) * float(fr / video_fps)).to(dtype=torch.long) if (fr > 0 and video_fps > 0) else torch.zeros_like(video_lengths_t)
        if max_vid_len > 0:
            # True = padding (V2SFlow convention)
            video_padding_mask = torch.arange(max_vid_len).unsqueeze(0) >= video_lengths_t.unsqueeze(1)
        else:
            video_padding_mask = torch.zeros((len(batch), 0), dtype=torch.bool)

        batch_out = {
            'speech_features': torch.stack(padded_speech),
            'emg': torch.stack(padded_emg),
            'phonemes': torch.stack(padded_phonemes),
            'target_lengths': target_lengths_t,
            'est_lengths': torch.tensor(est_lengths, dtype=torch.long),
            'silents': torch.tensor(silents, dtype=torch.bool),
            'voiced_emg': torch.stack(padded_voiced_emg),
            'video_features': torch.stack(padded_video) if max_vid_len > 0 else torch.zeros((len(batch), 0, 1024), dtype=torch.float32),
            'video_lengths': video_lengths_t,
            'video_padding_mask': video_padding_mask,
            'video_est_lengths': video_est_lengths,
            # Compatibility keys (referenced elsewhere but not used here).
            'mel': torch.stack(padded_speech),
            'f0': torch.zeros((len(batch), 1, max_target_length), dtype=torch.float32),
            'uv': torch.zeros((len(batch), 1, max_target_length), dtype=torch.float32),
        }

        return batch_out
