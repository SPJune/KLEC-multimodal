# KLEC Multimodal Silent Speech

This repository trains a multimodal silent-speech model that predicts target speech features and phoneme sequences from EMG and optional video features.

## Current Active Pipeline

The current workflow used in this codebase is:

1. Data split: `preprocess/split.py`
2. Phoneme/TextGrid preparation: `preprocess/mfa.py`, `textgrid_maker.py`
3. Feature preprocessing:
   - `preprocess/speech2feat.py`
   - `preprocess/prepare_video.py`
   - `preprocess/pp_video2feat.py`
4. Training: `main.py`
5. CER evaluation after training: `infer/emg2wer.sh`

## Environment

- Python dependencies: `requirements.txt`
- Config root: `configs/`
- Common paths in `configs/common.yaml`
  - `data_path` (default: `/data/path`)
  - `exp_path` (default: `/exp/path`)
  - `paths.*` (dataset/checkpoint absolute paths)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Expected Data Layout

The code expects data under:

- `/data/path/silent_speech_dataset/...`

Key generated artifacts:

- `preprocess/emg_split.csv`
- `.../preprocessed/target_feature/<target>/<sub_option>/...`
- `.../silent_speech_dataset/<type>/<sess>/data/video_preprocessed/*.mp4`
- `.../silent_speech_dataset/<type>/<sess>/data/video_features/*.pth`
- `.../silent_speech_dataset/phoneme_set.json`

## Step-by-Step Commands

### 1) Build train/valid/test split

```bash
python preprocess/split.py
```

This creates `preprocess/emg_split.csv`.

### 2) Prepare TextGrid/phoneme resources

MFA-based alignment:

```bash
python preprocess/mfa.py --sess 1-1 --all
```

Alternative/fallback TextGrid generation:

```bash
python textgrid_maker.py
```

### 3) Precompute speech target features

```bash
python preprocess/speech2feat.py
```

Related config: `configs/speech2feat.yaml` and `configs/feature/*.yaml`.

Current `speech2feat.py` supports:

- `mspec` (mel-spectrogram)
- `contentvec`

### 4) Preprocess videos (30 fps -> 25 fps, mouth crop)

Single file:

```bash
python preprocess/prepare_video.py --input <input_mp4> --output <output_mp4>
```

Dataset mode:

```bash
python preprocess/prepare_video.py --base-path <paths.silent_speech_dataset>
```

### 5) Extract AV-HuBERT features from preprocessed videos

Dry run first:

```bash
python preprocess/pp_video2feat.py --all --device cuda --dry-run
```

Actual extraction:

```bash
python preprocess/pp_video2feat.py --all --device cuda
```

### 6) Train

```bash
python main.py exp_name=<EXP_NAME>
```

Useful overrides:

```bash
python main.py exp_name=<EXP_NAME> emg_enc.modality=both emg_enc.fusion_method=ab
python main.py exp_name=<EXP_NAME> feature=mspec16
python main.py exp_name=260301ch4_both_num200_after4 emg_enc.use_channel='[0,2,4,5]' emg_enc.modality=both emg_enc.fusion_method=ab \
emg_enc.video_encoder_mode=finetune \
emg_enc.ab_num_tokens=200 emg_enc.fusion_after_layer=4
```

### 7) Evaluate CER after training

```bash
bash infer/emg2wer.sh <EXP_NAME> <CKPT_EPOCH|best|last> <train|valid|test> <GPU_ID>
```

Examples:

```bash
bash infer/emg2wer.sh my_exp best test 3
bash infer/emg2wer.sh my_exp last valid 3 --modality video
```

`infer/emg2wer.sh` runs:

1. `preprocess/emg2feat.py`
2. `infer/save_output_sr16.py`
3. `infer/asr.py`

## Notes

- Main training entrypoint is Hydra-based (`configs/config.yaml`).
- The dataloader reads split metadata from `preprocess/emg_split.csv`.
- Validation/test filtering is implemented in `loader.py`.
- Checkpoints are saved under `<exp_path>/<exp_name>/`.
- Preprocessing scripts write files to dataset directories, so run dry-run options first when available.
