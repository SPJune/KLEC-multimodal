#!/bin/bash
set -euo pipefail

usage () {
  cat <<'EOF'
Usage:
  bash infer/emg2wer.sh <exp_name> <ckpt_epoch> <split> <gpu> [--modality emg|video]

Notes:
  - Default modality is "emg" (runs preprocess/emg2feat.py).
  - If modality is "video", we still run preprocess/emg2feat.py but save the model's
    video-branch outputs (feat_vid/ph_vid) into *_feat.npy, *_ph.npy.
  - This requires a checkpoint trained with emg_enc.modality=both and emg_enc.fusion_method=ab.
EOF
}

if [[ $# -lt 4 ]]; then
  usage
  exit 1
fi

EXP_NAME="$1"
CKPT_EPOCH="$2"
SPLIT="$3"
GPU="$4"
shift 4

MODALITY="emg"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --modality)
      MODALITY="${2:-}"
      shift 2
      ;;
    modality=*)
      MODALITY="${1#modality=}"
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERR] Unknown argument: $1" 1>&2
      usage
      exit 2
      ;;
  esac
done

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${MODALITY}" == "emg" ]]; then
  (
    cd "${BASE_DIR}/preprocess"
    CUDA_VISIBLE_DEVICES="${GPU}" \
      python emg2feat.py "exp_name=${EXP_NAME}" "ckpt_epoch=${CKPT_EPOCH}" "split=${SPLIT}" "modality=emg"
  )
elif [[ "${MODALITY}" == "video" ]]; then
  (
    cd "${BASE_DIR}/preprocess"
    CUDA_VISIBLE_DEVICES="${GPU}" \
      python emg2feat.py "exp_name=${EXP_NAME}" "ckpt_epoch=${CKPT_EPOCH}" "split=${SPLIT}" "modality=video"
  )
else
  echo "[ERR] Invalid modality: ${MODALITY} (expected emg|video)" 1>&2
  exit 2
fi

(
  cd "${BASE_DIR}/infer"
  CUDA_VISIBLE_DEVICES="${GPU}" \
    python save_output_sr16.py "encoder=${EXP_NAME}" "data_split=${SPLIT}"
)

(
  cd "${BASE_DIR}/infer"
  CUDA_VISIBLE_DEVICES="${GPU}" \
    python asr.py "wav_dir=${EXP_NAME}/direct/${SPLIT}" "data_split=${SPLIT}"
)

# Optional:
# (
#   cd "${BASE_DIR}/infer"
#   CUDA_VISIBLE_DEVICES="${GPU}" \
#     python phoneme_per_pred.py "encoder=${EXP_NAME}" "data_split=${SPLIT}"
# )
