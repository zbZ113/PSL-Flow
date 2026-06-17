#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

DATASET_NAME="${DATASET_NAME:-UNKNOWN}"
BASE_CFG="${BASE_CFG:-configs/train/ldm/klvae_all_256_3rd.yml}"
VAE_ROOT="${VAE_ROOT:-checkpoints/klvae_3rd}"
RUN_ROOT="${RUN_ROOT:-logs/klvae/3rd/${DATASET_NAME}}"
TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-${DATASET_NAME}}"
VAL_DATASET_NAME="${VAL_DATASET_NAME:-${DATASET_NAME}}"
TEST_DATASET_NAME="${TEST_DATASET_NAME:-${DATASET_NAME}}"
TARGET_VAL_DATASET="${TARGET_VAL_DATASET:-${VAL_DATASET_NAME}}"
USE_FULL_DATASET_EPOCH="${USE_FULL_DATASET_EPOCH:-1}"
TRAIN_LOAD_CKPT="${TRAIN_LOAD_CKPT:-checkpoints/klvae/checkpoints/klvae_122_FID[7.8546]_LPIPS[0.0581].ckpt}"
TRAIN_LOAD_TYPE="${TRAIN_LOAD_TYPE:-finetune}"

if [[ "${DATASET_NAME}" == "UNKNOWN" ]]; then
  echo "[ERR] DATASET_NAME is empty or unknown."
  exit 1
fi

if [[ ! -f "${TRAIN_LOAD_CKPT}" ]]; then
  echo "[ERR] Stage-3 requires stage-1 checkpoint, but missing: ${TRAIN_LOAD_CKPT}"
  exit 1
fi

echo "[KLVAE-3rd] DATASET_NAME=${DATASET_NAME}"
echo "[KLVAE-3rd] BASE_CFG=${BASE_CFG}"
echo "[KLVAE-3rd] VAE_ROOT=${VAE_ROOT}"
echo "[KLVAE-3rd] RUN_ROOT=${RUN_ROOT}"
echo "[KLVAE-3rd] TRAIN_LOAD_CKPT=${TRAIN_LOAD_CKPT}"

bash shell/VAE/train_klvae_dataset.sh
