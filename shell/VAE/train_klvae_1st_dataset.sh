#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

DATASET_NAME="${DATASET_NAME:-UNKNOWN}"
BASE_CFG="${BASE_CFG:-configs/train/ldm/klvae_all_256_1st.yml}"
VAE_ROOT="${VAE_ROOT:-checkpoints/klvae_1st}"
RUN_ROOT="${RUN_ROOT:-logs/klvae/1st/${DATASET_NAME}}"
TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-${DATASET_NAME}}"
VAL_DATASET_NAME="${VAL_DATASET_NAME:-${DATASET_NAME}}"
TEST_DATASET_NAME="${TEST_DATASET_NAME:-${DATASET_NAME}}"
TARGET_VAL_DATASET="${TARGET_VAL_DATASET:-${VAL_DATASET_NAME}}"
USE_FULL_DATASET_EPOCH="${USE_FULL_DATASET_EPOCH:-1}"

if [[ "${DATASET_NAME}" == "UNKNOWN" ]]; then
  echo "[ERR] DATASET_NAME is empty or unknown."
  exit 1
fi

echo "[KLVAE-1st] DATASET_NAME=${DATASET_NAME}"
echo "[KLVAE-1st] BASE_CFG=${BASE_CFG}"
echo "[KLVAE-1st] VAE_ROOT=${VAE_ROOT}"
echo "[KLVAE-1st] RUN_ROOT=${RUN_ROOT}"

bash shell/VAE/train_klvae_dataset.sh
