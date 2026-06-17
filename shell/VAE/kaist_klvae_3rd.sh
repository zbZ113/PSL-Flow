#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export DATASET_NAME="${DATASET_NAME:-KAIST}"
export RUN_ROOT="${RUN_ROOT:-logs/klvae/3rd/${DATASET_NAME}}"
export VAE_ROOT="${VAE_ROOT:-checkpoints/klvae_3rd}"
export TRAIN_LOAD_CKPT="${TRAIN_LOAD_CKPT:-checkpoints/klvae/checkpoints/klvae_122_FID[7.8546]_LPIPS[0.0581].ckpt}"

bash shell/VAE/train_klvae_3rd_dataset.sh
