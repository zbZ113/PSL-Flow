#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export RUN_ROOT="${RUN_ROOT:-logs/klvae/shared_ft20_cart_dronevehicle}"
export VAE_ROOT="${VAE_ROOT:-checkpoints/klvae_shared_ft20_cart_dronevehicle}"
export TRAIN_DATASETS="${TRAIN_DATASETS:-AVIID CART DroneVehicle_day DroneVehicle_night}"
export VAL_DATASETS="${VAL_DATASETS:-${TRAIN_DATASETS}}"
export TEST_DATASETS="${TEST_DATASETS:-${TRAIN_DATASETS}}"
export TARGET_VAL_DATASET="${TARGET_VAL_DATASET:-AVIID}"
export TRAIN_LOAD_CKPT="${TRAIN_LOAD_CKPT:-logs/klvae/shared/checkpoints/last.ckpt}"
export TRAIN_LOAD_TYPE="${TRAIN_LOAD_TYPE:-finetune}"
export NUM_EPOCHS="${NUM_EPOCHS:-20}"
export CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
export USE_FULL_DATASET_EPOCH="${USE_FULL_DATASET_EPOCH:-1}"
export DISABLE_WANDB="${DISABLE_WANDB:-1}"
export PL_DEVICES="${PL_DEVICES:-1}"
export PL_STRATEGY="${PL_STRATEGY:-auto}"

# If caller wants true continuation training, prefer resume-from and disable finetune init.
if [[ -n "${KLVAE_RESUME_CKPT:-}" ]]; then
  export TRAIN_LOAD_CKPT=""
  export TRAIN_LOAD_TYPE=""
fi

bash shell/VAE/train_klvae_shared.sh
