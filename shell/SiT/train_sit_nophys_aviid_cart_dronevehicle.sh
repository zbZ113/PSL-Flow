#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SIT_VAE_CKPT="${SIT_VAE_CKPT:-checkpoints/klvae_shared_ft60_aviid_cart_dronevehicle_resume/checkpoints/last.ckpt}"
DISABLE_WANDB="${DISABLE_WANDB:-1}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-10}"
SIT_DATASETS_FOLDER="${SIT_DATASETS_FOLDER:-}"
USE_FULL_DATASET_EPOCH="${USE_FULL_DATASET_EPOCH:-1}"
PL_DEVICES="${PL_DEVICES:-}"
PL_NUM_NODES="${PL_NUM_NODES:-1}"
PL_ACCELERATOR="${PL_ACCELERATOR:-gpu}"
PL_STRATEGY="${PL_STRATEGY:-auto}"
SIT_RGB_VAE_CKPT="${SIT_RGB_VAE_CKPT:-}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
SIT_RESUME_CKPT="${SIT_RESUME_CKPT:-auto}"
SIT_RUN_ROOT_BASE="${SIT_RUN_ROOT_BASE:-logs/sit_noall}"

if [[ ! -f "${SIT_VAE_CKPT}" ]]; then
  echo "[ERR] SIT_VAE_CKPT not found: ${SIT_VAE_CKPT}"
  exit 1
fi

run_single() {
  local dataset_name="$1"
  echo "[Generic-SiT-4x] dataset=${dataset_name}"
  export SIT_DATASET="${dataset_name}"
  export SIT_RUN_ROOT="${SIT_RUN_ROOT_BASE}/${dataset_name}"
  export SIT_VAE_CKPT
  export DISABLE_WANDB
  export CHECK_VAL_EVERY_N_EPOCH
  export SIT_DATASETS_FOLDER
  export USE_FULL_DATASET_EPOCH
  export PL_DEVICES
  export PL_NUM_NODES
  export PL_ACCELERATOR
  export PL_STRATEGY
  export SIT_RGB_VAE_CKPT
  export LIMIT_TRAIN_BATCHES
  export LIMIT_VAL_BATCHES
  export SIT_RESUME_CKPT
  bash shell/SiT/train_generic_sit_dataset.sh
}

run_single "AVIID"
run_single "CART"
run_single "DroneVehicle_day"
run_single "DroneVehicle_night"
