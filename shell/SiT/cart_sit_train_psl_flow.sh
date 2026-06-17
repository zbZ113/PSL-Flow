#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Dataset / teacher
export SIT_DATASET="${SIT_DATASET:-CART}"
export TERB_NET_CKPT="${TERB_NET_CKPT:-checkpoints/physics/${SIT_DATASET}/terb_net/teacher_best.pth}"

# PSL-VAE selection
# Recommended selection from local metrics: epoch 260, best FID, thermal_normalizer ~ 0.9410622119903564
export SIT_VAE_SELECT="${SIT_VAE_SELECT:-epoch}"
export SIT_VAE_EPOCH="${SIT_VAE_EPOCH:-260}"
export SIT_VAE_RUN_DIR="${SIT_VAE_RUN_DIR:-logs/psl_vae/CART_lpips01_ft}"
export SIT_VAE_METRICS_CSV="${SIT_VAE_METRICS_CSV:-logs/psl_vae/cart_metrics.csv}"
export SIT_VAE_CKPT_DIR="${SIT_VAE_CKPT_DIR:-checkpoints/psl_vae_lpips01/CART/checkpoints}"

# Output / resume
export SIT_RUN_ROOT="${SIT_RUN_ROOT:-logs/psl_flow/CART_psl_vae_ep260}"
export SIT_RESUME_CKPT="${SIT_RESUME_CKPT:-auto}"

# Common training parameters (copied from current default config for readability)
export NUM_EPOCHS="${NUM_EPOCHS:-300}"
export USE_FULL_DATASET_EPOCH="${USE_FULL_DATASET_EPOCH:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-16}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
export GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-1}"
export MIXED_PRECISION="${MIXED_PRECISION:-false}"
export TRAIN_IMAGE_SIZE="${TRAIN_IMAGE_SIZE:-256,256}"
export VAL_FREQ="${VAL_FREQ:-50}"

# Trainer / runtime
export PL_DEVICES="${PL_DEVICES:-1}"
export PL_NUM_NODES="${PL_NUM_NODES:-1}"
export PL_ACCELERATOR="${PL_ACCELERATOR:-gpu}"
export PL_STRATEGY="${PL_STRATEGY:-auto}"
export DISABLE_WANDB="${DISABLE_WANDB:-0}"

bash shell/SiT/train_psl_flow_dataset.sh
