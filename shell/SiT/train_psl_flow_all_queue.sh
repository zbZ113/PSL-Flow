#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# =========================
# Shared training parameters
# Edit only this file when you want to change the queue behavior.
# These variables are exported to each dataset-specific SiT shortcut.
# =========================

# Dataset sampling
export USE_FULL_DATASET_EPOCH="${USE_FULL_DATASET_EPOCH:-1}"

# Recommended per-dataset local samples/epoch for current queue settings.
# TRAIN_BATCH_SIZE is per GPU/process.
# - approx optimizer steps/epoch = ceil(local_num_samples / TRAIN_BATCH_SIZE)
# - on single-GPU, global seen samples/epoch ~= local_num_samples
# These values keep the total optimization steps in a moderate range across datasets.
# If you later change TRAIN_BATCH_SIZE from 64 to 32, halve these values to keep a similar step budget.
export NUM_SAMPLES_PER_EPOCH_AVIID="${NUM_SAMPLES_PER_EPOCH_AVIID:-2560}"
export NUM_SAMPLES_PER_EPOCH_CART="${NUM_SAMPLES_PER_EPOCH_CART:-2304}"
export NUM_SAMPLES_PER_EPOCH_DRONEVEHICLE_DAY="${NUM_SAMPLES_PER_EPOCH_DRONEVEHICLE_DAY:-4096}"
export NUM_SAMPLES_PER_EPOCH_DRONEVEHICLE_NIGHT="${NUM_SAMPLES_PER_EPOCH_DRONEVEHICLE_NIGHT:-6144}"

# Training hyperparameters
export NUM_EPOCHS="${NUM_EPOCHS:-300}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-16}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
export GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-1}"
export MIXED_PRECISION="${MIXED_PRECISION:-false}"
export TRAIN_IMAGE_SIZE="${TRAIN_IMAGE_SIZE:-256,256}"
export VAL_FREQ="${VAL_FREQ:-20}"

# Trainer / runtime
export PL_DEVICES="${PL_DEVICES:-1}"
export PL_NUM_NODES="${PL_NUM_NODES:-1}"
export PL_ACCELERATOR="${PL_ACCELERATOR:-gpu}"
export PL_STRATEGY="${PL_STRATEGY:-auto}"
export DISABLE_WANDB="${DISABLE_WANDB:-1}"

# Resume policy
export SIT_RESUME_CKPT="${SIT_RESUME_CKPT:-auto}"

# Optional overrides that will be inherited by all runs when set
export LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
export LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
export CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-}"
export SIT_RGB_VAE_CKPT="${SIT_RGB_VAE_CKPT:-}"

# Queue order. Comment out a line if you want to skip one dataset.
RUN_SCRIPTS=(
  "shell/SiT/aviid_sit_train_psl_flow.sh"
  "shell/SiT/cart_sit_train_psl_flow.sh"
  "shell/SiT/dronevehicle_day_sit_train_psl_flow.sh"
  "shell/SiT/dronevehicle_night_sit_train_psl_flow.sh"
)

echo "[PSL-Flow-Queue] Repository: ${REPO_ROOT}"
echo "[PSL-Flow-Queue] USE_FULL_DATASET_EPOCH=${USE_FULL_DATASET_EPOCH}"
echo "[PSL-Flow-Queue] NUM_EPOCHS=${NUM_EPOCHS}"
echo "[PSL-Flow-Queue] TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
echo "[PSL-Flow-Queue] TEST_BATCH_SIZE=${TEST_BATCH_SIZE}"
echo "[PSL-Flow-Queue] NUM_WORKERS=${NUM_WORKERS}"
echo "[PSL-Flow-Queue] LEARNING_RATE=${LEARNING_RATE}"
echo "[PSL-Flow-Queue] WEIGHT_DECAY=${WEIGHT_DECAY}"
echo "[PSL-Flow-Queue] GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION}"
echo "[PSL-Flow-Queue] MIXED_PRECISION=${MIXED_PRECISION}"
echo "[PSL-Flow-Queue] TRAIN_IMAGE_SIZE=${TRAIN_IMAGE_SIZE}"
echo "[PSL-Flow-Queue] VAL_FREQ=${VAL_FREQ}"
echo "[PSL-Flow-Queue] PL_DEVICES=${PL_DEVICES}"
echo "[PSL-Flow-Queue] PL_NUM_NODES=${PL_NUM_NODES}"
echo "[PSL-Flow-Queue] PL_ACCELERATOR=${PL_ACCELERATOR}"
echo "[PSL-Flow-Queue] PL_STRATEGY=${PL_STRATEGY}"
echo "[PSL-Flow-Queue] DISABLE_WANDB=${DISABLE_WANDB}"
echo "[PSL-Flow-Queue] SIT_RESUME_CKPT=${SIT_RESUME_CKPT}"
echo "[PSL-Flow-Queue] Queue size=${#RUN_SCRIPTS[@]}"
echo "[PSL-Flow-Queue] NUM_SAMPLES_PER_EPOCH_AVIID=${NUM_SAMPLES_PER_EPOCH_AVIID}"
echo "[PSL-Flow-Queue] NUM_SAMPLES_PER_EPOCH_CART=${NUM_SAMPLES_PER_EPOCH_CART}"
echo "[PSL-Flow-Queue] NUM_SAMPLES_PER_EPOCH_DRONEVEHICLE_DAY=${NUM_SAMPLES_PER_EPOCH_DRONEVEHICLE_DAY}"
echo "[PSL-Flow-Queue] NUM_SAMPLES_PER_EPOCH_DRONEVEHICLE_NIGHT=${NUM_SAMPLES_PER_EPOCH_DRONEVEHICLE_NIGHT}"

for run_script in "${RUN_SCRIPTS[@]}"; do
  dataset_tag=""
  local_num_samples=""
  case "${run_script}" in
    *aviid*)
      dataset_tag="AVIID"
      local_num_samples="${NUM_SAMPLES_PER_EPOCH_AVIID}"
      ;;
    *cart*)
      dataset_tag="CART"
      local_num_samples="${NUM_SAMPLES_PER_EPOCH_CART}"
      ;;
    *dronevehicle_day*)
      dataset_tag="DroneVehicle_day"
      local_num_samples="${NUM_SAMPLES_PER_EPOCH_DRONEVEHICLE_DAY}"
      ;;
    *dronevehicle_night*)
      dataset_tag="DroneVehicle_night"
      local_num_samples="${NUM_SAMPLES_PER_EPOCH_DRONEVEHICLE_NIGHT}"
      ;;
    *)
      dataset_tag="UNKNOWN"
      ;;
  esac

  if [[ "${USE_FULL_DATASET_EPOCH}" == "0" && -n "${local_num_samples}" ]]; then
    export NUM_SAMPLES_PER_EPOCH="${local_num_samples}"
    steps_per_epoch=$(( (local_num_samples + TRAIN_BATCH_SIZE - 1) / TRAIN_BATCH_SIZE ))
    total_steps=$(( steps_per_epoch * NUM_EPOCHS ))
    global_samples_per_epoch=$(( local_num_samples * PL_DEVICES ))
  else
    unset NUM_SAMPLES_PER_EPOCH
    steps_per_epoch="auto"
    total_steps="auto"
    global_samples_per_epoch="auto"
  fi

  echo
  echo "============================================================"
  echo "[PSL-Flow-Queue] Starting: ${run_script}"
  echo "[PSL-Flow-Queue] Dataset=${dataset_tag}"
  echo "[PSL-Flow-Queue] NUM_SAMPLES_PER_EPOCH=${NUM_SAMPLES_PER_EPOCH:-<full-dataset>}"
  echo "[PSL-Flow-Queue] approx_steps_per_epoch=${steps_per_epoch}"
  echo "[PSL-Flow-Queue] approx_total_steps=${total_steps}"
  echo "[PSL-Flow-Queue] approx_global_samples_per_epoch=${global_samples_per_epoch}"
  echo "============================================================"
  bash "${run_script}"
  echo "[PSL-Flow-Queue] Finished: ${run_script}"
done

echo
echo "[PSL-Flow-Queue] All runs completed."
