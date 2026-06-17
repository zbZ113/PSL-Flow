#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DATASETS_STR="${DATASETS:-AVIID CART DroneVehicle_day DroneVehicle_night}"
MODES_STR="${PSL_RECOMPOSE_MODES:-full delta_only phys_only}"
EVAL_SPLITS="${EVAL_SPLITS:-both}"
SAVE_ALL_EVAL_SAMPLES="${SAVE_ALL_EVAL_SAMPLES:-1}"
SIT_CKPT_SELECT="${SIT_CKPT_SELECT:-best}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-logs/psl_flow_ablation}"

read -r -a DATASETS <<< "${DATASETS_STR}"
read -r -a MODES <<< "${MODES_STR}"

for dataset_name in "${DATASETS[@]}"; do
  for mode_name in "${MODES[@]}"; do
    echo "[PSL-Flow-Ablation-All] dataset=${dataset_name} mode=${mode_name}"
    SIT_DATASET="${dataset_name}" \
    PSL_RECOMPOSE_MODE="${mode_name}" \
    EVAL_SPLITS="${EVAL_SPLITS}" \
    SAVE_ALL_EVAL_SAMPLES="${SAVE_ALL_EVAL_SAMPLES}" \
    SIT_CKPT_SELECT="${SIT_CKPT_SELECT}" \
    OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE}" \
    DISABLE_WANDB="${DISABLE_WANDB:-1}" \
    TEST_DEVICES="${TEST_DEVICES:-1}" \
    TEST_NUM_NODES="${TEST_NUM_NODES:-1}" \
    TEST_ACCELERATOR="${TEST_ACCELERATOR:-gpu}" \
    TEST_STRATEGY="${TEST_STRATEGY:-auto}" \
    bash shell/SiT/eval_psl_flow_ablation_dataset.sh
  done
done
