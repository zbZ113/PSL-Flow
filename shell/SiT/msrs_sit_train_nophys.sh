#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export SIT_DATASET="${SIT_DATASET:-MSRS}"
export TRAIN_CFG="${TRAIN_CFG:-configs/train/sit_cond/generic_sit_l2_concat.yml}"
export SIT_RUN_ROOT="${SIT_RUN_ROOT:-logs/generic_sit/${SIT_DATASET}}"
export SIT_RESUME_CKPT="${SIT_RESUME_CKPT:-auto}"

bash shell/SiT/train_generic_sit_dataset.sh
