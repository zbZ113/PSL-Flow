#!/usr/bin/env bash
set -euo pipefail

TRAIN_CFG="${TRAIN_CFG:-configs/train/sit_cond/generic_sit_l2_concat.yml}"
SIT_RUN_ROOT="${SIT_RUN_ROOT:-logs/generic_sit/${SIT_DATASET:-UNKNOWN}}"
exec bash shell/SiT/train_sit_nophys_dataset.sh
