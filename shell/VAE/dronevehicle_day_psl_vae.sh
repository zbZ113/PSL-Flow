#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export DATASET_NAME="${DATASET_NAME:-DroneVehicle_day}"
export TERB_NET_CKPT="${TERB_NET_CKPT:-checkpoints/physics/${DATASET_NAME}/terb_net/teacher_best.pth}"
export RUN_ROOT="${RUN_ROOT:-logs/psl_vae/${DATASET_NAME}}"
export VAE_ROOT="${VAE_ROOT:-checkpoints/psl_vae}"

bash shell/VAE/train_psl_vae_dataset.sh