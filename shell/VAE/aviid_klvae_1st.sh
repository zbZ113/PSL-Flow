#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export DATASET_NAME="${DATASET_NAME:-AVIID}"
export RUN_ROOT="${RUN_ROOT:-logs/klvae/1st/${DATASET_NAME}}"
export VAE_ROOT="${VAE_ROOT:-checkpoints/klvae_1st}"

bash shell/VAE/train_klvae_1st_dataset.sh
