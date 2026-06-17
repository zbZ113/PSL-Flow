#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export DATASET_NAME="${DATASET_NAME:-AVIID}"
export TERB_NET_LOG_DIR="${TERB_NET_LOG_DIR:-logs/physics/${DATASET_NAME}/terb_net}"
export TERB_NET_CKPT_DIR="${TERB_NET_CKPT_DIR:-checkpoints/physics/${DATASET_NAME}/terb_net}"

bash shell/Physics/train_terb_net_dataset.sh
