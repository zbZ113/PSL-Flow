#!/usr/bin/env bash
set -Eeuo pipefail
CONFIG="${CONFIG:-psl_flow/configs/experiments/klvae_sit/AVIID.yml}"
ROOT="${ROOT:-logs/experiments/AVIID/klvae_sit}"
exec python -m psl_flow.training.train_psl_flow --config "${CONFIG}" --mode fit --default-root-dir "${ROOT}" "$@"
