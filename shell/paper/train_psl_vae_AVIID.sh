#!/usr/bin/env bash
set -Eeuo pipefail
CONFIG="${CONFIG:-psl_flow/configs/paper/psl_vae_AVIID.yml}"
ROOT="${ROOT:-logs/paper/AVIID/psl_vae}"
exec python -m psl_flow.training.train_psl_vae --config "${CONFIG}" --default-root-dir "${ROOT}" "$@"
