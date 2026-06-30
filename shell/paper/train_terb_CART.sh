#!/usr/bin/env bash
set -Eeuo pipefail
CONFIG="${CONFIG:-psl_flow/configs/paper/terb_CART.yml}"
ROOT="${ROOT:-logs/paper/CART/terb}"
exec python -m psl_flow.training.train_terb --config "${CONFIG}" --default-root-dir "${ROOT}" "$@"
