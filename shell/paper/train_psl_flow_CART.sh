#!/usr/bin/env bash
set -Eeuo pipefail
CONFIG="${CONFIG:-psl_flow/configs/paper/psl_flow_CART.yml}"
ROOT="${ROOT:-logs/paper/CART/psl_flow}"
exec python -m psl_flow.training.train_psl_flow --config "${CONFIG}" --mode fit --default-root-dir "${ROOT}" "$@"
