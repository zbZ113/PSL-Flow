#!/usr/bin/env bash
set -Eeuo pipefail
CONFIG="${CONFIG:-psl_flow/configs/paper/psl_flow_AVIID.yml}"
ROOT="${ROOT:-logs/paper/AVIID/psl_flow}"
exec python -m psl_flow.training.train_psl_flow --config "${CONFIG}" --mode fit --default-root-dir "${ROOT}" "$@"
