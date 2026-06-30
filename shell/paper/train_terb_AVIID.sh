#!/usr/bin/env bash
set -Eeuo pipefail
CONFIG="${CONFIG:-psl_flow/configs/paper/terb_AVIID.yml}"
ROOT="${ROOT:-logs/paper/AVIID/terb}"
exec python -m psl_flow.training.train_terb --config "${CONFIG}" --default-root-dir "${ROOT}" "$@"
