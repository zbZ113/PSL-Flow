#!/usr/bin/env bash
set -Eeuo pipefail
CONFIG="${CONFIG:-psl_flow/configs/experiments/psl_vae/DroneVehicle_day.yml}"
ROOT="${ROOT:-logs/experiments/DroneVehicle_day/psl_vae}"
exec python -m psl_flow.training.train_psl_vae --config "${CONFIG}" --default-root-dir "${ROOT}" "$@"
