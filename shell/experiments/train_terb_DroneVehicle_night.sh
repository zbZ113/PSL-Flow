#!/usr/bin/env bash
set -Eeuo pipefail
CONFIG="${CONFIG:-psl_flow/configs/experiments/terb_DroneVehicle_night.yml}"
ROOT="${ROOT:-logs/experiments/DroneVehicle_night/terb}"
exec python -m psl_flow.training.train_terb --config "${CONFIG}" --default-root-dir "${ROOT}" "$@"
