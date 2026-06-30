#!/usr/bin/env bash
set -Eeuo pipefail
CONFIG="${CONFIG:-psl_flow/configs/paper/terb_DroneVehicle_day.yml}"
ROOT="${ROOT:-logs/paper/DroneVehicle_day/terb}"
exec python -m psl_flow.training.train_terb --config "${CONFIG}" --default-root-dir "${ROOT}" "$@"
