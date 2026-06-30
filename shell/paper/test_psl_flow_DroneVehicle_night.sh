#!/usr/bin/env bash
set -Eeuo pipefail
CONFIG="${CONFIG:-psl_flow/configs/paper/psl_flow_DroneVehicle_night.yml}"
ROOT="${ROOT:-logs/paper/DroneVehicle_night/psl_flow}"
exec python -m psl_flow.training.train_psl_flow --config "${CONFIG}" --mode test --default-root-dir "${ROOT}" "$@"
