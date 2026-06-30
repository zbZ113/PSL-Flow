#!/usr/bin/env bash
set -Eeuo pipefail
CONFIG="${CONFIG:-psl_flow/configs/paper/psl_flow_DroneVehicle_day.yml}"
ROOT="${ROOT:-logs/paper/DroneVehicle_day/psl_flow}"
exec python -m psl_flow.training.train_psl_flow --config "${CONFIG}" --mode fit --default-root-dir "${ROOT}" "$@"
