#!/usr/bin/env bash
set -Eeuo pipefail
CONFIG="${CONFIG:-psl_flow/configs/paper/klvae_sit_DroneVehicle_night.yml}"
ROOT="${ROOT:-logs/paper/DroneVehicle_night/klvae_sit}"
exec python -m psl_flow.training.train_psl_flow --config "${CONFIG}" --mode fit --default-root-dir "${ROOT}" "$@"
