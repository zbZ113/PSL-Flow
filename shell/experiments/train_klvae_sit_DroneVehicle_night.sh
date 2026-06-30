#!/usr/bin/env bash
set -Eeuo pipefail
CONFIG="${CONFIG:-psl_flow/configs/experiments/klvae_sit_DroneVehicle_night.yml}"
ROOT="${ROOT:-logs/experiments/DroneVehicle_night/klvae_sit}"
exec python -m psl_flow.training.train_psl_flow --config "${CONFIG}" --mode fit --default-root-dir "${ROOT}" "$@"
