#!/usr/bin/env bash
set -Eeuo pipefail
CONFIG="${CONFIG:-psl_flow/configs/paper/psl_vae_DroneVehicle_night.yml}"
ROOT="${ROOT:-logs/paper/DroneVehicle_night/psl_vae}"
exec python -m psl_flow.training.train_psl_vae --config "${CONFIG}" --default-root-dir "${ROOT}" "$@"
