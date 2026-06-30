#!/usr/bin/env bash
set -Eeuo pipefail
CONFIG="${CONFIG:-psl_flow/configs/paper/klvae_sit_DroneVehicle_day.yml}"
ROOT="${ROOT:-logs/paper/DroneVehicle_day/klvae_sit}"
exec python -m psl_flow.training.train_psl_flow --config "${CONFIG}" --mode test --default-root-dir "${ROOT}" "$@"
