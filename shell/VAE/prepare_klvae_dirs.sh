#!/usr/bin/env bash
set -euo pipefail

VAE_ROOT="${VAE_ROOT:-checkpoints/klvae}"

mkdir -p "${VAE_ROOT}/checkpoints"
echo "[VAE] Created shared checkpoint folder: ${VAE_ROOT}/checkpoints"

echo "[VAE] Done."
