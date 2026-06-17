#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

BASE_CFG="${BASE_CFG:-configs/test/ldm/klvae_all_256_1st.yml}"
KLVAE_CKPT="${KLVAE_CKPT:-checkpoints/klvae/checkpoints/klvae_122_FID[7.8546]_LPIPS[0.0581].ckpt}"
TEST_DEVICES="${TEST_DEVICES:-1}"
TEST_NUM_NODES="${TEST_NUM_NODES:-1}"
TEST_ACCELERATOR="${TEST_ACCELERATOR:-gpu}"
TEST_STRATEGY="${TEST_STRATEGY:-auto}"
DISABLE_WANDB="${DISABLE_WANDB:-1}"
TEST_RUN_ROOT="${TEST_RUN_ROOT:-logs/klvae/shared}"

if [[ ! -f "${BASE_CFG}" ]]; then
  echo "[ERR] BASE_CFG not found: ${BASE_CFG}"
  exit 1
fi

if [[ ! -f "${KLVAE_CKPT}" ]]; then
  echo "[ERR] KLVAE_CKPT not found: ${KLVAE_CKPT}"
  exit 1
fi

TMP_CFG="$(mktemp "/tmp/klvae_test_shared_XXXXXX.yml")"
cleanup() {
  rm -f "${TMP_CFG}"
}
trap cleanup EXIT

python - "${BASE_CFG}" "${TMP_CFG}" "${KLVAE_CKPT}" <<'PY'
import sys
import yaml

base_cfg, out_cfg, klvae_ckpt = sys.argv[1:]

with open(base_cfg, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

cfg.setdefault("training", {})
cfg["training"]["load"] = klvae_ckpt

with open(out_cfg, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

echo "[KLVAE-Test-Shared] BASE_CFG=${BASE_CFG}"
echo "[KLVAE-Test-Shared] TMP_CFG=${TMP_CFG}"
echo "[KLVAE-Test-Shared] KLVAE_CKPT=${KLVAE_CKPT}"
echo "[KLVAE-Test-Shared] TEST_RUN_ROOT=${TEST_RUN_ROOT}"
echo "[KLVAE-Test-Shared] TEST_DEVICES=${TEST_DEVICES}"
echo "[KLVAE-Test-Shared] TEST_NUM_NODES=${TEST_NUM_NODES}"
echo "[KLVAE-Test-Shared] TEST_ACCELERATOR=${TEST_ACCELERATOR}"
echo "[KLVAE-Test-Shared] TEST_STRATEGY=${TEST_STRATEGY}"

CMD=(python test.py --config "${TMP_CFG}" --devices "${TEST_DEVICES}" --num-nodes "${TEST_NUM_NODES}" --accelerator "${TEST_ACCELERATOR}" --strategy "${TEST_STRATEGY}" --default-root-dir "${TEST_RUN_ROOT}")
if [[ "${DISABLE_WANDB}" == "1" ]]; then
  CMD+=(--disable-wandb)
fi
"${CMD[@]}"
