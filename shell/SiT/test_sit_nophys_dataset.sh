#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

SIT_DATASET="${SIT_DATASET:-UNKNOWN}"
TEST_CFG="${TEST_CFG:-configs/test/sit_cond/generic_sit_l2_concat.yml}"
SIT_CKPT="${SIT_CKPT:-}"
SIT_RUN_ROOT="${SIT_RUN_ROOT:-logs/generic_sit/${SIT_DATASET}}"
SIT_VAE_CKPT="${SIT_VAE_CKPT:-checkpoints/klvae/checkpoints/klvae_122_FID[7.8546]_LPIPS[0.0581].ckpt}"
SIT_RGB_VAE_CKPT="${SIT_RGB_VAE_CKPT:-}"

TEST_DEVICES="${TEST_DEVICES:-1}"
TEST_NUM_NODES="${TEST_NUM_NODES:-1}"
TEST_ACCELERATOR="${TEST_ACCELERATOR:-gpu}"
TEST_STRATEGY="${TEST_STRATEGY:-auto}"
DISABLE_WANDB="${DISABLE_WANDB:-1}"

if [[ "${SIT_DATASET}" == "UNKNOWN" ]]; then
  echo "[ERR] SIT_DATASET is empty or unknown."
  exit 1
fi

if [[ ! -f "${TEST_CFG}" ]]; then
  echo "[ERR] TEST_CFG not found: ${TEST_CFG}"
  exit 1
fi

if [[ "${SIT_CKPT}" == "auto" ]]; then
  shopt -s globstar nullglob
  CKPTS=("${SIT_RUN_ROOT}"/**/checkpoints/last.ckpt)
  if [[ "${#CKPTS[@]}" -gt 0 ]]; then
    SIT_CKPT="$(ls -1t "${CKPTS[@]}" | head -n 1)"
    echo "[Generic-SiT-Test] Auto ckpt=${SIT_CKPT}"
  else
    echo "[ERR] SIT_CKPT=auto but no last.ckpt found under ${SIT_RUN_ROOT}"
    exit 1
  fi
fi

if [[ -z "${SIT_CKPT}" ]]; then
  echo "[ERR] SIT_CKPT is empty."
  echo "Usage: SIT_CKPT=path/to/last.ckpt bash shell/SiT/test_generic_sit_dataset.sh"
  exit 1
fi

if [[ ! -f "${SIT_CKPT}" ]]; then
  echo "[ERR] SIT_CKPT not found: ${SIT_CKPT}"
  exit 1
fi

if [[ ! -f "${SIT_VAE_CKPT}" ]]; then
  echo "[ERR] Missing shared thermal VAE ckpt: ${SIT_VAE_CKPT}"
  exit 1
fi

if [[ -n "${SIT_RGB_VAE_CKPT}" ]] && [[ ! -f "${SIT_RGB_VAE_CKPT}" ]]; then
  echo "[ERR] SIT_RGB_VAE_CKPT not found: ${SIT_RGB_VAE_CKPT}"
  exit 1
fi

TMP_CFG="$(mktemp "/tmp/generic_sit_test_${SIT_DATASET}_XXXXXX.yml")"
cleanup() {
  rm -f "${TMP_CFG}"
}
trap cleanup EXIT

python - "${TEST_CFG}" "${TMP_CFG}" "${SIT_DATASET}" "${SIT_CKPT}" "${SIT_VAE_CKPT}" <<'PY'
import sys
import yaml

test_cfg, out_cfg, dataset_name, sit_ckpt, vae_ckpt = sys.argv[1:]

with open(test_cfg, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

cfg["datasets"]["train_datasets"] = [dataset_name]
cfg["datasets"]["val_datasets"] = [dataset_name]
cfg["datasets"]["test_datasets"] = [dataset_name]
cfg["datasets"]["target_val_dataset"] = dataset_name

model_cfg = cfg.setdefault("model", {}).setdefault("model_config", {})
model_cfg["vae_path"] = vae_ckpt
model_cfg.pop("tevnet_config", None)
model_cfg.pop("latent_surrogate_config", None)
model_cfg.pop("physics_config", None)

cfg.setdefault("training", {})
cfg["training"]["load"] = sit_ckpt

with open(out_cfg, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

echo "[Generic-SiT-Test] SIT_DATASET=${SIT_DATASET}"
echo "[Generic-SiT-Test] TEST_CFG=${TEST_CFG}"
echo "[Generic-SiT-Test] TMP_CFG=${TMP_CFG}"
echo "[Generic-SiT-Test] SIT_CKPT=${SIT_CKPT}"
echo "[Generic-SiT-Test] SIT_VAE_CKPT=${SIT_VAE_CKPT}"
echo "[Generic-SiT-Test] TEST_DEVICES=${TEST_DEVICES}"
echo "[Generic-SiT-Test] TEST_NUM_NODES=${TEST_NUM_NODES}"
echo "[Generic-SiT-Test] TEST_ACCELERATOR=${TEST_ACCELERATOR}"
echo "[Generic-SiT-Test] TEST_STRATEGY=${TEST_STRATEGY}"
echo "[Generic-SiT-Test] DISABLE_WANDB=${DISABLE_WANDB}"

CMD=(python test.py --config "${TMP_CFG}" --devices "${TEST_DEVICES}" --num-nodes "${TEST_NUM_NODES}" --accelerator "${TEST_ACCELERATOR}" --strategy "${TEST_STRATEGY}" --default-root-dir "${SIT_RUN_ROOT}" --vae-path "${SIT_VAE_CKPT}")
if [[ "${DISABLE_WANDB}" == "1" ]]; then
  CMD+=(--disable-wandb)
fi
if [[ -n "${SIT_RGB_VAE_CKPT}" ]]; then
  CMD+=(--rgb-vae-path "${SIT_RGB_VAE_CKPT}")
fi
"${CMD[@]}"
