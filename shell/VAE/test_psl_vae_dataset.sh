#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

BASE_CFG="${BASE_CFG:-configs/test/ldm/psl_vae_all_256_1st.yml}"
DATASET_NAME="${DATASET_NAME:-UNKNOWN}"
VAE_ROOT="${VAE_ROOT:-checkpoints/psl_vae}"
TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-${DATASET_NAME}}"
VAL_DATASET_NAME="${VAL_DATASET_NAME:-${DATASET_NAME}}"
TEST_DATASET_NAME="${TEST_DATASET_NAME:-${DATASET_NAME}}"
TARGET_VAL_DATASET="${TARGET_VAL_DATASET:-${VAL_DATASET_NAME}}"
PSL_VAE_CKPT="${PSL_VAE_CKPT:-${VAE_ROOT}/${DATASET_NAME}/checkpoints/last.ckpt}"
TERB_NET_CKPT="${TERB_NET_CKPT:-}"

TEST_DEVICES="${TEST_DEVICES:-1}"
TEST_NUM_NODES="${TEST_NUM_NODES:-1}"
TEST_ACCELERATOR="${TEST_ACCELERATOR:-gpu}"
TEST_STRATEGY="${TEST_STRATEGY:-auto}"
DISABLE_WANDB="${DISABLE_WANDB:-1}"

if [[ ! -f "${BASE_CFG}" ]]; then
  echo "[ERR] BASE_CFG not found: ${BASE_CFG}"
  exit 1
fi

if [[ "${DATASET_NAME}" == "UNKNOWN" ]]; then
  echo "[ERR] DATASET_NAME is empty or unknown."
  exit 1
fi

if [[ ! -f "${PSL_VAE_CKPT}" ]]; then
  echo "[ERR] PSL_VAE_CKPT not found: ${PSL_VAE_CKPT}"
  exit 1
fi

TMP_CFG="$(mktemp "/tmp/psl_vae_test_${DATASET_NAME}_XXXXXX.yml")"

python - "${BASE_CFG}" "${TMP_CFG}" "${TRAIN_DATASET_NAME}" "${VAL_DATASET_NAME}" "${TEST_DATASET_NAME}" "${TARGET_VAL_DATASET}" "${PSL_VAE_CKPT}" "${TERB_NET_CKPT}" <<'PY'
import sys
import yaml

(
    base_cfg,
    out_cfg,
    train_dataset_name,
    val_dataset_name,
    test_dataset_name,
    target_val_dataset,
    vae_ckpt,
    teacher_ckpt,
) = sys.argv[1:]

with open(base_cfg, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

cfg["datasets"]["train_datasets"] = [train_dataset_name]
cfg["datasets"]["val_datasets"] = [val_dataset_name]
cfg["datasets"]["test_datasets"] = [test_dataset_name]
cfg["datasets"]["target_val_dataset"] = target_val_dataset
cfg.setdefault("training", {})
cfg["training"]["load"] = vae_ckpt
if teacher_ckpt:
    cfg.setdefault("training", {}).setdefault("loss", {}).setdefault("config", {}).setdefault("teacher", {})
    cfg["training"]["loss"]["config"]["teacher"]["ckpt"] = teacher_ckpt

with open(out_cfg, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

echo "[PSL-VAE-Test] DATASET_NAME=${DATASET_NAME}"
echo "[PSL-VAE-Test] BASE_CFG=${BASE_CFG}"
echo "[PSL-VAE-Test] TMP_CFG=${TMP_CFG}"
echo "[PSL-VAE-Test] PSL_VAE_CKPT=${PSL_VAE_CKPT}"
echo "[PSL-VAE-Test] TERB_NET_CKPT=${TERB_NET_CKPT:-<base_cfg>}"

CMD=(python test.py --config "${TMP_CFG}" --devices "${TEST_DEVICES}" --num-nodes "${TEST_NUM_NODES}" --accelerator "${TEST_ACCELERATOR}" --strategy "${TEST_STRATEGY}")
if [[ "${DISABLE_WANDB}" == "1" ]]; then
  CMD+=(--disable-wandb)
fi
"${CMD[@]}"

rm -f "${TMP_CFG}"
