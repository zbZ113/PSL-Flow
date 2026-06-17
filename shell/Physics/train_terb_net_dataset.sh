#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

DATASET_NAME="${DATASET_NAME:-UNKNOWN}"
BASE_CFG="${BASE_CFG:-configs/train/sit_cond/psl_flow_l2_concat.yml}"
TERB_NET_LOG_DIR="${TERB_NET_LOG_DIR:-logs/physics/${DATASET_NAME}/terb_net}"
TERB_NET_CKPT_DIR="${TERB_NET_CKPT_DIR:-checkpoints/physics/${DATASET_NAME}/terb_net}"
TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-${DATASET_NAME}}"
VAL_DATASET_NAME="${VAL_DATASET_NAME:-${DATASET_NAME}}"
TEST_DATASET_NAME="${TEST_DATASET_NAME:-${VAL_DATASET_NAME}}"
TARGET_VAL_DATASET="${TARGET_VAL_DATASET:-${VAL_DATASET_NAME}}"
USE_FULL_DATASET_EPOCH="${USE_FULL_DATASET_EPOCH:-1}"
PHYS_DATASETS_FOLDER="${PHYS_DATASETS_FOLDER:-}"

if [[ "${DATASET_NAME}" == "UNKNOWN" ]]; then
  echo "[ERR] DATASET_NAME is empty or unknown."
  exit 1
fi

if [[ ! -f "${BASE_CFG}" ]]; then
  echo "[ERR] BASE_CFG not found: ${BASE_CFG}"
  exit 1
fi

mkdir -p "${TERB_NET_LOG_DIR}" "${TERB_NET_CKPT_DIR}"

TMP_CFG="$(mktemp "/tmp/terb_net_${DATASET_NAME}_XXXXXX.yml")"
cleanup() {
  rm -f "${TMP_CFG}"
}
trap cleanup EXIT

python - "${BASE_CFG}" "${TMP_CFG}" "${TRAIN_DATASET_NAME}" "${VAL_DATASET_NAME}" "${TEST_DATASET_NAME}" "${TARGET_VAL_DATASET}" "${USE_FULL_DATASET_EPOCH}" "${PHYS_DATASETS_FOLDER}" <<'PY'
import json
import os
import sys
import yaml

(base_cfg, out_cfg, train_dataset_name, val_dataset_name, test_dataset_name, target_val_dataset, use_full_dataset_epoch, datasets_folder_override) = sys.argv[1:]

with open(base_cfg, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

cfg["datasets"]["train_datasets"] = [train_dataset_name]
cfg["datasets"]["val_datasets"] = [val_dataset_name]
cfg["datasets"]["test_datasets"] = [test_dataset_name]
cfg["datasets"]["target_val_dataset"] = target_val_dataset
if datasets_folder_override:
    cfg["datasets"]["datasets_folder"] = datasets_folder_override

if str(use_full_dataset_epoch).lower() in {"1", "true", "yes", "y", "on"}:
    dataset_cfg_path = os.path.join("configs", "datasets", f"{train_dataset_name}.yml")
    with open(dataset_cfg_path, "r", encoding="utf-8") as f:
        dataset_cfg = yaml.safe_load(f)
    train_splits = dataset_cfg.get("train", [])
    datasets_folder = datasets_folder_override or str(cfg.get("datasets", {}).get("datasets_folder", "./datasets_preprocess"))
    total_samples = 0
    for split_cfg in train_splits:
        datafolder_name = split_cfg.get("datafolder_name", None)
        if not datafolder_name:
            continue
        metadata_path = os.path.join(datasets_folder, datafolder_name, "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as meta_f:
            meta = json.load(meta_f)
        total_samples += int(meta.get("num_samples", 0))
    if total_samples > 0:
        cfg.setdefault("training", {})
        cfg["training"]["num_samples_per_epoch"] = int(total_samples)
        print(f"[TeR-B Net][CFG] USE_FULL_DATASET_EPOCH enabled, num_samples_per_epoch={total_samples}")

with open(out_cfg, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

export TERB_NET_CFG="${TMP_CFG}"
export TERB_NET_LOG_DIR="${TERB_NET_LOG_DIR}"
export TERB_NET_CKPT_DIR="${TERB_NET_CKPT_DIR}"

echo "[TeR-B-Dataset] DATASET_NAME=${DATASET_NAME}"
echo "[TeR-B-Dataset] TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME}, VAL_DATASET_NAME=${VAL_DATASET_NAME}, TEST_DATASET_NAME=${TEST_DATASET_NAME}, TARGET_VAL_DATASET=${TARGET_VAL_DATASET}"
echo "[TeR-B-Dataset] BASE_CFG=${BASE_CFG}"
echo "[TeR-B-Dataset] TMP_CFG=${TMP_CFG}"
echo "[TeR-B-Dataset] LOG_DIR=${TERB_NET_LOG_DIR}"
echo "[TeR-B-Dataset] CKPT_DIR=${TERB_NET_CKPT_DIR}"
echo "[TeR-B-Dataset] PHYS_DATASETS_FOLDER=${PHYS_DATASETS_FOLDER:-<cfg>}"

bash shell/Physics/train_terb_net.sh
