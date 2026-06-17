#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

BASE_CFG="${BASE_CFG:-configs/train/ldm/klvae_all_256_1st.yml}"
VAE_ROOT="${VAE_ROOT:-checkpoints/klvae}"
RUN_ROOT="${RUN_ROOT:-logs/klvae/shared}"
DATASETS_FOLDER="${DATASETS_FOLDER:-}"
TRAIN_DATASETS="${TRAIN_DATASETS:-}"
VAL_DATASETS="${VAL_DATASETS:-}"
TEST_DATASETS="${TEST_DATASETS:-}"
TARGET_VAL_DATASET="${TARGET_VAL_DATASET:-}"

PL_DEVICES="${PL_DEVICES:-}"
PL_NUM_NODES="${PL_NUM_NODES:-1}"
PL_ACCELERATOR="${PL_ACCELERATOR:-gpu}"
PL_STRATEGY="${PL_STRATEGY:-ddp}"
DISABLE_WANDB="${DISABLE_WANDB:-1}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MIXED_PRECISION="${MIXED_PRECISION:-False}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-}"
NUM_SAMPLES_PER_EPOCH="${NUM_SAMPLES_PER_EPOCH:-}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-}"
USE_FULL_DATASET_EPOCH="${USE_FULL_DATASET_EPOCH:-1}"
KLVAE_RESUME_CKPT="${KLVAE_RESUME_CKPT:-}"
TRAIN_LOAD_CKPT="${TRAIN_LOAD_CKPT:-}"
TRAIN_LOAD_TYPE="${TRAIN_LOAD_TYPE:-}"

if [[ ! -f "${BASE_CFG}" ]]; then
  echo "[ERR] BASE_CFG not found: ${BASE_CFG}"
  exit 1
fi

if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments"* ]]; then
  echo "[WARN] Current torch does not support CachingAllocator option 'expandable_segments'. Remove it."
  CLEAN_ALLOC_CONF="$(printf '%s\n' "${PYTORCH_CUDA_ALLOC_CONF}" | sed -E 's/(^|,)expandable_segments:[^,]*//g; s/^,+//; s/,+$//; s/,,+/,/g')"
  if [[ -z "${CLEAN_ALLOC_CONF}" ]]; then
    unset PYTORCH_CUDA_ALLOC_CONF
  else
    export PYTORCH_CUDA_ALLOC_CONF="${CLEAN_ALLOC_CONF}"
  fi
fi

mkdir -p "${VAE_ROOT}/checkpoints"
mkdir -p "${RUN_ROOT}"

TMP_CFG="$(mktemp "/tmp/klvae_shared_XXXXXX.yml")"
cleanup() {
  rm -f "${TMP_CFG}"
}
trap cleanup EXIT

if [[ "${KLVAE_RESUME_CKPT}" == "auto" ]]; then
  shopt -s globstar nullglob
  CKPTS_RESUME=("${RUN_ROOT}"/**/checkpoints/last.ckpt)
  if [[ "${#CKPTS_RESUME[@]}" -gt 0 ]]; then
    KLVAE_RESUME_CKPT="$(ls -1t "${CKPTS_RESUME[@]}" | head -n 1)"
    echo "[KLVAE-Shared] Auto resume ckpt=${KLVAE_RESUME_CKPT}"
  else
    echo "[KLVAE-Shared] Auto resume requested but no last.ckpt found under ${RUN_ROOT}, training from scratch."
    KLVAE_RESUME_CKPT=""
  fi
fi

if [[ -n "${KLVAE_RESUME_CKPT}" ]] && [[ ! -f "${KLVAE_RESUME_CKPT}" ]]; then
  echo "[ERR] KLVAE_RESUME_CKPT not found: ${KLVAE_RESUME_CKPT}"
  exit 1
fi

if [[ -n "${TRAIN_LOAD_CKPT}" ]] && [[ ! -f "${TRAIN_LOAD_CKPT}" ]]; then
  echo "[ERR] TRAIN_LOAD_CKPT not found: ${TRAIN_LOAD_CKPT}"
  exit 1
fi

python - "${BASE_CFG}" "${TMP_CFG}" "${TRAIN_BATCH_SIZE}" "${TEST_BATCH_SIZE}" "${NUM_WORKERS}" "${MIXED_PRECISION}" "${GRADIENT_ACCUMULATION}" "${NUM_EPOCHS}" "${NUM_SAMPLES_PER_EPOCH}" "${LIMIT_TRAIN_BATCHES}" "${LIMIT_VAL_BATCHES}" "${CHECK_VAL_EVERY_N_EPOCH}" "${USE_FULL_DATASET_EPOCH}" "${TRAIN_LOAD_CKPT}" "${TRAIN_LOAD_TYPE}" <<'PY'
import json
import os
import sys
import yaml

(
    base_cfg,
    out_cfg,
    train_bs,
    test_bs,
    num_workers,
    mixed_precision,
    gradient_accumulation,
    num_epochs,
    num_samples_per_epoch,
    limit_train_batches,
    limit_val_batches,
    check_val_every_n_epoch,
    use_full_dataset_epoch,
    train_load_ckpt,
    train_load_type,
) = sys.argv[1:]


def parse_dataset_list(value):
    return [item for item in str(value).replace(",", " ").split() if item]

with open(base_cfg, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

datasets_cfg = cfg.setdefault("datasets", {})
datasets_folder_override = os.environ.get("DATASETS_FOLDER", "").strip()
if datasets_folder_override:
    datasets_cfg["datasets_folder"] = datasets_folder_override

train_datasets_override = parse_dataset_list(os.environ.get("TRAIN_DATASETS", ""))
val_datasets_override = parse_dataset_list(os.environ.get("VAL_DATASETS", ""))
test_datasets_override = parse_dataset_list(os.environ.get("TEST_DATASETS", ""))
target_val_override = os.environ.get("TARGET_VAL_DATASET", "").strip()

if train_datasets_override:
    datasets_cfg["train_datasets"] = train_datasets_override
if val_datasets_override:
    datasets_cfg["val_datasets"] = val_datasets_override
elif train_datasets_override:
    datasets_cfg["val_datasets"] = list(train_datasets_override)
if test_datasets_override:
    datasets_cfg["test_datasets"] = test_datasets_override
elif train_datasets_override:
    datasets_cfg["test_datasets"] = list(train_datasets_override)
if target_val_override:
    datasets_cfg["target_val_dataset"] = target_val_override

cfg["training"]["train_batch_size"] = int(train_bs)
cfg["training"]["test_batch_size"] = int(test_bs)
cfg["training"]["num_workers"] = int(num_workers)
cfg["training"]["mixed_precision"] = str(mixed_precision).lower() in {"1", "true", "yes", "y", "on"}
cfg["training"]["gradient_accumulation"] = int(gradient_accumulation)
if num_epochs:
    cfg["training"]["num_epochs"] = int(num_epochs)
if num_samples_per_epoch:
    cfg["training"]["num_samples_per_epoch"] = int(num_samples_per_epoch)
elif str(use_full_dataset_epoch).lower() in {"1", "true", "yes", "y", "on"}:
    datasets_folder = str(cfg.get("datasets", {}).get("datasets_folder", "./datasets_preprocess"))
    train_dataset_names = list(cfg.get("datasets", {}).get("train_datasets", []))
    if not train_dataset_names:
        raise RuntimeError("No train_datasets configured for shared KLVAE training.")
    total_samples = 0
    for dataset_name in train_dataset_names:
        dataset_cfg_path = os.path.join("configs", "datasets", f"{dataset_name}.yml")
        if not os.path.isfile(dataset_cfg_path):
            raise FileNotFoundError(f"Dataset config not found: {dataset_cfg_path}")
        with open(dataset_cfg_path, "r", encoding="utf-8") as f:
            dataset_cfg = yaml.safe_load(f)
        train_splits = dataset_cfg.get("train", [])
        if not train_splits:
            raise RuntimeError(f"No `train` entries in dataset config: {dataset_cfg_path}")
        for split_cfg in train_splits:
            datafolder_name = split_cfg.get("datafolder_name", None)
            if not datafolder_name:
                continue
            metadata_path = os.path.join(datasets_folder, datafolder_name, "metadata.json")
            if not os.path.isfile(metadata_path):
                raise FileNotFoundError(
                    f"metadata.json not found for full-epoch mode: {metadata_path}"
                )
            with open(metadata_path, "r", encoding="utf-8") as meta_f:
                meta = json.load(meta_f)
            total_samples += int(meta.get("num_samples", 0))
    if total_samples <= 0:
        raise RuntimeError("Invalid total_samples computed for shared KLVAE training.")
    cfg["training"]["num_samples_per_epoch"] = int(total_samples)
    print(f"[KLVAE-Shared][CFG] USE_FULL_DATASET_EPOCH enabled, num_samples_per_epoch={total_samples}")
if limit_train_batches:
    cfg["training"]["limit_train_batches"] = float(limit_train_batches)
if limit_val_batches:
    cfg["training"]["limit_val_batches"] = float(limit_val_batches)
if check_val_every_n_epoch:
    cfg["training"]["check_val_every_n_epoch"] = int(check_val_every_n_epoch)
if train_load_ckpt:
    cfg.setdefault("training", {})
    cfg["training"]["load"] = train_load_ckpt
if train_load_type:
    cfg.setdefault("training", {})
    cfg["training"]["load_type"] = train_load_type

with open(out_cfg, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

echo "[KLVAE-Shared] BASE_CFG=${BASE_CFG}"
echo "[KLVAE-Shared] TMP_CFG=${TMP_CFG}"
echo "[KLVAE-Shared] RUN_ROOT=${RUN_ROOT}"
echo "[KLVAE-Shared] VAE_ROOT=${VAE_ROOT}"
echo "[KLVAE-Shared] DATASETS_FOLDER=${DATASETS_FOLDER:-<base_cfg>}"
echo "[KLVAE-Shared] TRAIN_DATASETS=${TRAIN_DATASETS:-<base_cfg>}"
echo "[KLVAE-Shared] VAL_DATASETS=${VAL_DATASETS:-<base_cfg>}"
echo "[KLVAE-Shared] TEST_DATASETS=${TEST_DATASETS:-<base_cfg>}"
echo "[KLVAE-Shared] TARGET_VAL_DATASET=${TARGET_VAL_DATASET:-<base_cfg>}"
echo "[KLVAE-Shared] OUTPUT_LAST=${VAE_ROOT}/checkpoints/last.ckpt"
echo "[KLVAE-Shared] OUTPUT_BEST=${VAE_ROOT}/checkpoints/best.ckpt"
echo "[KLVAE-Shared] TRAIN_LOAD_CKPT=${TRAIN_LOAD_CKPT:-<base_cfg>}"
echo "[KLVAE-Shared] TRAIN_LOAD_TYPE=${TRAIN_LOAD_TYPE:-<base_cfg>}"
echo "[KLVAE-Shared] TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}, TEST_BATCH_SIZE=${TEST_BATCH_SIZE}, NUM_WORKERS=${NUM_WORKERS}, MIXED_PRECISION=${MIXED_PRECISION}, GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION}"
echo "[KLVAE-Shared] NUM_EPOCHS=${NUM_EPOCHS:-<base_cfg>}, NUM_SAMPLES_PER_EPOCH=${NUM_SAMPLES_PER_EPOCH:-<base_cfg>}, USE_FULL_DATASET_EPOCH=${USE_FULL_DATASET_EPOCH}, LIMIT_TRAIN_BATCHES=${LIMIT_TRAIN_BATCHES:-<base_cfg>}, LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES:-<base_cfg>}, CHECK_VAL_EVERY_N_EPOCH=${CHECK_VAL_EVERY_N_EPOCH:-<base_cfg>}"

CMD=(python main.py --config "${TMP_CFG}" --num-nodes "${PL_NUM_NODES}" --accelerator "${PL_ACCELERATOR}" --strategy "${PL_STRATEGY}" --default-root-dir "${RUN_ROOT}")
if [[ -n "${PL_DEVICES}" ]]; then
  CMD+=(--devices "${PL_DEVICES}")
fi
if [[ "${DISABLE_WANDB}" == "1" ]]; then
  CMD+=(--disable-wandb)
fi
if [[ -n "${KLVAE_RESUME_CKPT}" ]]; then
  CMD+=(--resume-from "${KLVAE_RESUME_CKPT}")
fi
"${CMD[@]}"

shopt -s globstar nullglob
LAST_CKPTS=("${RUN_ROOT}"/**/checkpoints/last.ckpt)
if [[ "${#LAST_CKPTS[@]}" -eq 0 ]]; then
  echo "[ERR] No last.ckpt found under ${RUN_ROOT}"
  exit 1
fi

LATEST_LAST_CKPT="$(ls -1t "${LAST_CKPTS[@]}" | head -n 1)"
cp "${LATEST_LAST_CKPT}" "${VAE_ROOT}/checkpoints/last.ckpt"
echo "[KLVAE-Shared] Copied last checkpoint:"
echo "  from ${LATEST_LAST_CKPT}"
echo "  to   ${VAE_ROOT}/checkpoints/last.ckpt"

BEST_CKPTS=("${RUN_ROOT}"/**/checkpoints/*.ckpt)
FILTERED_BEST=()
for ckpt in "${BEST_CKPTS[@]}"; do
  if [[ "${ckpt}" != *"/last.ckpt" ]]; then
    FILTERED_BEST+=("${ckpt}")
  fi
done
if [[ "${#FILTERED_BEST[@]}" -gt 0 ]]; then
  LATEST_BEST_CKPT="$(ls -1t "${FILTERED_BEST[@]}" | head -n 1)"
  cp "${LATEST_BEST_CKPT}" "${VAE_ROOT}/checkpoints/best.ckpt"
  echo "[KLVAE-Shared] Copied best checkpoint:"
  echo "  from ${LATEST_BEST_CKPT}"
  echo "  to   ${VAE_ROOT}/checkpoints/best.ckpt"
fi
