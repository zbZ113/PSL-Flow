#!/usr/bin/env bash
set -euo pipefail

BASE_CFG="${BASE_CFG:-configs/train/ldm/klvae_all_256_1st.yml}"
DATASET_NAME="${DATASET_NAME:-MSRS}"
VAE_ROOT="${VAE_ROOT:-checkpoints/klvae_1st}"
TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-${DATASET_NAME}}"
VAL_DATASET_NAME="${VAL_DATASET_NAME:-${DATASET_NAME}}"
TEST_DATASET_NAME="${TEST_DATASET_NAME:-${VAL_DATASET_NAME}}"
TARGET_VAL_DATASET="${TARGET_VAL_DATASET:-${VAL_DATASET_NAME}}"
TRAIN_LOAD_CKPT="${TRAIN_LOAD_CKPT:-}"
TRAIN_LOAD_TYPE="${TRAIN_LOAD_TYPE:-}"

PL_DEVICES="${PL_DEVICES:-}"
PL_NUM_NODES="${PL_NUM_NODES:-1}"
PL_ACCELERATOR="${PL_ACCELERATOR:-gpu}"
PL_STRATEGY="${PL_STRATEGY:-ddp}"
DISABLE_WANDB="${DISABLE_WANDB:-1}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MIXED_PRECISION="${MIXED_PRECISION:-True}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-}"
NUM_SAMPLES_PER_EPOCH="${NUM_SAMPLES_PER_EPOCH:-}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-}"
USE_FULL_DATASET_EPOCH="${USE_FULL_DATASET_EPOCH:-0}"
KLVAE_RESUME_CKPT="${KLVAE_RESUME_CKPT:-}"

RUN_ROOT="${RUN_ROOT:-logs/klvae/${DATASET_NAME}}"

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

mkdir -p "${VAE_ROOT}/${DATASET_NAME}/checkpoints"
mkdir -p "${RUN_ROOT}"

TMP_CFG="$(mktemp "/tmp/klvae_${DATASET_NAME}_XXXXXX.yml")"

if [[ "${TRAIN_LOAD_CKPT}" == "auto" ]]; then
  AUTO_CKPT="${VAE_ROOT}/${DATASET_NAME}/checkpoints/last.ckpt"
  if [[ -f "${AUTO_CKPT}" ]]; then
    TRAIN_LOAD_CKPT="${AUTO_CKPT}"
    echo "[KLVAE] Auto train load ckpt=${TRAIN_LOAD_CKPT}"
  else
    echo "[KLVAE] Auto train load requested but not found: ${AUTO_CKPT}"
    TRAIN_LOAD_CKPT=""
  fi
fi

if [[ -n "${TRAIN_LOAD_CKPT}" ]] && [[ ! -f "${TRAIN_LOAD_CKPT}" ]]; then
  echo "[ERR] TRAIN_LOAD_CKPT not found: ${TRAIN_LOAD_CKPT}"
  exit 1
fi

python - "${BASE_CFG}" "${TMP_CFG}" "${TRAIN_DATASET_NAME}" "${VAL_DATASET_NAME}" "${TEST_DATASET_NAME}" "${TARGET_VAL_DATASET}" "${TRAIN_BATCH_SIZE}" "${TEST_BATCH_SIZE}" "${NUM_WORKERS}" "${MIXED_PRECISION}" "${GRADIENT_ACCUMULATION}" "${NUM_EPOCHS}" "${NUM_SAMPLES_PER_EPOCH}" "${LIMIT_TRAIN_BATCHES}" "${LIMIT_VAL_BATCHES}" "${CHECK_VAL_EVERY_N_EPOCH}" "${USE_FULL_DATASET_EPOCH}" "${TRAIN_LOAD_CKPT}" "${TRAIN_LOAD_TYPE}" <<'PY'
import sys
import os
import json
import yaml

(
    base_cfg,
    out_cfg,
    train_dataset_name,
    val_dataset_name,
    test_dataset_name,
    target_val_dataset,
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

with open(base_cfg, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

cfg["datasets"]["train_datasets"] = [train_dataset_name]
cfg["datasets"]["val_datasets"] = [val_dataset_name]
cfg["datasets"]["test_datasets"] = [test_dataset_name]
cfg["datasets"]["target_val_dataset"] = target_val_dataset

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
    dataset_cfg_path = os.path.join("configs", "datasets", f"{train_dataset_name}.yml")
    if not os.path.isfile(dataset_cfg_path):
        raise FileNotFoundError(f"Dataset config not found: {dataset_cfg_path}")
    with open(dataset_cfg_path, "r", encoding="utf-8") as f:
        dataset_cfg = yaml.safe_load(f)
    train_splits = dataset_cfg.get("train", [])
    if not train_splits:
        raise RuntimeError(f"No `train` entries in dataset config: {dataset_cfg_path}")
    datasets_folder = str(cfg.get("datasets", {}).get("datasets_folder", "./datasets_preprocess"))
    total_samples = 0
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
        raise RuntimeError(
            f"Invalid total_samples={total_samples} from {dataset_cfg_path}. "
            "Check metadata.json files."
        )
    cfg["training"]["num_samples_per_epoch"] = int(total_samples)
    print(f"[KLVAE][CFG] USE_FULL_DATASET_EPOCH enabled, num_samples_per_epoch={total_samples}")
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

echo "[KLVAE] DATASET_NAME=${DATASET_NAME}"
echo "[KLVAE] TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME}, VAL_DATASET_NAME=${VAL_DATASET_NAME}, TEST_DATASET_NAME=${TEST_DATASET_NAME}, TARGET_VAL_DATASET=${TARGET_VAL_DATASET}"
echo "[KLVAE] BASE_CFG=${BASE_CFG}"
echo "[KLVAE] TMP_CFG=${TMP_CFG}"
echo "[KLVAE] RUN_ROOT=${RUN_ROOT}"
echo "[KLVAE] OUTPUT_CKPT=${VAE_ROOT}/${DATASET_NAME}/checkpoints/last.ckpt"
echo "[KLVAE] TRAIN_LOAD_CKPT=${TRAIN_LOAD_CKPT:-<base_cfg>}"
echo "[KLVAE] TRAIN_LOAD_TYPE=${TRAIN_LOAD_TYPE:-<base_cfg>}"
echo "[KLVAE] TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}, TEST_BATCH_SIZE=${TEST_BATCH_SIZE}, NUM_WORKERS=${NUM_WORKERS}, MIXED_PRECISION=${MIXED_PRECISION}, GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION}"
echo "[KLVAE] NUM_EPOCHS=${NUM_EPOCHS:-<base_cfg>}, NUM_SAMPLES_PER_EPOCH=${NUM_SAMPLES_PER_EPOCH:-<base_cfg>}, USE_FULL_DATASET_EPOCH=${USE_FULL_DATASET_EPOCH}, LIMIT_TRAIN_BATCHES=${LIMIT_TRAIN_BATCHES:-<base_cfg>}, LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES:-<base_cfg>}, CHECK_VAL_EVERY_N_EPOCH=${CHECK_VAL_EVERY_N_EPOCH:-<base_cfg>}"

if [[ "${KLVAE_RESUME_CKPT}" == "auto" ]]; then
  shopt -s globstar nullglob
  CKPTS_RESUME=("${RUN_ROOT}"/**/checkpoints/last.ckpt)
  if [[ "${#CKPTS_RESUME[@]}" -gt 0 ]]; then
    KLVAE_RESUME_CKPT="$(ls -1t "${CKPTS_RESUME[@]}" | head -n 1)"
    echo "[KLVAE] Auto resume ckpt=${KLVAE_RESUME_CKPT}"
  else
    echo "[KLVAE] Auto resume requested but no last.ckpt found under ${RUN_ROOT}, training from scratch."
    KLVAE_RESUME_CKPT=""
  fi
fi

if [[ -n "${KLVAE_RESUME_CKPT}" ]] && [[ ! -f "${KLVAE_RESUME_CKPT}" ]]; then
  echo "[ERR] KLVAE_RESUME_CKPT not found: ${KLVAE_RESUME_CKPT}"
  rm -f "${TMP_CFG}"
  exit 1
fi

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
CKPTS=("${RUN_ROOT}"/**/checkpoints/last.ckpt)
if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  echo "[ERR] No last.ckpt found under ${RUN_ROOT}"
  rm -f "${TMP_CFG}"
  exit 1
fi

LATEST_CKPT="$(ls -1t "${CKPTS[@]}" | head -n 1)"
TARGET_CKPT="${VAE_ROOT}/${DATASET_NAME}/checkpoints/last.ckpt"
cp "${LATEST_CKPT}" "${TARGET_CKPT}"
echo "[KLVAE] Copied:"
echo "  from ${LATEST_CKPT}"
echo "  to   ${TARGET_CKPT}"

rm -f "${TMP_CFG}"
