#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

SIT_DATASET="${SIT_DATASET:-UNKNOWN}"
TRAIN_CFG="${TRAIN_CFG:-configs/train/sit_cond/generic_sit_l2_concat.yml}"
SIT_VAE_CKPT="${SIT_VAE_CKPT:-checkpoints/klvae/checkpoints/klvae_122_FID[7.8546]_LPIPS[0.0581].ckpt}"
SIT_RUN_ROOT="${SIT_RUN_ROOT:-logs/generic_sit/${SIT_DATASET}}"
SIT_RESUME_CKPT="${SIT_RESUME_CKPT:-}"
SIT_RGB_VAE_CKPT="${SIT_RGB_VAE_CKPT:-}"
USE_FULL_DATASET_EPOCH="${USE_FULL_DATASET_EPOCH:-1}"
SIT_DATASETS_FOLDER="${SIT_DATASETS_FOLDER:-}"

PL_DEVICES="${PL_DEVICES:-}"
PL_NUM_NODES="${PL_NUM_NODES:-1}"
PL_ACCELERATOR="${PL_ACCELERATOR:-gpu}"
PL_STRATEGY="${PL_STRATEGY:-auto}"
DISABLE_WANDB="${DISABLE_WANDB:-0}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-}"

if [[ "${SIT_DATASET}" == "UNKNOWN" ]]; then
  echo "[ERR] SIT_DATASET is empty or unknown."
  exit 1
fi

if [[ ! -f "${TRAIN_CFG}" ]]; then
  echo "[ERR] TRAIN_CFG not found: ${TRAIN_CFG}"
  exit 1
fi

if [[ ! -f "${SIT_VAE_CKPT}" ]]; then
  echo "[ERR] Missing shared thermal VAE ckpt: ${SIT_VAE_CKPT}"
  echo "[ERR] Train shared KLVAE first: bash shell/VAE/train_klvae_shared.sh"
  exit 1
fi

if [[ -n "${SIT_RGB_VAE_CKPT}" ]] && [[ ! -f "${SIT_RGB_VAE_CKPT}" ]]; then
  echo "[ERR] SIT_RGB_VAE_CKPT not found: ${SIT_RGB_VAE_CKPT}"
  exit 1
fi

mkdir -p "${SIT_RUN_ROOT}"

TMP_CFG="$(mktemp "/tmp/generic_sit_train_${SIT_DATASET}_XXXXXX.yml")"
cleanup() {
  rm -f "${TMP_CFG}"
}
trap cleanup EXIT

python - "${TRAIN_CFG}" "${TMP_CFG}" "${SIT_DATASET}" "${SIT_VAE_CKPT}" "${USE_FULL_DATASET_EPOCH}" "${SIT_DATASETS_FOLDER}" <<'PY'
import json
import os
import sys
import yaml

train_cfg, out_cfg, dataset_name, vae_ckpt, use_full_dataset_epoch, datasets_folder_override = sys.argv[1:]

with open(train_cfg, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

cfg["datasets"]["train_datasets"] = [dataset_name]
cfg["datasets"]["val_datasets"] = [dataset_name]
cfg["datasets"]["test_datasets"] = [dataset_name]
cfg["datasets"]["target_val_dataset"] = dataset_name
if datasets_folder_override:
    cfg["datasets"]["datasets_folder"] = datasets_folder_override

model_cfg = cfg.setdefault("model", {}).setdefault("model_config", {})
model_cfg["vae_path"] = vae_ckpt
model_cfg.pop("tevnet_config", None)
model_cfg.pop("latent_surrogate_config", None)
model_cfg.pop("physics_config", None)

if str(use_full_dataset_epoch).lower() in {"1", "true", "yes", "y", "on"}:
    dataset_cfg_path = os.path.join("configs", "datasets", f"{dataset_name}.yml")
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
        print(f"[Generic-SiT][CFG] USE_FULL_DATASET_EPOCH enabled, num_samples_per_epoch={total_samples}")

with open(out_cfg, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

if [[ "${SIT_RESUME_CKPT}" == "auto" ]]; then
  shopt -s globstar nullglob
  CKPTS=("${SIT_RUN_ROOT}"/**/checkpoints/last.ckpt)
  if [[ "${#CKPTS[@]}" -gt 0 ]]; then
    SIT_RESUME_CKPT="$(ls -1t "${CKPTS[@]}" | head -n 1)"
    echo "[Generic-SiT] Auto resume ckpt=${SIT_RESUME_CKPT}"
  else
    echo "[Generic-SiT] Auto resume requested but no last.ckpt found under ${SIT_RUN_ROOT}, training from scratch."
    SIT_RESUME_CKPT=""
  fi
fi

if [[ -n "${SIT_RESUME_CKPT}" ]] && [[ ! -f "${SIT_RESUME_CKPT}" ]]; then
  echo "[ERR] SIT_RESUME_CKPT not found: ${SIT_RESUME_CKPT}"
  exit 1
fi

echo "[Generic-SiT-Train] SIT_DATASET=${SIT_DATASET}"
echo "[Generic-SiT-Train] TRAIN_CFG=${TRAIN_CFG}"
echo "[Generic-SiT-Train] TMP_CFG=${TMP_CFG}"
echo "[Generic-SiT-Train] SIT_VAE_CKPT=${SIT_VAE_CKPT}"
echo "[Generic-SiT-Train] SIT_RUN_ROOT=${SIT_RUN_ROOT}"
echo "[Generic-SiT-Train] SIT_DATASETS_FOLDER=${SIT_DATASETS_FOLDER:-<cfg>}"
echo "[Generic-SiT-Train] PL_DEVICES=${PL_DEVICES:-auto_visible}"
echo "[Generic-SiT-Train] PL_NUM_NODES=${PL_NUM_NODES}"
echo "[Generic-SiT-Train] PL_ACCELERATOR=${PL_ACCELERATOR}"
echo "[Generic-SiT-Train] PL_STRATEGY=${PL_STRATEGY}"
echo "[Generic-SiT-Train] DISABLE_WANDB=${DISABLE_WANDB}"
if [[ -n "${SIT_RESUME_CKPT}" ]]; then
  echo "[Generic-SiT-Train] SIT_RESUME_CKPT=${SIT_RESUME_CKPT}"
fi

CMD=(python main.py --config "${TMP_CFG}" --num-nodes "${PL_NUM_NODES}" --accelerator "${PL_ACCELERATOR}" --strategy "${PL_STRATEGY}" --default-root-dir "${SIT_RUN_ROOT}" --vae-path "${SIT_VAE_CKPT}")
if [[ -n "${PL_DEVICES}" ]]; then
  CMD+=(--devices "${PL_DEVICES}")
fi
if [[ -n "${LIMIT_TRAIN_BATCHES}" ]]; then
  CMD+=(--limit-train-batches "${LIMIT_TRAIN_BATCHES}")
fi
if [[ -n "${LIMIT_VAL_BATCHES}" ]]; then
  CMD+=(--limit-val-batches "${LIMIT_VAL_BATCHES}")
fi
if [[ -n "${CHECK_VAL_EVERY_N_EPOCH}" ]]; then
  CMD+=(--check-val-every-n-epoch "${CHECK_VAL_EVERY_N_EPOCH}")
fi
if [[ "${DISABLE_WANDB}" == "1" ]]; then
  CMD+=(--disable-wandb)
fi
if [[ -n "${SIT_RESUME_CKPT}" ]]; then
  CMD+=(--resume-from "${SIT_RESUME_CKPT}")
fi
if [[ -n "${SIT_RGB_VAE_CKPT}" ]]; then
  CMD+=(--rgb-vae-path "${SIT_RGB_VAE_CKPT}")
fi
"${CMD[@]}"
