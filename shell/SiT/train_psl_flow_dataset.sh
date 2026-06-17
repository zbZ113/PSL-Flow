#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

SIT_DATASET="${SIT_DATASET:-UNKNOWN}"
TRAIN_CFG="${TRAIN_CFG:-configs/train/sit_cond/psl_flow_l2_concat.yml}"
SIT_VAE_CKPT="${SIT_VAE_CKPT:-auto}"
SIT_VAE_SELECT="${SIT_VAE_SELECT:-best_fid}"
SIT_VAE_EPOCH="${SIT_VAE_EPOCH:-}"
SIT_VAE_METRICS_CSV="${SIT_VAE_METRICS_CSV:-}"
SIT_VAE_RUN_DIR="${SIT_VAE_RUN_DIR:-}"
SIT_VAE_CKPT_DIR="${SIT_VAE_CKPT_DIR:-}"
SIT_RGB_VAE_CKPT="${SIT_RGB_VAE_CKPT:-}"
TERB_NET_CKPT="${TERB_NET_CKPT:-}"
SIT_RUN_ROOT="${SIT_RUN_ROOT:-logs/psl_flow/${SIT_DATASET}}"
SIT_RESUME_CKPT="${SIT_RESUME_CKPT:-}"
USE_FULL_DATASET_EPOCH="${USE_FULL_DATASET_EPOCH:-1}"
SIT_DATASETS_FOLDER="${SIT_DATASETS_FOLDER:-}"
NUM_EPOCHS="${NUM_EPOCHS:-}"
NUM_SAMPLES_PER_EPOCH="${NUM_SAMPLES_PER_EPOCH:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-}"
NUM_WORKERS="${NUM_WORKERS:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-}"
MIXED_PRECISION="${MIXED_PRECISION:-}"
TRAIN_IMAGE_SIZE="${TRAIN_IMAGE_SIZE:-}"
VAL_FREQ="${VAL_FREQ:-}"

PL_DEVICES="${PL_DEVICES:-}"
PL_NUM_NODES="${PL_NUM_NODES:-1}"
PL_ACCELERATOR="${PL_ACCELERATOR:-gpu}"
PL_STRATEGY="${PL_STRATEGY:-auto}"
DISABLE_WANDB="${DISABLE_WANDB:-0}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-}"

if [[ -z "${CHECK_VAL_EVERY_N_EPOCH}" ]] && [[ -n "${VAL_FREQ}" ]]; then
  CHECK_VAL_EVERY_N_EPOCH="${VAL_FREQ}"
fi

if [[ "${SIT_DATASET}" == "UNKNOWN" ]]; then
  echo "[ERR] SIT_DATASET is empty or unknown."
  exit 1
fi

if [[ ! -f "${TRAIN_CFG}" ]]; then
  echo "[ERR] TRAIN_CFG not found: ${TRAIN_CFG}"
  exit 1
fi

RESOLVE_JSON="$(python - "${SIT_DATASET}" "${SIT_VAE_SELECT}" "${SIT_VAE_EPOCH}" "${SIT_VAE_CKPT}" "${SIT_VAE_METRICS_CSV}" "${SIT_VAE_RUN_DIR}" "${SIT_VAE_CKPT_DIR}" <<'PY'
import csv
import glob
import json
import os
import sys
from pathlib import Path


def slugify(name: str) -> str:
    return ''.join(ch for ch in name.lower() if ch.isalnum())


def try_float(x):
    try:
        return float(x)
    except Exception:
        return None


def find_candidate_run_dirs(dataset_name: str, explicit_run_dir: str):
    out = []
    if explicit_run_dir:
        out.append(explicit_run_dir)
    roots = [os.path.join('logs', 'psl_vae')]
    ds_slug = slugify(dataset_name)
    for root in roots:
        if os.path.isdir(root):
            for name in os.listdir(root):
                full = os.path.join(root, name)
                if not os.path.isdir(full):
                    continue
                if slugify(name).startswith(ds_slug):
                    out.append(full)
    seen = []
    for item in out:
        if item and item not in seen:
            seen.append(item)
    return seen


def find_metrics_csv(dataset_name: str, explicit: str, run_dir: str):
    candidates = []
    if explicit:
        candidates.append(explicit)
    for rd in find_candidate_run_dirs(dataset_name, run_dir):
        candidates.append(os.path.join(rd, 'local_logs', 'metrics.csv'))
        candidates.append(os.path.join(rd, 'metrics.csv'))
    ds_slug = slugify(dataset_name)
    candidates.append(os.path.join('logs', 'psl_vae', f'{ds_slug}_metrics.csv'))
    seen = []
    for item in candidates:
        if item and item not in seen:
            seen.append(item)
    for item in seen:
        if os.path.isfile(item):
            return item
    return ''


def choose_row(rows, mode: str, epoch_value: str):
    if not rows:
        return None
    metric_keys = rows[0].keys()
    fid_key = 'val_all/FID' if 'val_all/FID' in metric_keys else next((k for k in metric_keys if k.endswith('/FID')), None)
    lpips_key = 'val_all/LPIPS' if 'val_all/LPIPS' in metric_keys else next((k for k in metric_keys if k.endswith('/LPIPS')), None)
    psnr_key = 'val_all/PSNR' if 'val_all/PSNR' in metric_keys else next((k for k in metric_keys if k.endswith('/PSNR')), None)
    ssim_key = 'val_all/SSIM' if 'val_all/SSIM' in metric_keys else next((k for k in metric_keys if k.endswith('/SSIM')), None)

    usable = [r for r in rows if r.get('event') == 'validation_end']
    if not usable:
        return None

    def min_by(key):
        pool = [r for r in usable if try_float(r.get(key)) is not None]
        return min(pool, key=lambda r: float(r[key])) if pool else None

    def max_by(key):
        pool = [r for r in usable if try_float(r.get(key)) is not None]
        return max(pool, key=lambda r: float(r[key])) if pool else None

    if mode == 'last':
        return usable[-1]
    if mode == 'best_fid':
        return min_by(fid_key)
    if mode == 'best_lpips':
        return min_by(lpips_key)
    if mode == 'best_psnr':
        return max_by(psnr_key)
    if mode == 'best_ssim':
        return max_by(ssim_key)
    if mode == 'epoch':
        if not epoch_value:
            raise SystemExit('SIT_VAE_SELECT=epoch requires SIT_VAE_EPOCH')
        target = int(epoch_value)
        for row in usable:
            if int(row['epoch']) == target:
                return row
        raise SystemExit(f'epoch {target} not found in validation rows')
    raise SystemExit(f'Unsupported SIT_VAE_SELECT={mode}')


def resolve_normalizer(rows, picked_row):
    if picked_row is None:
        return None
    norm = try_float(picked_row.get('latent_normalizer'))
    if norm is not None:
        return norm
    target_epoch = int(picked_row['epoch'])
    train_rows = [r for r in rows if r.get('event') == 'train_epoch_end']
    same = [r for r in train_rows if int(r['epoch']) == target_epoch and try_float(r.get('latent_normalizer')) is not None]
    if same:
        return float(same[-1]['latent_normalizer'])
    later = [r for r in train_rows if try_float(r.get('latent_normalizer')) is not None]
    return float(later[-1]['latent_normalizer']) if later else None


def resolve_ckpt(dataset_name: str, current_ckpt: str, mode: str, picked_epoch: int | None, run_dir: str, ckpt_dir: str):
    if current_ckpt and current_ckpt != 'auto' and os.path.isfile(current_ckpt):
        return current_ckpt

    candidate_dirs = []
    if ckpt_dir:
        candidate_dirs.append(ckpt_dir)
    for rd in find_candidate_run_dirs(dataset_name, run_dir):
        candidate_dirs.append(os.path.join(rd, 'checkpoints'))
    candidate_dirs.append(os.path.join('checkpoints', 'psl_vae', dataset_name, 'checkpoints'))
    candidate_dirs.append(os.path.join('checkpoints', 'psl_vae_lpips01', dataset_name, 'checkpoints'))

    dirs = []
    for item in candidate_dirs:
        if item and item not in dirs and os.path.isdir(item):
            dirs.append(item)

    if mode == 'last':
        for d in dirs:
            p = os.path.join(d, 'last.ckpt')
            if os.path.isfile(p):
                return p
        return ''

    if picked_epoch is None:
        return ''

    zero_epoch = picked_epoch - 1
    patterns = [
        f'*_{zero_epoch}_*.ckpt',
        f'*epoch={zero_epoch}*.ckpt',
        f'*_{picked_epoch}_*.ckpt',
        f'*epoch={picked_epoch}*.ckpt',
    ]
    matches = []
    for d in dirs:
        for pat in patterns:
            matches.extend(glob.glob(os.path.join(d, pat)))
    matches = sorted(set(matches))
    if matches:
        return matches[0]

    for d in dirs:
        p = os.path.join(d, 'best.ckpt')
        if os.path.isfile(p):
            return p
    return ''


dataset_name, mode, epoch_value, current_ckpt, metrics_csv, run_dir, ckpt_dir = sys.argv[1:]
metrics_path = find_metrics_csv(dataset_name, metrics_csv, run_dir)
rows = []
if metrics_path:
    with open(metrics_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
picked = choose_row(rows, mode, epoch_value) if rows else None
picked_epoch = int(picked['epoch']) if picked is not None else None
normalizer = resolve_normalizer(rows, picked)
resolved_ckpt = resolve_ckpt(dataset_name, current_ckpt, mode, picked_epoch, run_dir, ckpt_dir)
print(json.dumps({
    'metrics_csv': metrics_path,
    'picked_epoch': picked_epoch,
    'thermal_normalizer': normalizer,
    'vae_ckpt': resolved_ckpt,
    'selection': mode,
    'row': picked,
}, ensure_ascii=False))
PY
)"

SIT_VAE_METRICS_CSV_RESOLVED="$(printf '%s' "${RESOLVE_JSON}" | python -c "import json,sys; print(json.load(sys.stdin).get('metrics_csv',''))")"
SIT_VAE_PICKED_EPOCH="$(printf '%s' "${RESOLVE_JSON}" | python -c "import json,sys; v=json.load(sys.stdin).get('picked_epoch',''); print('' if v is None else v)")"
SIT_THERMAL_NORMALIZER="$(printf '%s' "${RESOLVE_JSON}" | python -c "import json,sys; v=json.load(sys.stdin).get('thermal_normalizer',''); print('' if v is None else v)")"
SIT_VAE_CKPT_RESOLVED="$(printf '%s' "${RESOLVE_JSON}" | python -c "import json,sys; print(json.load(sys.stdin).get('vae_ckpt',''))")"

if [[ -z "${SIT_VAE_CKPT_RESOLVED}" ]]; then
  echo "[ERR] Unable to resolve PSL-VAE checkpoint."
  echo "[ERR] SIT_VAE_CKPT=${SIT_VAE_CKPT}, SIT_VAE_SELECT=${SIT_VAE_SELECT}, SIT_VAE_EPOCH=${SIT_VAE_EPOCH}, SIT_VAE_RUN_DIR=${SIT_VAE_RUN_DIR}, SIT_VAE_CKPT_DIR=${SIT_VAE_CKPT_DIR}"
  exit 1
fi
if [[ ! -f "${SIT_VAE_CKPT_RESOLVED}" ]]; then
  echo "[ERR] Resolved PSL-VAE ckpt not found: ${SIT_VAE_CKPT_RESOLVED}"
  exit 1
fi
if [[ -z "${SIT_THERMAL_NORMALIZER}" ]]; then
  echo "[ERR] Unable to resolve thermal_normalizer from metrics.csv."
  echo "[ERR] SIT_VAE_METRICS_CSV=${SIT_VAE_METRICS_CSV_RESOLVED:-<none>}"
  exit 1
fi

if [[ -n "${SIT_RGB_VAE_CKPT}" ]] && [[ ! -f "${SIT_RGB_VAE_CKPT}" ]]; then
  echo "[ERR] SIT_RGB_VAE_CKPT not found: ${SIT_RGB_VAE_CKPT}"
  exit 1
fi

mkdir -p "${SIT_RUN_ROOT}"

TMP_CFG="$(mktemp "/tmp/psl_flow_train_${SIT_DATASET}_XXXXXX.yml")"
cleanup() {
  rm -f "${TMP_CFG}"
}
trap cleanup EXIT

python - "${TRAIN_CFG}" "${TMP_CFG}" "${SIT_DATASET}" "${SIT_VAE_CKPT_RESOLVED}" "${USE_FULL_DATASET_EPOCH}" "${SIT_DATASETS_FOLDER}" "${SIT_RGB_VAE_CKPT}" "${TERB_NET_CKPT}" "${SIT_THERMAL_NORMALIZER}" "${NUM_EPOCHS}" "${NUM_SAMPLES_PER_EPOCH}" "${TRAIN_BATCH_SIZE}" "${TEST_BATCH_SIZE}" "${NUM_WORKERS}" "${LEARNING_RATE}" "${WEIGHT_DECAY}" "${GRADIENT_ACCUMULATION}" "${MIXED_PRECISION}" "${TRAIN_IMAGE_SIZE}" "${VAL_FREQ}" <<'PY'
import json
import os
import sys
import yaml

(
    train_cfg,
    out_cfg,
    dataset_name,
    vae_ckpt,
    use_full_dataset_epoch,
    datasets_folder_override,
    rgb_vae_ckpt,
    teacher_ckpt,
    thermal_normalizer,
    num_epochs,
    num_samples_per_epoch,
    train_batch_size,
    test_batch_size,
    num_workers,
    learning_rate,
    weight_decay,
    gradient_accumulation,
    mixed_precision,
    train_image_size,
    val_freq,
) = sys.argv[1:]

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
model_cfg["thermal_normalizer"] = float(thermal_normalizer)
if rgb_vae_ckpt:
    model_cfg["rgb_vae_path"] = rgb_vae_ckpt
if teacher_ckpt:
    model_cfg.setdefault("teacher", {})["ckpt"] = teacher_ckpt

cfg.setdefault("training", {})
training_cfg = cfg["training"]

def maybe_int(value):
    return int(value) if value not in ("", None) else None

def maybe_float(value):
    return float(value) if value not in ("", None) else None

if maybe_int(num_epochs) is not None:
    training_cfg["num_epochs"] = maybe_int(num_epochs)
if maybe_int(num_samples_per_epoch) is not None:
    training_cfg["num_samples_per_epoch"] = maybe_int(num_samples_per_epoch)
if maybe_int(train_batch_size) is not None:
    training_cfg["train_batch_size"] = maybe_int(train_batch_size)
if maybe_int(test_batch_size) is not None:
    training_cfg["test_batch_size"] = maybe_int(test_batch_size)
if maybe_int(num_workers) is not None:
    training_cfg["num_workers"] = maybe_int(num_workers)
if maybe_float(learning_rate) is not None:
    training_cfg.setdefault("optimizer", {})["lr"] = maybe_float(learning_rate)
if maybe_float(weight_decay) is not None:
    training_cfg.setdefault("optimizer", {})["weight_decay"] = maybe_float(weight_decay)
if maybe_int(gradient_accumulation) is not None:
    training_cfg["gradient_accumulation"] = maybe_int(gradient_accumulation)
if mixed_precision not in ("", None):
    training_cfg["mixed_precision"] = str(mixed_precision).lower() in {"1", "true", "yes", "y", "on"}
if maybe_int(val_freq) is not None:
    training_cfg["val_freq"] = maybe_int(val_freq)
if train_image_size not in ("", None):
    dims = [int(x.strip()) for x in str(train_image_size).replace("x", ",").split(",") if x.strip()]
    if len(dims) == 2:
        training_cfg["train_image_size"] = dims

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
        print(f"[PSL-Flow][CFG] USE_FULL_DATASET_EPOCH enabled, num_samples_per_epoch={total_samples}")

with open(out_cfg, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

if [[ "${SIT_RESUME_CKPT}" == "auto" ]]; then
  shopt -s globstar nullglob
  CKPTS=("${SIT_RUN_ROOT}"/**/checkpoints/last.ckpt)
  if [[ "${#CKPTS[@]}" -gt 0 ]]; then
    SIT_RESUME_CKPT="$(ls -1t "${CKPTS[@]}" | head -n 1)"
    echo "[PSL-Flow] Auto resume ckpt=${SIT_RESUME_CKPT}"
  else
    echo "[PSL-Flow] Auto resume requested but no last.ckpt found under ${SIT_RUN_ROOT}, training from scratch."
    SIT_RESUME_CKPT=""
  fi
fi

if [[ -n "${SIT_RESUME_CKPT}" ]] && [[ ! -f "${SIT_RESUME_CKPT}" ]]; then
  echo "[ERR] SIT_RESUME_CKPT not found: ${SIT_RESUME_CKPT}"
  exit 1
fi

echo "[PSL-Flow-Train] SIT_DATASET=${SIT_DATASET}"
echo "[PSL-Flow-Train] TRAIN_CFG=${TRAIN_CFG}"
echo "[PSL-Flow-Train] TMP_CFG=${TMP_CFG}"
echo "[PSL-Flow-Train] SIT_VAE_SELECT=${SIT_VAE_SELECT}"
echo "[PSL-Flow-Train] SIT_VAE_PICKED_EPOCH=${SIT_VAE_PICKED_EPOCH:-<none>}"
echo "[PSL-Flow-Train] SIT_VAE_METRICS_CSV=${SIT_VAE_METRICS_CSV_RESOLVED:-<none>}"
echo "[PSL-Flow-Train] PSL_VAE_CKPT=${SIT_VAE_CKPT_RESOLVED}"
echo "[PSL-Flow-Train] thermal_normalizer=${SIT_THERMAL_NORMALIZER}"
echo "[PSL-Flow-Train] TERB_NET_CKPT=${TERB_NET_CKPT:-<cfg>}"
echo "[PSL-Flow-Train] SIT_RGB_VAE_CKPT=${SIT_RGB_VAE_CKPT:-<default>}"
echo "[PSL-Flow-Train] SIT_RUN_ROOT=${SIT_RUN_ROOT}"
echo "[PSL-Flow-Train] num_epochs=${NUM_EPOCHS:-<cfg>}"
echo "[PSL-Flow-Train] num_samples_per_epoch=${NUM_SAMPLES_PER_EPOCH:-<cfg/auto>}"
echo "[PSL-Flow-Train] train_batch_size=${TRAIN_BATCH_SIZE:-<cfg>}"
echo "[PSL-Flow-Train] test_batch_size=${TEST_BATCH_SIZE:-<cfg>}"
echo "[PSL-Flow-Train] num_workers=${NUM_WORKERS:-<cfg>}"
echo "[PSL-Flow-Train] learning_rate=${LEARNING_RATE:-<cfg>}"
echo "[PSL-Flow-Train] weight_decay=${WEIGHT_DECAY:-<cfg>}"
echo "[PSL-Flow-Train] gradient_accumulation=${GRADIENT_ACCUMULATION:-<cfg>}"
echo "[PSL-Flow-Train] mixed_precision=${MIXED_PRECISION:-<cfg>}"
echo "[PSL-Flow-Train] train_image_size=${TRAIN_IMAGE_SIZE:-<cfg>}"
echo "[PSL-Flow-Train] val_freq=${VAL_FREQ:-<cfg>}"

CMD=(python main.py --config "${TMP_CFG}" --num-nodes "${PL_NUM_NODES}" --accelerator "${PL_ACCELERATOR}" --strategy "${PL_STRATEGY}" --default-root-dir "${SIT_RUN_ROOT}" --vae-path "${SIT_VAE_CKPT_RESOLVED}")
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
