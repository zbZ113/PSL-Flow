#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

strip_cr_var() {
  local var_name
  for var_name in "$@"; do
    printf -v "${var_name}" '%s' "${!var_name//$'\r'/}"
  done
}

SIT_DATASET="${SIT_DATASET:-UNKNOWN}"
TEST_CFG="${TEST_CFG:-configs/test/sit_cond/psl_flow_l2_concat.yml}"
SIT_SOURCE_RUN_DIR="${SIT_SOURCE_RUN_DIR:-auto}"
SIT_CKPT="${SIT_CKPT:-auto}"
SIT_CKPT_SELECT="${SIT_CKPT_SELECT:-best}"
SIT_VAE_CKPT="${SIT_VAE_CKPT:-auto}"
SIT_VAE_SELECT="${SIT_VAE_SELECT:-best_fid}"
SIT_VAE_EPOCH="${SIT_VAE_EPOCH:-}"
SIT_VAE_METRICS_CSV="${SIT_VAE_METRICS_CSV:-}"
SIT_VAE_RUN_DIR="${SIT_VAE_RUN_DIR:-}"
SIT_VAE_CKPT_DIR="${SIT_VAE_CKPT_DIR:-}"
SIT_RGB_VAE_CKPT="${SIT_RGB_VAE_CKPT:-}"
TERB_NET_CKPT="${TERB_NET_CKPT:-}"
PSL_RECOMPOSE_MODE="${PSL_RECOMPOSE_MODE:-full}"
SAVE_ALL_EVAL_SAMPLES="${SAVE_ALL_EVAL_SAMPLES:-1}"
EVAL_VIS_NUM="${EVAL_VIS_NUM:-}"
EVAL_SPLITS="${EVAL_SPLITS:-both}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-logs/psl_flow_ablation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUT_ROOT_BASE}/${SIT_DATASET}/${PSL_RECOMPOSE_MODE}}"

TEST_DEVICES="${TEST_DEVICES:-1}"
TEST_NUM_NODES="${TEST_NUM_NODES:-1}"
TEST_ACCELERATOR="${TEST_ACCELERATOR:-gpu}"
TEST_STRATEGY="${TEST_STRATEGY:-auto}"
DISABLE_WANDB="${DISABLE_WANDB:-1}"

strip_cr_var \
  SIT_DATASET \
  TEST_CFG \
  SIT_SOURCE_RUN_DIR \
  SIT_CKPT \
  SIT_CKPT_SELECT \
  SIT_VAE_CKPT \
  SIT_VAE_SELECT \
  SIT_VAE_EPOCH \
  SIT_VAE_METRICS_CSV \
  SIT_VAE_RUN_DIR \
  SIT_VAE_CKPT_DIR \
  SIT_RGB_VAE_CKPT \
  TERB_NET_CKPT \
  PSL_RECOMPOSE_MODE \
  SAVE_ALL_EVAL_SAMPLES \
  EVAL_VIS_NUM \
  EVAL_SPLITS \
  OUTPUT_ROOT_BASE \
  OUTPUT_ROOT \
  TEST_DEVICES \
  TEST_NUM_NODES \
  TEST_ACCELERATOR \
  TEST_STRATEGY \
  DISABLE_WANDB

if [[ "${SIT_DATASET}" == "UNKNOWN" ]]; then
  echo "[ERR] SIT_DATASET is empty or unknown."
  exit 1
fi

if [[ ! -f "${TEST_CFG}" ]]; then
  echo "[ERR] TEST_CFG not found: ${TEST_CFG}"
  exit 1
fi

if [[ -z "${TERB_NET_CKPT}" ]]; then
  CANDIDATE_TEACHER_CKPTS=(
    "checkpoints/physics/${SIT_DATASET}/terb_net/teacher_best.pth"
    "checkpoints/physics/${SIT_DATASET}/terb_net/teacher_last.pth"
    "logs/physics/${SIT_DATASET}/terb_net/states/best.pth"
    "logs/physics/${SIT_DATASET}/terb_net/states/last.pth"
    "checkpoints/physics/${SIT_DATASET}/teacher/teacher_best.pth"
    "checkpoints/physics/${SIT_DATASET}/teacher/teacher_last.pth"
    "logs/physics/${SIT_DATASET}/teacher/states/best.pth"
    "logs/physics/${SIT_DATASET}/teacher/states/last.pth"
  )
  for candidate in "${CANDIDATE_TEACHER_CKPTS[@]}"; do
    if [[ -f "${candidate}" ]]; then
      TERB_NET_CKPT="${candidate}"
      break
    fi
  done
fi

if [[ -z "${TERB_NET_CKPT}" ]] || [[ ! -f "${TERB_NET_CKPT}" ]]; then
  echo "[ERR] Unable to resolve TERB_NET_CKPT for ${SIT_DATASET}."
  echo "[ERR] Tried checkpoints/physics/${SIT_DATASET}/terb_net/teacher_best.pth and legacy teacher fallbacks."
  exit 1
fi
if [[ "${SIT_SOURCE_RUN_DIR}" == "auto" ]]; then
  SIT_SOURCE_RUN_DIR="$(python - "${SIT_DATASET}" <<'PY'
import os
import sys
from pathlib import Path

def slugify(name: str) -> str:
    return ''.join(ch for ch in name.lower() if ch.isalnum())

dataset_name = sys.argv[1]
root = Path('logs') / 'psl_flow'
ds_slug = slugify(dataset_name)
candidates = []
if root.is_dir():
    for item in root.iterdir():
        if item.is_dir() and slugify(item.name).startswith(ds_slug):
            candidates.append(item)
if not candidates:
    print('')
else:
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    print(str(candidates[0]))
PY
)"
fi

if [[ -z "${SIT_SOURCE_RUN_DIR}" ]] || [[ ! -d "${SIT_SOURCE_RUN_DIR}" ]]; then
  echo "[ERR] Unable to resolve SIT_SOURCE_RUN_DIR for ${SIT_DATASET}."
  exit 1
fi

if [[ "${SIT_CKPT}" == "auto" ]]; then
  SIT_CKPT="${SIT_SOURCE_RUN_DIR}/checkpoints/${SIT_CKPT_SELECT}.ckpt"
fi
if [[ ! -f "${SIT_CKPT}" ]]; then
  echo "[ERR] SIT_CKPT not found: ${SIT_CKPT}"
  exit 1
fi

RESOLVE_JSON="$(python - "${SIT_DATASET}" "${SIT_VAE_SELECT}" "${SIT_VAE_EPOCH}" "${SIT_VAE_CKPT}" "${SIT_VAE_METRICS_CSV}" "${SIT_VAE_RUN_DIR}" "${SIT_VAE_CKPT_DIR}" <<'PY'
import csv
import glob
import json
import os
import sys

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
                if os.path.isdir(full) and slugify(name).startswith(ds_slug):
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
    candidates.append(os.path.join('logs', 'psl_vae', f'{slugify(dataset_name)}_metrics.csv'))
    for item in dict.fromkeys(candidates):
        if item and os.path.isfile(item):
            return item
    return ''

def choose_row(rows, mode: str, epoch_value: str):
    if not rows:
        return None
    keys = rows[0].keys()
    fid_key = 'val_all/FID' if 'val_all/FID' in keys else next((k for k in keys if k.endswith('/FID')), None)
    lpips_key = 'val_all/LPIPS' if 'val_all/LPIPS' in keys else next((k for k in keys if k.endswith('/LPIPS')), None)
    psnr_key = 'val_all/PSNR' if 'val_all/PSNR' in keys else next((k for k in keys if k.endswith('/PSNR')), None)
    ssim_key = 'val_all/SSIM' if 'val_all/SSIM' in keys else next((k for k in keys if k.endswith('/SSIM')), None)
    usable = [r for r in rows if r.get('event') == 'validation_end']
    if not usable:
        return None
    if mode == 'last':
        return usable[-1]
    if mode == 'best_fid':
        pool = [r for r in usable if try_float(r.get(fid_key)) is not None]
        return min(pool, key=lambda r: float(r[fid_key])) if pool else None
    if mode == 'best_lpips':
        pool = [r for r in usable if try_float(r.get(lpips_key)) is not None]
        return min(pool, key=lambda r: float(r[lpips_key])) if pool else None
    if mode == 'best_psnr':
        pool = [r for r in usable if try_float(r.get(psnr_key)) is not None]
        return max(pool, key=lambda r: float(r[psnr_key])) if pool else None
    if mode == 'best_ssim':
        pool = [r for r in usable if try_float(r.get(ssim_key)) is not None]
        return max(pool, key=lambda r: float(r[ssim_key])) if pool else None
    if mode == 'epoch':
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
    dirs = []
    for item in ([ckpt_dir] if ckpt_dir else []) + [os.path.join(rd, 'checkpoints') for rd in find_candidate_run_dirs(dataset_name, run_dir)] + [os.path.join('checkpoints', 'psl_vae', dataset_name, 'checkpoints'), os.path.join('checkpoints', 'psl_vae_lpips01', dataset_name, 'checkpoints')]:
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
    patterns = [f'*_{zero_epoch}_*.ckpt', f'*epoch={zero_epoch}*.ckpt', f'*_{picked_epoch}_*.ckpt', f'*epoch={picked_epoch}*.ckpt']
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
print(json.dumps({'metrics_csv': metrics_path, 'picked_epoch': picked_epoch, 'thermal_normalizer': normalizer, 'vae_ckpt': resolved_ckpt, 'selection': mode}, ensure_ascii=False))
PY
)"

SIT_VAE_METRICS_CSV_RESOLVED="$(printf '%s' "${RESOLVE_JSON}" | python -c "import json,sys; print(json.load(sys.stdin).get('metrics_csv',''))")"
SIT_VAE_PICKED_EPOCH="$(printf '%s' "${RESOLVE_JSON}" | python -c "import json,sys; v=json.load(sys.stdin).get('picked_epoch',''); print('' if v is None else v)")"
SIT_THERMAL_NORMALIZER="$(printf '%s' "${RESOLVE_JSON}" | python -c "import json,sys; v=json.load(sys.stdin).get('thermal_normalizer',''); print('' if v is None else v)")"
SIT_VAE_CKPT_RESOLVED="$(printf '%s' "${RESOLVE_JSON}" | python -c "import json,sys; print(json.load(sys.stdin).get('vae_ckpt',''))")"

if [[ -z "${SIT_VAE_CKPT_RESOLVED}" ]] || [[ ! -f "${SIT_VAE_CKPT_RESOLVED}" ]]; then
  echo "[ERR] Unable to resolve PSL-VAE checkpoint."
  exit 1
fi
if [[ -z "${SIT_THERMAL_NORMALIZER}" ]]; then
  echo "[ERR] Unable to resolve thermal_normalizer from metrics.csv."
  exit 1
fi
if [[ -n "${SIT_RGB_VAE_CKPT}" ]] && [[ ! -f "${SIT_RGB_VAE_CKPT}" ]]; then
  echo "[ERR] SIT_RGB_VAE_CKPT not found: ${SIT_RGB_VAE_CKPT}"
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
TMP_CFG="$(mktemp "/tmp/psl_flow_ablation_${SIT_DATASET}_XXXXXX.yml")"
cleanup() {
  rm -f "${TMP_CFG}"
}
trap cleanup EXIT

python - "${TEST_CFG}" "${TMP_CFG}" "${SIT_DATASET}" "${SIT_CKPT}" "${SIT_VAE_CKPT_RESOLVED}" "${SIT_RGB_VAE_CKPT}" "${TERB_NET_CKPT}" "${SIT_THERMAL_NORMALIZER}" "${PSL_RECOMPOSE_MODE}" "${SAVE_ALL_EVAL_SAMPLES}" "${EVAL_VIS_NUM}" <<'PY'
import sys
import yaml

test_cfg, out_cfg, dataset_name, sit_ckpt, vae_ckpt, rgb_vae_ckpt, teacher_ckpt, thermal_normalizer, recompose_mode, save_all_eval_samples, eval_vis_num = sys.argv[1:]

with open(test_cfg, 'r', encoding='utf-8') as handle:
    cfg = yaml.safe_load(handle)

cfg['datasets']['train_datasets'] = [dataset_name]
cfg['datasets']['val_datasets'] = [dataset_name]
cfg['datasets']['test_datasets'] = [dataset_name]
cfg['datasets']['target_val_dataset'] = dataset_name

model_cfg = cfg.setdefault('model', {}).setdefault('model_config', {})
model_cfg['vae_path'] = vae_ckpt
model_cfg['thermal_normalizer'] = float(thermal_normalizer)
model_cfg['psl_recompose_mode'] = str(recompose_mode)
model_cfg['save_all_eval_samples'] = str(save_all_eval_samples).lower() in {'1', 'true', 'yes', 'y', 'on'}
model_cfg['save_eval_images_local'] = True
if eval_vis_num not in ('', None):
    model_cfg['eval_vis_num'] = int(eval_vis_num)
if rgb_vae_ckpt:
    model_cfg['rgb_vae_path'] = rgb_vae_ckpt
if teacher_ckpt:
    model_cfg.setdefault('teacher', {})['ckpt'] = teacher_ckpt

cfg.setdefault('training', {})
cfg['training']['load'] = sit_ckpt

with open(out_cfg, 'w', encoding='utf-8') as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

echo "[PSL-Flow-Ablation] SIT_DATASET=${SIT_DATASET}"
echo "[PSL-Flow-Ablation] SIT_SOURCE_RUN_DIR=${SIT_SOURCE_RUN_DIR}"
echo "[PSL-Flow-Ablation] SIT_CKPT=${SIT_CKPT}"
echo "[PSL-Flow-Ablation] SIT_VAE_CKPT=${SIT_VAE_CKPT_RESOLVED}"
echo "[PSL-Flow-Ablation] thermal_normalizer=${SIT_THERMAL_NORMALIZER}"
echo "[PSL-Flow-Ablation] PSL_RECOMPOSE_MODE=${PSL_RECOMPOSE_MODE}"
echo "[PSL-Flow-Ablation] SAVE_ALL_EVAL_SAMPLES=${SAVE_ALL_EVAL_SAMPLES}"
echo "[PSL-Flow-Ablation] EVAL_SPLITS=${EVAL_SPLITS}"
echo "[PSL-Flow-Ablation] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[PSL-Flow-Ablation] TMP_CFG=${TMP_CFG}"

if [[ "${DISABLE_WANDB}" == "1" ]]; then
  export WANDB_MODE="disabled"
fi

CMD=(python test.py --config "${TMP_CFG}" --devices "${TEST_DEVICES}" --num-nodes "${TEST_NUM_NODES}" --accelerator "${TEST_ACCELERATOR}" --strategy "${TEST_STRATEGY}" --default-root-dir "${OUTPUT_ROOT}" --vae-path "${SIT_VAE_CKPT_RESOLVED}" --eval-splits "${EVAL_SPLITS}")
if [[ -n "${SIT_RGB_VAE_CKPT}" ]]; then
  CMD+=(--rgb-vae-path "${SIT_RGB_VAE_CKPT}")
fi
"${CMD[@]}"
