#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -n "${OMP_NUM_THREADS:-}" ]] && ! [[ "${OMP_NUM_THREADS}" =~ ^[0-9]+$ ]]; then
  unset OMP_NUM_THREADS
fi

TERB_NET_CFG="${TERB_NET_CFG:-configs/train/sit_cond/psl_flow_l2_concat.yml}"
TERB_NET_LOG_DIR="${TERB_NET_LOG_DIR:-logs/physics/terb_net}"
TERB_NET_CKPT_DIR="${TERB_NET_CKPT_DIR:-checkpoints/physics/terb_net}"
TERB_NET_EPOCHS="${TERB_NET_EPOCHS:-1000}"
TERB_NET_MAX_STEPS_PER_EPOCH="${TERB_NET_MAX_STEPS_PER_EPOCH:-0}"
TERB_NET_VAL_EVERY_EPOCHS="${TERB_NET_VAL_EVERY_EPOCHS:-50}"
TERB_NET_SAVE_EVERY_EPOCHS="${TERB_NET_SAVE_EVERY_EPOCHS:-1}"
TERB_NET_LOG_EVERY_STEPS="${TERB_NET_LOG_EVERY_STEPS:-50}"
TERB_NET_SEED="${TERB_NET_SEED:-42}"
TERB_NET_BATCH_SIZE="${TERB_NET_BATCH_SIZE:-0}"
TERB_NET_NUM_WORKERS="${TERB_NET_NUM_WORKERS:-0}"
TERB_NET_DATASETS_FOLDER="${TERB_NET_DATASETS_FOLDER:-}"
TERB_NET_LR="${TERB_NET_LR:-2e-4}"
TERB_NET_WEIGHT_DECAY="${TERB_NET_WEIGHT_DECAY:-1e-4}"
TERB_NET_GRAD_CLIP_NORM="${TERB_NET_GRAD_CLIP_NORM:-1.0}"
TERB_NET_EARLY_STOP_PATIENCE="${TERB_NET_EARLY_STOP_PATIENCE:-0}"
TERB_NET_EARLY_STOP_MIN_DELTA="${TERB_NET_EARLY_STOP_MIN_DELTA:-0.0}"
TERB_NET_AMP="${TERB_NET_AMP:-true}"
TERB_NET_SAVE_VIS_EVERY_EPOCHS="${TERB_NET_SAVE_VIS_EVERY_EPOCHS:-1}"
TERB_NET_NUM_VIS_SAMPLES="${TERB_NET_NUM_VIS_SAMPLES:-4}"
TERB_NET_RESUME="${TERB_NET_RESUME:-auto}"

if [[ ! -f "${TERB_NET_CFG}" ]]; then
  echo "[ERR] TERB_NET_CFG not found: ${TERB_NET_CFG}"
  exit 1
fi

mkdir -p "${TERB_NET_LOG_DIR}/states" "${TERB_NET_CKPT_DIR}"

if [[ "${TERB_NET_RESUME}" == "auto" ]]; then
  AUTO_RESUME="${TERB_NET_LOG_DIR}/states/last.pth"
  if [[ -f "${AUTO_RESUME}" ]]; then
    TERB_NET_RESUME="${AUTO_RESUME}"
    echo "[TeR-B Net] Auto resume ckpt=${TERB_NET_RESUME}"
  else
    echo "[TeR-B Net] Auto resume requested but not found: ${AUTO_RESUME}"
    TERB_NET_RESUME=""
  fi
fi

if [[ -n "${TERB_NET_RESUME}" ]] && [[ ! -f "${TERB_NET_RESUME}" ]]; then
  echo "[ERR] TERB_NET_RESUME not found: ${TERB_NET_RESUME}"
  exit 1
fi

CMD=(
  python scripts/train_terb_net.py
  --config "${TERB_NET_CFG}"
  --run-dir "${TERB_NET_LOG_DIR}"
  --ckpt-dir "${TERB_NET_CKPT_DIR}"
  --epochs "${TERB_NET_EPOCHS}"
  --max-steps-per-epoch "${TERB_NET_MAX_STEPS_PER_EPOCH}"
  --val-every-epochs "${TERB_NET_VAL_EVERY_EPOCHS}"
  --save-every-epochs "${TERB_NET_SAVE_EVERY_EPOCHS}"
  --log-every-steps "${TERB_NET_LOG_EVERY_STEPS}"
  --seed "${TERB_NET_SEED}"
  --batch-size "${TERB_NET_BATCH_SIZE}"
  --num-workers "${TERB_NET_NUM_WORKERS}"
  --lr "${TERB_NET_LR}"
  --weight-decay "${TERB_NET_WEIGHT_DECAY}"
  --grad-clip-norm "${TERB_NET_GRAD_CLIP_NORM}"
  --early-stop-patience "${TERB_NET_EARLY_STOP_PATIENCE}"
  --early-stop-min-delta "${TERB_NET_EARLY_STOP_MIN_DELTA}"
  --amp "${TERB_NET_AMP}"
  --save-vis-every-epochs "${TERB_NET_SAVE_VIS_EVERY_EPOCHS}"
  --num-vis-samples "${TERB_NET_NUM_VIS_SAMPLES}"
)

if [[ -n "${TERB_NET_DATASETS_FOLDER}" ]]; then
  CMD+=(--datasets-folder "${TERB_NET_DATASETS_FOLDER}")
fi
if [[ -n "${TERB_NET_RESUME}" ]]; then
  CMD+=(--resume "${TERB_NET_RESUME}")
fi

echo "[TeR-B Net] CFG=${TERB_NET_CFG}"
echo "[TeR-B Net] LOG_DIR=${TERB_NET_LOG_DIR}"
echo "[TeR-B Net] CKPT_DIR=${TERB_NET_CKPT_DIR}"
echo "[TeR-B Net] EPOCHS=${TERB_NET_EPOCHS}, MAX_STEPS_PER_EPOCH=${TERB_NET_MAX_STEPS_PER_EPOCH}, VAL_EVERY_EPOCHS=${TERB_NET_VAL_EVERY_EPOCHS}"
echo "[TeR-B Net] BATCH_SIZE=${TERB_NET_BATCH_SIZE}, NUM_WORKERS=${TERB_NET_NUM_WORKERS}, DATASETS_FOLDER=${TERB_NET_DATASETS_FOLDER:-<cfg>}"
echo "[TeR-B Net] LR=${TERB_NET_LR}, WEIGHT_DECAY=${TERB_NET_WEIGHT_DECAY}, AMP=${TERB_NET_AMP}"
echo "[TeR-B Net] SAVE_VIS_EVERY_EPOCHS=${TERB_NET_SAVE_VIS_EVERY_EPOCHS}, NUM_VIS_SAMPLES=${TERB_NET_NUM_VIS_SAMPLES}"
echo "[TeR-B Net] EARLY_STOP_PATIENCE=${TERB_NET_EARLY_STOP_PATIENCE}, EARLY_STOP_MIN_DELTA=${TERB_NET_EARLY_STOP_MIN_DELTA}"
if [[ -n "${TERB_NET_RESUME}" ]]; then
  echo "[TeR-B Net] RESUME=${TERB_NET_RESUME}"
fi

"${CMD[@]}"
