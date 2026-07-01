#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

DATASET_NAME="${DATASET_NAME:-AVIID}"
ROUTE="${ROUTE:-psl_flow}"
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET_NAME="$2"
      shift 2
      ;;
    --route)
      ROUTE="$2"
      shift 2
      ;;
    *)
      echo "[ERR] Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "${DATASET_NAME}" in
  AVIID|CART|DroneVehicle_day|DroneVehicle_night) ;;
  *)
    echo "[ERR] Unsupported dataset: ${DATASET_NAME}" >&2
    exit 2
    ;;
esac

case "${ROUTE}" in
  psl_flow|klvae_sit) ;;
  *)
    echo "[ERR] Unsupported route: ${ROUTE}" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_REQUESTED:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export DATASETS_PREPROCESS_ROOT="${DATASETS_PREPROCESS_ROOT:-datasets_preprocess}"

NVIDIA_SMI_GPU_ID="${NVIDIA_SMI_GPU_ID:-1}"
GPU_POLL_INTERVAL_SECONDS="${GPU_POLL_INTERVAL_SECONDS:-1}"
RUN_ID_WAS_SET="${RUN_ID+x}"
BENCH_ROOT_WAS_SET="${BENCH_ROOT+x}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BENCH_ROOT="${BENCH_ROOT:-logs/experiments/${DATASET_NAME}/${ROUTE}/${RUN_ID}}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${BENCH_ROOT}/artifacts}"
SUMMARY_CSV="${SUMMARY_CSV:-${BENCH_ROOT}/summary.csv}"
RUN_LOG="${RUN_LOG:-${BENCH_ROOT}/run.log}"
AUTO_RESUME_LATEST="${AUTO_RESUME_LATEST:-1}"
RESUME_EXISTING_BENCH_ROOT="${RESUME_EXISTING_BENCH_ROOT:-1}"
ALLOW_EXISTING_BENCH_ROOT="${ALLOW_EXISTING_BENCH_ROOT:-0}"
RERUN_VALIDATION="${RERUN_VALIDATION:-0}"
RERUN_NORMALIZER="${RERUN_NORMALIZER:-0}"

PL_DEVICES_DEFAULT="${PL_DEVICES_DEFAULT:-1}"
PL_STRATEGY_DEFAULT="${PL_STRATEGY_DEFAULT:-auto}"
PL_ACCELERATOR_DEFAULT="${PL_ACCELERATOR_DEFAULT:-gpu}"
TRAIN_BATCH_SIZE_DEFAULT="${TRAIN_BATCH_SIZE_DEFAULT:-16}"
TEST_BATCH_SIZE_DEFAULT="${TEST_BATCH_SIZE_DEFAULT:-16}"
NUM_WORKERS_DEFAULT="${NUM_WORKERS_DEFAULT:-8}"
TRAIN_WITH_VALIDATION="${TRAIN_WITH_VALIDATION:-1}"
TERB_NUM_EPOCHS="${TERB_NUM_EPOCHS:-200}"
PSLVAE_NUM_EPOCHS="${PSLVAE_NUM_EPOCHS:-300}"
TERB_VAL_EVERY_EPOCHS="${TERB_VAL_EVERY_EPOCHS:-20}"
PSLVAE_VAL_EVERY_EPOCHS="${PSLVAE_VAL_EVERY_EPOCHS:-50}"
FLOW_VAL_EVERY_STEPS_WAS_SET="${FLOW_VAL_EVERY_STEPS+x}"
SIT_CHECKPOINT_EVERY_STEPS_WAS_SET="${SIT_CHECKPOINT_EVERY_STEPS+x}"
FLOW_VAL_EVERY_STEPS="${FLOW_VAL_EVERY_STEPS:-5000}"
TERB_NUM_SAMPLES_PER_EPOCH="${TERB_NUM_SAMPLES_PER_EPOCH:-${NUM_SAMPLES_PER_EPOCH:-auto}}"
PSLVAE_NUM_SAMPLES_PER_EPOCH="${PSLVAE_NUM_SAMPLES_PER_EPOCH:-${NUM_SAMPLES_PER_EPOCH:-auto}}"
FLOW_NUM_SAMPLES_PER_EPOCH="${FLOW_NUM_SAMPLES_PER_EPOCH:-${NUM_SAMPLES_PER_EPOCH:-auto}}"
MIXED_PRECISION="${MIXED_PRECISION:-false}"
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-1.0}"
RUN_SEED="${RUN_SEED:-1234}"
CUDA_TF32="${CUDA_TF32:-true}"
FLOAT32_MATMUL_PRECISION="${FLOAT32_MATMUL_PRECISION:-high}"
NUM_SAMPLE_IMAGES="${NUM_SAMPLE_IMAGES:-4}"
NUM_SAMPLE_BATCHES="${NUM_SAMPLE_BATCHES:-1}"
PSLVAE_SELECT="${PSLVAE_SELECT:-best_lpips}"
PSLVAE_EPOCH="${PSLVAE_EPOCH:-}"
PSLVAE_CHECKPOINT_MONITOR="${PSLVAE_CHECKPOINT_MONITOR:-val/LPIPS}"
FLOW_CHECKPOINT_MONITOR="${FLOW_CHECKPOINT_MONITOR:-val/LPIPS}"
FLOW_SELECT="${FLOW_SELECT:-step2}"
EVAL_FID="${EVAL_FID:-0}"
FINAL_EVAL_FID="${FINAL_EVAL_FID:-1}"
EVAL_EFFICIENCY="${EVAL_EFFICIENCY:-0}"
FINAL_EVAL_EFFICIENCY="${FINAL_EVAL_EFFICIENCY:-1}"
NORMALIZER_MAX_SAMPLES="${NORMALIZER_MAX_SAMPLES:-auto}"
NORMALIZER_DEVICE="${NORMALIZER_DEVICE:-auto}"
NORMALIZER_LATENT_SAMPLE_MODE="${NORMALIZER_LATENT_SAMPLE_MODE:-sample}"
NORMALIZER_SEED="${NORMALIZER_SEED:-1234}"

TERB_MAX_STEPS="${TERB_MAX_STEPS:-}"
PSLVAE_MAX_STEPS="${PSLVAE_MAX_STEPS:-}"
case "${DATASET_NAME}:${ROUTE}" in
  AVIID:psl_flow) DEFAULT_SIT_STEP_1=45000; DEFAULT_SIT_STEP_2=75000 ;;
  CART:psl_flow) DEFAULT_SIT_STEP_1=35000; DEFAULT_SIT_STEP_2=65000 ;;
  DroneVehicle_day:psl_flow) DEFAULT_SIT_STEP_1=70000; DEFAULT_SIT_STEP_2=100000 ;;
  DroneVehicle_night:psl_flow) DEFAULT_SIT_STEP_1=90000; DEFAULT_SIT_STEP_2=120000 ;;
  AVIID:klvae_sit) DEFAULT_SIT_STEP_1=45000; DEFAULT_SIT_STEP_2=45000 ;;
  CART:klvae_sit) DEFAULT_SIT_STEP_1=35000; DEFAULT_SIT_STEP_2=35000 ;;
  DroneVehicle_day:klvae_sit) DEFAULT_SIT_STEP_1=70000; DEFAULT_SIT_STEP_2=70000 ;;
  DroneVehicle_night:klvae_sit) DEFAULT_SIT_STEP_1=90000; DEFAULT_SIT_STEP_2=90000 ;;
esac
SIT_STEP_UNIT="${SIT_STEP_UNIT:-steps}"
SIT_STEP_1_RAW="${SIT_STEP_1:-${DEFAULT_SIT_STEP_1}}"
SIT_STEP_2_RAW="${SIT_STEP_2:-${DEFAULT_SIT_STEP_2}}"
SIT_SAMPLE_1="${SIT_SAMPLE_1:-auto}"
SIT_SAMPLE_2="${SIT_SAMPLE_2:-auto}"
SIT_EPOCH_1="${SIT_EPOCH_1:-1}"
SIT_EPOCH_2="${SIT_EPOCH_2:-${SIT_EPOCH_1}}"
SIT_EVAL_SPLIT="${SIT_EVAL_SPLIT:-val}"

RGB_VAE_PATH="${RGB_VAE_PATH:-}"
RGB_VAE_REPO="${RGB_VAE_REPO:-stabilityai/sd-vae-ft-ema}"
RGB_VAE_LOCAL_FILES_ONLY="${RGB_VAE_LOCAL_FILES_ONLY:-false}"
THERMAL_KLVAE_CKPT="${THERMAL_KLVAE_CKPT:-}"
THERMAL_KLVAE_NORMALIZER="${THERMAL_KLVAE_NORMALIZER:-1.0}"

dataset_required_splits() {
  case "${DATASET_NAME}" in
    AVIID)
      printf "%s\n" "AVIID/train" "AVIID/test"
      ;;
    CART)
      printf "%s\n" "CART/train" "CART/val" "CART/test"
      ;;
    DroneVehicle_day)
      printf "%s\n" "DroneVehicle/train/day" "DroneVehicle/test/day"
      ;;
    DroneVehicle_night)
      printf "%s\n" "DroneVehicle/train/night" "DroneVehicle/test/night"
      ;;
  esac
}

assert_dataset_preprocessed() {
  local missing=0 rel split_dir
  while IFS= read -r rel; do
    split_dir="${DATASETS_PREPROCESS_ROOT%/}/${rel}"
    if [[ ! -f "${split_dir}/metadata.json" ]] || ! compgen -G "${split_dir}/dataset-*.tar" >/dev/null; then
      echo "[ERR] Missing preprocessed WebDataset split: ${split_dir}" >&2
      missing=1
    fi
  done < <(dataset_required_splits)
  if [[ "${missing}" -ne 0 ]]; then
    echo "[ERR] Preprocess the raw images before training, for example:" >&2
    echo "  bash scripts/preprocess_dataset.sh --dataset ${DATASET_NAME} --raw-root datasets_raw --output-root ${DATASETS_PREPROCESS_ROOT} --overwrite" >&2
    exit 1
  fi
}

assert_dataset_preprocessed

dataset_train_sample_count() {
  python - "${DATASET_NAME}" "${DATASETS_PREPROCESS_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

dataset, preprocess_root = sys.argv[1:]
cfg_path = Path("psl_flow/configs/experiments/datasets") / f"{dataset}.yml"
with cfg_path.open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
total = 0
for item in cfg.get("train", []):
    metadata = Path(preprocess_root) / item["datafolder_name"] / "metadata.json"
    with metadata.open("r", encoding="utf-8") as handle:
        total += int(json.load(handle)["num_samples"])
print(total)
PY
}

resolve_samples_per_epoch() {
  local value="$1" label="$2" resolved
  if [[ "${value}" == "auto" ]]; then
    resolved="$(dataset_train_sample_count)"
    log_msg "Resolved ${label} num_samples_per_epoch=auto -> ${resolved}" >&2
    printf "%s" "${resolved}"
  else
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
      log_msg "ERR ${label} num_samples_per_epoch must be a positive integer or auto: ${value}" >&2
      exit 1
    fi
    printf "%s" "${value}"
  fi
}

ceil_div() {
  local numerator="$1"
  local denominator="$2"
  if (( denominator <= 0 )); then
    log_msg "ERR denominator must be positive for ceil_div: ${denominator}" >&2
    exit 1
  fi
  printf "%s" "$(((numerator + denominator - 1) / denominator))"
}

resolve_sit_steps() {
  local target="$1" label="$2" resolved_samples resolved_steps
  case "${SIT_STEP_UNIT}" in
    steps)
      if [[ ! "${target}" =~ ^[0-9]+$ ]] || (( target <= 0 )); then
        log_msg "ERR ${label} must be a positive integer when SIT_STEP_UNIT=steps: ${target}" >&2
        exit 1
      fi
      printf "%s" "${target}"
      ;;
    samples)
      if [[ "${target}" == "auto" ]]; then
        resolved_samples="${FLOW_NUM_SAMPLES_PER_EPOCH}"
      else
        if [[ ! "${target}" =~ ^[0-9]+$ ]] || (( target <= 0 )); then
          log_msg "ERR ${label} sample target must be a positive integer or auto: ${target}" >&2
          exit 1
        fi
        resolved_samples="${target}"
      fi
      resolved_steps="$(ceil_div "${resolved_samples}" "${TRAIN_BATCH_SIZE_DEFAULT}")"
      log_msg "Resolved ${label}: SIT_STEP_UNIT=samples, samples=${resolved_samples}, batch=${TRAIN_BATCH_SIZE_DEFAULT} -> steps=${resolved_steps}" >&2
      printf "%s" "${resolved_steps}"
      ;;
    epochs)
      if [[ ! "${target}" =~ ^[0-9]+$ ]] || (( target <= 0 )); then
        log_msg "ERR ${label} epoch target must be a positive integer: ${target}" >&2
        exit 1
      fi
      resolved_samples="$((FLOW_NUM_SAMPLES_PER_EPOCH * target))"
      resolved_steps="$(ceil_div "${resolved_samples}" "${TRAIN_BATCH_SIZE_DEFAULT}")"
      log_msg "Resolved ${label}: SIT_STEP_UNIT=epochs, epochs=${target}, samples_per_epoch=${FLOW_NUM_SAMPLES_PER_EPOCH}, batch=${TRAIN_BATCH_SIZE_DEFAULT} -> steps=${resolved_steps}" >&2
      printf "%s" "${resolved_steps}"
      ;;
    *)
      log_msg "ERR unsupported SIT_STEP_UNIT=${SIT_STEP_UNIT}; expected steps, samples, or epochs" >&2
      exit 1
      ;;
  esac
}

is_truthy() {
  [[ "${1:-}" =~ ^(1|true|TRUE|yes|YES|y|Y|on|ON)$ ]]
}

json_metrics_ready() {
  local metrics_json="$1"
  [[ -f "${metrics_json}" ]] || return 1
  python - "${metrics_json}" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert isinstance(payload, list) and payload and isinstance(payload[0], dict)
except Exception:
    raise SystemExit(1)
PY
}

if [[ -z "${RUN_ID_WAS_SET}" && -z "${BENCH_ROOT_WAS_SET}" ]] && is_truthy "${AUTO_RESUME_LATEST}"; then
  latest_root="$(find "logs/experiments/${DATASET_NAME}/${ROUTE}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
  latest_final_metrics="${latest_root}/artifacts/metrics/${ROUTE}_final_${SIT_EVAL_SPLIT}_${DATASET_NAME}.json"
  if [[ -n "${latest_root}" && -e "${latest_root}" ]] && ! json_metrics_ready "${latest_final_metrics}"; then
    BENCH_ROOT="${latest_root}"
    RUN_ID="$(basename "${BENCH_ROOT}")"
    ARTIFACT_ROOT="${BENCH_ROOT}/artifacts"
    SUMMARY_CSV="${BENCH_ROOT}/summary.csv"
    RUN_LOG="${BENCH_ROOT}/run.log"
    RESUME_EXISTING_BENCH_ROOT=1
  fi
fi

if [[ -e "${BENCH_ROOT}" && "${ALLOW_EXISTING_BENCH_ROOT}" != "1" && "${RESUME_EXISTING_BENCH_ROOT}" != "1" ]]; then
  echo "[ERR] BENCH_ROOT already exists: ${BENCH_ROOT}" >&2
  exit 1
fi

mkdir -p "${BENCH_ROOT}/logs" "${BENCH_ROOT}/tmp" "${ARTIFACT_ROOT}/configs" "${ARTIFACT_ROOT}/metrics"
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  printf "route,stage,step,elapsed_seconds,elapsed_hms,peak_gpu_mib,peak_gpu_gib,log_file,checkpoint\n" > "${SUMMARY_CSV}"
fi

log_msg() {
  printf "[%s] %s\n" "$(date '+%F %T')" "$*" | tee -a "${RUN_LOG}"
}

TERB_NUM_SAMPLES_PER_EPOCH="$(resolve_samples_per_epoch "${TERB_NUM_SAMPLES_PER_EPOCH}" "TeR-B")"
PSLVAE_NUM_SAMPLES_PER_EPOCH="$(resolve_samples_per_epoch "${PSLVAE_NUM_SAMPLES_PER_EPOCH}" "PSL-VAE")"
FLOW_NUM_SAMPLES_PER_EPOCH="$(resolve_samples_per_epoch "${FLOW_NUM_SAMPLES_PER_EPOCH}" "${ROUTE}")"
case "${SIT_STEP_UNIT}" in
  steps)
    SIT_STEP_1="$(resolve_sit_steps "${SIT_STEP_1_RAW}" "SiT step1")"
    SIT_STEP_2="$(resolve_sit_steps "${SIT_STEP_2_RAW}" "SiT step2")"
    ;;
  samples)
    SIT_STEP_1="$(resolve_sit_steps "${SIT_SAMPLE_1}" "SiT step1")"
    SIT_STEP_2="$(resolve_sit_steps "${SIT_SAMPLE_2}" "SiT step2")"
    ;;
  epochs)
    SIT_STEP_1="$(resolve_sit_steps "${SIT_EPOCH_1}" "SiT step1")"
    SIT_STEP_2="$(resolve_sit_steps "${SIT_EPOCH_2}" "SiT step2")"
    ;;
  *)
    log_msg "ERR unsupported SIT_STEP_UNIT=${SIT_STEP_UNIT}; expected steps, samples, or epochs"
    exit 1
    ;;
esac
if [[ -z "${FLOW_VAL_EVERY_STEPS_WAS_SET}" && "${SIT_STEP_UNIT}" != "steps" ]]; then
  FLOW_VAL_EVERY_STEPS="${SIT_STEP_1}"
  log_msg "Resolved FLOW_VAL_EVERY_STEPS=${FLOW_VAL_EVERY_STEPS} from ${SIT_STEP_UNIT} target"
fi
if [[ -z "${SIT_CHECKPOINT_EVERY_STEPS_WAS_SET}" ]]; then
  SIT_CHECKPOINT_EVERY_STEPS="${FLOW_VAL_EVERY_STEPS}"
fi
log_msg "Resolved SiT targets: unit=${SIT_STEP_UNIT}, step1=${SIT_STEP_1}, step2=${SIT_STEP_2}, val_every_steps=${FLOW_VAL_EVERY_STEPS}, checkpoint_every_steps=${SIT_CHECKPOINT_EVERY_STEPS}"
if [[ "${NORMALIZER_MAX_SAMPLES}" == "auto" ]]; then
  NORMALIZER_MAX_SAMPLES="${PSLVAE_NUM_SAMPLES_PER_EPOCH}"
fi

hms() {
  local total="${1:-0}"
  printf "%02d:%02d:%02d" "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

peak_gib() {
  awk -v mib="${1:-0}" 'BEGIN { printf "%.3f", mib / 1024.0 }'
}

max_int() {
  local lhs="${1:-0}"
  local rhs="${2:-0}"
  if (( rhs > lhs )); then printf "%s" "${rhs}"; else printf "%s" "${lhs}"; fi
}

append_summary() {
  local route="$1" stage="$2" step="$3" elapsed="$4" peak_mib="$5" log_file="$6" checkpoint="$7"
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${route}" "${stage}" "${step}" "${elapsed}" "$(hms "${elapsed}")" \
    "${peak_mib}" "$(peak_gib "${peak_mib}")" "${log_file}" "${checkpoint}" >> "${SUMMARY_CSV}"
}

monitor_gpu_peak() {
  local peak_file="$1" stop_file="$2" gpu_id="$3" mem prev
  while [[ ! -f "${stop_file}" ]]; do
    mem="$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1 { gsub(/ /, ""); print $1 }')"
    if [[ "${mem}" =~ ^[0-9]+$ ]]; then
      prev="$(cat "${peak_file}" 2>/dev/null || printf "0")"
      if (( mem > prev )); then printf "%s" "${mem}" > "${peak_file}"; fi
    fi
    sleep "${GPU_POLL_INTERVAL_SECONDS}"
  done
}

run_stage() {
  local route="$1" stage="$2" step="$3" checkpoint="$4"
  shift 4
  local safe_stage="${route}_${stage//[^A-Za-z0-9_]/_}"
  local log_file="${BENCH_ROOT}/logs/${safe_stage}.log"
  local peak_file="${BENCH_ROOT}/tmp/${safe_stage}.peak"
  local stop_file="${BENCH_ROOT}/tmp/${safe_stage}.stop"
  local monitor_pid="" start_ts end_ts status
  rm -f "${stop_file}"
  printf "0" > "${peak_file}"
  log_msg "START route=${route} stage=${stage} step=${step:-NA}"
  log_msg "LOG ${log_file}"
  start_ts="$(date +%s)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    monitor_gpu_peak "${peak_file}" "${stop_file}" "${NVIDIA_SMI_GPU_ID}" &
    monitor_pid="$!"
  else
    log_msg "WARN nvidia-smi not found; peak GPU memory will be 0 MiB."
  fi
  set +e
  "$@" 2>&1 | tee "${log_file}"
  status="${PIPESTATUS[0]}"
  set -e
  touch "${stop_file}"
  if [[ -n "${monitor_pid}" ]]; then wait "${monitor_pid}" 2>/dev/null || true; fi
  end_ts="$(date +%s)"
  RESULT_ELAPSED_SECONDS="$((end_ts - start_ts))"
  RESULT_PEAK_GPU_MIB="$(cat "${peak_file}" 2>/dev/null || printf "0")"
  append_summary "${route}" "${stage}" "${step}" "${RESULT_ELAPSED_SECONDS}" "${RESULT_PEAK_GPU_MIB}" "${log_file}" "${checkpoint}"
  if [[ "${status}" -ne 0 ]]; then
    log_msg "FAILED route=${route} stage=${stage} status=${status}"
    exit "${status}"
  fi
  log_msg "END route=${route} stage=${stage} elapsed=$(hms "${RESULT_ELAPSED_SECONDS}") peak_gpu=$(peak_gib "${RESULT_PEAK_GPU_MIB}")GiB"
}

assert_file() {
  local path="$1" label="$2"
  if [[ ! -f "${path}" ]]; then
    log_msg "ERR missing ${label}: ${path}"
    exit 1
  fi
}

checkpoint_global_step() {
  local ckpt="$1"
  python - "${ckpt}" <<'PY'
import sys
try:
    import torch
    payload = torch.load(sys.argv[1], map_location="cpu")
    print(int(payload.get("global_step", -1)) if isinstance(payload, dict) else -1)
except Exception:
    raise SystemExit(1)
PY
}

checkpoint_reaches_step() {
  local ckpt="$1" min_step="$2" step
  step="$(checkpoint_global_step "${ckpt}" 2>/dev/null || true)"
  [[ "${step}" =~ ^[0-9]+$ ]] && (( step >= min_step ))
}

copy_latest_last_ckpt() {
  local search_root="$1" dst="$2"
  shopt -s globstar nullglob
  local matches=("${search_root}"/**/last.ckpt)
  if [[ "${#matches[@]}" -eq 0 ]]; then matches=("${search_root}"/**/*.ckpt); fi
  if [[ "${#matches[@]}" -eq 0 ]]; then
    log_msg "ERR no checkpoint under ${search_root}"
    exit 1
  fi
  local latest
  latest="$(ls -1t "${matches[@]}" | head -n 1)"
  mkdir -p "$(dirname "${dst}")"
  cp "${latest}" "${dst}"
  log_msg "Copied latest checkpoint: ${latest} -> ${dst}"
}

latest_last_ckpt() {
  local search_root="$1"
  shopt -s globstar nullglob
  local matches=("${search_root}"/**/last.ckpt)
  if [[ "${#matches[@]}" -eq 0 ]]; then return 1; fi
  ls -1t "${matches[@]}" | head -n 1
}

copy_latest_best_ckpt() {
  local search_root="$1" dst="$2"
  shopt -s globstar nullglob
  local matches=("${search_root}"/**/best.ckpt)
  if [[ "${#matches[@]}" -eq 0 ]]; then return 1; fi
  local latest
  latest="$(ls -1t "${matches[@]}" | head -n 1)"
  mkdir -p "$(dirname "${dst}")"
  cp "${latest}" "${dst}"
  log_msg "Copied best checkpoint: ${latest} -> ${dst}"
}

select_best_or_last_ckpt() {
  local best="$1" last="$2" dst="$3" label="$4"
  local src=""
  if [[ -f "${best}" ]]; then src="${best}"; else src="${last}"; fi
  assert_file "${src}" "${label}"
  mkdir -p "$(dirname "${dst}")"
  cp "${src}" "${dst}"
  log_msg "Selected ${label}: ${src} -> ${dst}"
}

select_psl_vae_ckpt() {
  local run_root="$1" last="$2" best="$3" selected="$4" strategy="$5" epoch="$6"
  python - "${run_root}" "${last}" "${best}" "${selected}" "${strategy}" "${epoch}" <<'PY'
import csv
import glob
import os
import shutil
import sys
from pathlib import Path

run_root, last_ckpt, best_ckpt, selected_ckpt, strategy, epoch_arg = sys.argv[1:]
strategy = (strategy or "best_lpips").lower()
metric_by_strategy = {
    "best_fid": ("val/FID", "min"),
    "best_lpips": ("val/LPIPS", "min"),
    "best_psnr": ("val/PSNR", "max"),
    "best_ssim": ("val/SSIM", "max"),
}

def copy(src: str, reason: str) -> None:
    if not src or not os.path.isfile(src):
        raise SystemExit(f"missing selected PSL-VAE checkpoint for {reason}: {src}")
    os.makedirs(os.path.dirname(selected_ckpt), exist_ok=True)
    shutil.copy2(src, selected_ckpt)
    print(f"[select-psl-vae] {reason}: {src} -> {selected_ckpt}")

def fallback(reason: str) -> None:
    if os.path.isfile(best_ckpt):
        copy(best_ckpt, f"fallback_best_alias_after_{reason}")
    else:
        copy(last_ckpt, f"fallback_last_after_{reason}")

def checkpoint_epoch(path: str) -> int | None:
    try:
        import torch

        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict) and "epoch" in payload:
            return int(payload["epoch"])
    except Exception:
        return None
    return None

def find_epoch_ckpt(wanted: int) -> str | None:
    patterns = [
        f"**/epoch_{wanted:04d}.ckpt",
        f"**/epoch_{wanted - 1:04d}.ckpt",
        f"**/*epoch={wanted}*.ckpt",
        f"**/*epoch={wanted - 1}*.ckpt",
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(run_root, pattern), recursive=True), key=os.path.getmtime, reverse=True)
        if matches:
            return matches[0]
    candidates = sorted(glob.glob(os.path.join(run_root, "**", "*.ckpt"), recursive=True), key=os.path.getmtime, reverse=True)
    for candidate in candidates:
        epoch = checkpoint_epoch(candidate)
        if epoch in {wanted, wanted - 1}:
            return candidate
    return None

if strategy == "last":
    copy(last_ckpt, "last")
    raise SystemExit(0)

if strategy == "epoch":
    if not epoch_arg:
        raise SystemExit("PSLVAE_EPOCH is required when PSLVAE_SELECT=epoch")
    wanted = int(epoch_arg)
    match = find_epoch_ckpt(wanted)
    if match:
        copy(match, f"epoch_{wanted}")
    else:
        print(f"[select-psl-vae][WARN] no checkpoint found for epoch {wanted} under {run_root}; using fallback")
        fallback(f"missing_epoch_{wanted}")
    raise SystemExit(0)

if strategy not in metric_by_strategy:
    print(f"[select-psl-vae][WARN] unsupported strategy={strategy}; using fallback")
    fallback(f"unsupported_{strategy}")
    raise SystemExit(0)

metric_name, mode = metric_by_strategy[strategy]
metrics_path = Path(run_root) / "local_logs" / "metrics.csv"
if metrics_path.is_file():
    best_epoch = None
    best_value = None
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("stage") != "val" or row.get("key") != metric_name:
                continue
            value = float(row["value"])
            if best_value is None or (value < best_value if mode == "min" else value > best_value):
                best_value = value
                best_epoch = int(row["epoch"])
    if best_epoch is not None:
        for epoch in (best_epoch, best_epoch + 1, best_epoch - 1):
            match = find_epoch_ckpt(epoch)
            if match:
                copy(match, f"{strategy}:{metric_name}={best_value:.6g},metric_epoch={best_epoch},ckpt_epoch={epoch}")
                raise SystemExit(0)
        print(
            f"[select-psl-vae][WARN] metric {metric_name} best epoch {best_epoch} was found, "
            "but no matching epoch checkpoint exists; using fallback"
        )
        fallback(f"missing_metric_epoch_{best_epoch}")
        raise SystemExit(0)
else:
    print(f"[select-psl-vae][WARN] metrics file not found: {metrics_path}; using fallback")

if strategy == "best_fid":
    print("[select-psl-vae][WARN] PSLVAE_SELECT=best_fid requested but val/FID is unavailable; using fallback")
else:
    print(f"[select-psl-vae][WARN] metric {metric_name} unavailable for strategy={strategy}; using fallback")
fallback(f"missing_{metric_name.replace('/', '_')}")
PY
}

select_flow_ckpt() {
  local run_root="$1" step1="$2" step2="$3" selected="$4" strategy="$5"
  python - "${run_root}" "${step1}" "${step2}" "${selected}" "${strategy}" <<'PY'
import csv
import glob
import os
import shutil
import sys
from pathlib import Path

run_root, step1_ckpt, step2_ckpt, selected_ckpt, strategy = sys.argv[1:]
strategy = (strategy or "step2").lower()
best_ckpt = os.path.join(run_root, "checkpoints", "best.ckpt")
last_ckpt = os.path.join(run_root, "checkpoints", "last.ckpt")
metrics_path = Path(run_root) / "local_logs" / "metrics.csv"

metric_by_strategy = {
    "best": ("val/LPIPS", "min"),
    "best_lpips": ("val/LPIPS", "min"),
    "best_psnr": ("val/PSNR", "max"),
    "best_ssim": ("val/SSIM", "max"),
}

def copy(src: str, reason: str, metric_value: str = "") -> None:
    if not src or not os.path.isfile(src):
        raise SystemExit(f"missing selected Flow checkpoint for {reason}: {src}")
    os.makedirs(os.path.dirname(selected_ckpt), exist_ok=True)
    shutil.copy2(src, selected_ckpt)
    suffix = f" metric={metric_value}" if metric_value else ""
    print(f"[select-flow] strategy={strategy} source={reason}{suffix}: {src} -> {selected_ckpt}")

def fallback(reason: str) -> None:
    if os.path.isfile(best_ckpt):
        copy(best_ckpt, f"fallback_best_alias_after_{reason}")
    elif os.path.isfile(last_ckpt):
        copy(last_ckpt, f"fallback_last_after_{reason}")
    else:
        copy(step2_ckpt, f"fallback_step2_after_{reason}")

def ckpt_step(path: str) -> int | None:
    try:
        import torch

        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict) and "global_step" in payload:
            return int(payload["global_step"])
    except Exception:
        return None
    return None

def find_step_ckpt(step: int) -> str | None:
    patterns = [
        f"**/step_{step:06d}.ckpt",
        f"**/step_{step:07d}.ckpt",
        f"**/*step={step}*.ckpt",
        f"**/*step_{step}*.ckpt",
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(run_root, pattern), recursive=True), key=os.path.getmtime, reverse=True)
        if matches:
            return matches[0]
    candidates = sorted(glob.glob(os.path.join(run_root, "**", "*.ckpt"), recursive=True), key=os.path.getmtime, reverse=True)
    for candidate in candidates:
        if ckpt_step(candidate) == step:
            return candidate
    return None

if strategy == "step1":
    copy(step1_ckpt, "step1")
    raise SystemExit(0)
if strategy == "step2":
    copy(step2_ckpt, "step2")
    raise SystemExit(0)
if strategy == "last":
    copy(last_ckpt, "last")
    raise SystemExit(0)
if strategy == "best" and os.path.isfile(best_ckpt):
    copy(best_ckpt, "best_alias")
    raise SystemExit(0)

if strategy not in metric_by_strategy:
    print(f"[select-flow][WARN] unsupported FLOW_SELECT={strategy}; using fallback")
    fallback(f"unsupported_{strategy}")
    raise SystemExit(0)

metric_name, mode = metric_by_strategy[strategy]
if not metrics_path.is_file():
    print(f"[select-flow][WARN] metrics file not found: {metrics_path}; using fallback")
    fallback("missing_metrics_csv")
    raise SystemExit(0)

best_step = None
best_value = None
with metrics_path.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        if row.get("stage") != "val" or row.get("key") != metric_name:
            continue
        value = float(row["value"])
        if best_value is None or (value < best_value if mode == "min" else value > best_value):
            best_value = value
            best_step = int(row["global_step"])

if best_step is None:
    print(f"[select-flow][WARN] metric {metric_name} unavailable for FLOW_SELECT={strategy}; using fallback")
    if strategy == "best_lpips" and os.path.isfile(best_ckpt):
        copy(best_ckpt, "fallback_best_alias_after_missing_val_LPIPS")
        raise SystemExit(0)
    fallback(f"missing_{metric_name.replace('/', '_')}")
    raise SystemExit(0)

match = find_step_ckpt(best_step)
if match:
    copy(match, f"{strategy}:{metric_name}:global_step={best_step}", f"{best_value:.6g}")
else:
    print(
        f"[select-flow][WARN] metric {metric_name} best global_step {best_step} was found, "
        "but no matching checkpoint exists; using fallback"
    )
    fallback(f"missing_metric_step_{best_step}")
PY
}

copy_existing_ckpt_if_any() {
  local search_root="$1" dst="$2" label="$3" min_step="$4"
  if [[ -f "${dst}" ]] && checkpoint_reaches_step "${dst}" "${min_step}"; then
    log_msg "Reuse existing ${label}: ${dst}"
    return 0
  fi
  shopt -s globstar nullglob
  local matches=("${search_root}"/**/last.ckpt)
  if [[ "${#matches[@]}" -eq 0 ]]; then matches=("${search_root}"/**/*.ckpt); fi
  if [[ "${#matches[@]}" -eq 0 ]]; then return 1; fi
  local candidate latest=""
  for candidate in $(ls -1t "${matches[@]}"); do
    if checkpoint_reaches_step "${candidate}" "${min_step}"; then
      latest="${candidate}"
      break
    fi
  done
  if [[ -z "${latest}" ]]; then return 1; fi
  mkdir -p "$(dirname "${dst}")"
  cp "${latest}" "${dst}"
  log_msg "Resume with existing ${label}: ${latest} -> ${dst}"
  return 0
}

copy_existing_ckpt_at_or_after_step() {
  local search_root="$1" dst="$2" target_step="$3" label="$4"
  if [[ -f "${dst}" ]]; then
    if checkpoint_reaches_step "${dst}" "${target_step}"; then
      log_msg "Reuse existing ${label}: ${dst}"
      return 0
    fi
    log_msg "WARN existing ${label} does not reach step ${target_step}; removing stale alias: ${dst}"
    rm -f "${dst}"
  fi
  python - "${search_root}" "${target_step}" "${dst}" "${label}" <<'PY'
import glob
import os
import shutil
import sys

root, target, dst, label = sys.argv[1:]
target = int(target)
try:
    import torch
except Exception:
    raise SystemExit(1)

best = None
best_step = -1
for path in glob.glob(os.path.join(root, "**", "*.ckpt"), recursive=True):
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        continue
    step = int(payload.get("global_step", -1)) if isinstance(payload, dict) else -1
    if step >= target and step > best_step:
        best = path
        best_step = step
if not best:
    raise SystemExit(1)
os.makedirs(os.path.dirname(dst), exist_ok=True)
shutil.copy2(best, dst)
print(f"[resume] {label}: {best} (global_step={best_step}) -> {dst}")
PY
}

expected_fit_steps() {
  local cfg="$1" override_max_steps="$2"
  python - "${cfg}" "${override_max_steps}" <<'PY'
import math
import sys
import yaml
import os

cfg_path, override = sys.argv[1:]
with open(cfg_path, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
training = cfg.get("training", {})
if override:
    print(int(override))
    raise SystemExit(0)
max_steps = int(training.get("max_steps", -1))
if max_steps > 0:
    print(max_steps)
    raise SystemExit(0)
epochs = int(training.get("num_epochs", 1))
samples = int(training.get("num_samples_per_epoch", 0))
batch = max(1, int(training.get("train_batch_size", 1)))
print(int(math.ceil(samples / batch) * epochs))
PY
}

should_run_validation() {
  local metrics_json="$1"
  if is_truthy "${RERUN_VALIDATION}"; then return 0; fi
  ! json_metrics_ready "${metrics_json}"
}

flow_config_ready() {
  local cfg="$1"
  python - "${cfg}" <<'PY'
import sys
import yaml
try:
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    float(cfg.get("model", {}).get("model_config", {}).get("thermal_normalizer"))
except Exception:
    raise SystemExit(1)
PY
}

stats_json_ready() {
  local stats_json="$1"
  [[ -f "${stats_json}" ]] || return 1
  python - "${stats_json}" <<'PY'
import json
import sys
try:
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    float(payload.get("thermal_normalizer"))
except Exception:
    raise SystemExit(1)
PY
}

normalizer_cache_status() {
  local stats_json="$1" flow_cfg="$2" psl_vae_ckpt="$3" teacher_ckpt="$4" sample_mode="$5" seed="$6" max_samples="$7" train_samples="$8" train_batch_size="$9"
  python - "${stats_json}" "${flow_cfg}" "${psl_vae_ckpt}" "${teacher_ckpt}" "${sample_mode}" "${seed}" "${max_samples}" "${train_samples}" "${train_batch_size}" <<'PY'
import json
import os
import sys
from pathlib import Path

import yaml

(
    stats_json,
    flow_cfg,
    psl_vae_ckpt,
    teacher_ckpt,
    sample_mode,
    seed,
    max_samples,
    train_samples,
    train_batch_size,
) = sys.argv[1:]

def norm_path(value: str) -> str:
    if not value:
        return ""
    return os.path.normcase(os.path.abspath(os.path.normpath(value)))

def mismatch(reason: str) -> None:
    print(f"NORMALIZER_CACHE_MISMATCH {reason}")
    raise SystemExit(1)

if not os.path.isfile(stats_json):
    mismatch(f"missing_stats_json={stats_json}")
if not os.path.isfile(flow_cfg):
    mismatch(f"missing_flow_config={flow_cfg}")

with open(stats_json, "r", encoding="utf-8") as handle:
    stats = json.load(handle)
with open(flow_cfg, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
model_cfg = cfg.get("model", {}).get("model_config", {})

try:
    stats_normalizer = float(stats.get("thermal_normalizer"))
    flow_normalizer = float(model_cfg.get("thermal_normalizer"))
except Exception:
    mismatch("missing_or_invalid_thermal_normalizer")
if abs(stats_normalizer - flow_normalizer) > 1e-9:
    mismatch(f"thermal_normalizer: stats={stats_normalizer} flow_config={flow_normalizer}")

checks = [
    ("psl_vae_ckpt", norm_path(stats.get("psl_vae_ckpt", "")), norm_path(psl_vae_ckpt)),
    ("teacher_ckpt", norm_path(stats.get("teacher_ckpt", "")), norm_path(teacher_ckpt)),
    ("latent_sample_mode", str(stats.get("latent_sample_mode", "")), str(sample_mode)),
    ("seed", str(stats.get("seed", "")), str(int(seed))),
]
flow_checks = [
    ("flow.psl_vae_ckpt", norm_path(model_cfg.get("psl_vae_ckpt", "")), norm_path(psl_vae_ckpt)),
    ("flow.teacher.ckpt", norm_path(model_cfg.get("teacher", {}).get("ckpt", "")), norm_path(teacher_ckpt)),
    ("flow.normalizer_stats_json", norm_path(model_cfg.get("normalizer_stats_json", "")), norm_path(stats_json)),
]
for key, got, expected in checks + flow_checks:
    if got != expected:
        mismatch(f"{key}: got={got or '<empty>'} expected={expected or '<empty>'}")

def fingerprint(path: str) -> dict[str, int]:
    stat = os.stat(path)
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}

for key, path in (("psl_vae_ckpt", psl_vae_ckpt), ("teacher_ckpt", teacher_ckpt)):
    recorded = stats.get(f"{key}_fingerprint", {})
    current = fingerprint(path)
    for field in ("size", "mtime_ns"):
        if int(recorded.get(field, -1)) != int(current[field]):
            mismatch(f"{key}_fingerprint.{field}: got={recorded.get(field)} expected={current[field]}")

num_samples = int(stats.get("num_samples", -1))
max_samples = int(max_samples)
train_samples = int(train_samples)
train_batch_size = max(1, int(train_batch_size))
expected_cap = train_samples if max_samples <= 0 else min(train_samples, max_samples)
upper = train_samples if max_samples <= 0 else min(train_samples, expected_cap + train_batch_size - 1)
if num_samples < expected_cap or num_samples > upper:
    mismatch(
        "num_samples: "
        f"got={num_samples} expected_range=[{expected_cap},{upper}] "
        f"max_samples={max_samples} train_samples={train_samples}"
    )

print(
    "NORMALIZER_CACHE_REUSE "
    f"stats={stats_json} psl_vae_ckpt={psl_vae_ckpt} teacher_ckpt={teacher_ckpt} "
    f"mode={sample_mode} seed={seed} num_samples={num_samples}"
)
PY
}

patch_cfg_common() {
  local base="$1" out="$2" dataset="$3" teacher_ckpt="$4" psl_vae_ckpt="$5" samples_per_epoch="$6" route="$7"
  python - "${base}" "${out}" "${dataset}" "${teacher_ckpt}" "${psl_vae_ckpt}" "${RGB_VAE_PATH}" "${RGB_VAE_REPO}" "${RGB_VAE_LOCAL_FILES_ONLY}" "${TRAIN_BATCH_SIZE_DEFAULT}" "${TEST_BATCH_SIZE_DEFAULT}" "${NUM_WORKERS_DEFAULT}" "${samples_per_epoch}" "${MIXED_PRECISION}" "${route}" "${THERMAL_KLVAE_CKPT}" "${THERMAL_KLVAE_NORMALIZER}" <<'PY'
import sys
import yaml
import os

(
    base,
    out,
    dataset,
    teacher_ckpt,
    psl_vae_ckpt,
    rgb_path,
    rgb_repo,
    rgb_local_only,
    train_bs,
    test_bs,
    workers,
    samples,
    mixed,
    route,
    thermal_klvae_ckpt,
    thermal_klvae_normalizer,
) = sys.argv[1:]
with open(base, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

def ensure_dict(parent, key):
    value = parent.get(key)
    if not isinstance(value, dict):
        value = {}
        parent[key] = value
    return value

cfg["route"] = route if route in {"psl_flow", "klvae_sit"} else cfg.get("route")
datasets = ensure_dict(cfg, "datasets")
datasets["datasets_folder"] = os.environ.get("DATASETS_PREPROCESS_ROOT", datasets.get("datasets_folder", "./datasets_preprocess"))
datasets["train_datasets"] = [dataset]
datasets["val_datasets"] = [dataset]
datasets["test_datasets"] = [dataset]
datasets["target_val_dataset"] = dataset
training = ensure_dict(cfg, "training")
training["train_batch_size"] = int(train_bs)
training["test_batch_size"] = int(test_bs)
training["num_workers"] = int(workers)
training["num_samples_per_epoch"] = int(samples)
training["mixed_precision"] = str(mixed).lower() in {"1", "true", "yes", "y", "on"}
model_cfg = ensure_dict(ensure_dict(cfg, "model"), "model_config")
if teacher_ckpt:
    loss_cfg = ensure_dict(ensure_dict(training, "loss"), "config")
    ensure_dict(loss_cfg, "teacher")["ckpt"] = teacher_ckpt
    ensure_dict(model_cfg, "teacher")["ckpt"] = teacher_ckpt
if psl_vae_ckpt:
    model_cfg["psl_vae_ckpt"] = psl_vae_ckpt
uses_rgb_vae = bool(
    route == "klvae_sit"
    or "rgb_vae_config" in model_cfg
    or "rgb_vae_repo" in model_cfg
    or "rgb_vae_path" in model_cfg
)
if uses_rgb_vae:
    if rgb_path:
        model_cfg["rgb_vae_path"] = rgb_path
    if rgb_repo:
        model_cfg["rgb_vae_repo"] = rgb_repo
    model_cfg["rgb_vae_local_files_only"] = str(rgb_local_only).lower() in {"1", "true", "yes", "y", "on"}
if route == "klvae_sit":
    if thermal_klvae_ckpt:
        model_cfg["thermal_vae_ckpt"] = thermal_klvae_ckpt
    model_cfg["thermal_normalizer"] = float(thermal_klvae_normalizer)
with open(out, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY
}

patch_training_strategy() {
  local cfg="$1" epochs="$2" check_val_every="$3" val_check_interval="$4" checkpoint_monitor="$5" checkpoint_mode="$6" checkpoint_every_epochs="$7"
  python - "${cfg}" "${epochs}" "${check_val_every}" "${val_check_interval}" "${checkpoint_monitor}" "${checkpoint_mode}" "${checkpoint_every_epochs}" "${GRADIENT_CLIP_VAL}" "${RUN_SEED}" "${CUDA_TF32}" "${FLOAT32_MATMUL_PRECISION}" "${NUM_SAMPLE_IMAGES}" "${NUM_SAMPLE_BATCHES}" "${EVAL_FID}" "${EVAL_EFFICIENCY}" <<'PY'
import sys
import yaml

(
    path,
    epochs,
    check_val_every,
    val_check_interval,
    checkpoint_monitor,
    checkpoint_mode,
    checkpoint_every_epochs,
    gradient_clip,
    seed,
    cuda_tf32,
    matmul_precision,
    num_sample_images,
    num_sample_batches,
    eval_fid,
    eval_efficiency,
) = sys.argv[1:]
with open(path, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
training = cfg.setdefault("training", {})
if epochs:
    training["num_epochs"] = int(epochs)
if check_val_every:
    training["check_val_every_n_epoch"] = int(check_val_every)
if val_check_interval:
    training["val_check_interval"] = int(val_check_interval) if val_check_interval.isdigit() else float(val_check_interval)
else:
    training.pop("val_check_interval", None)
if checkpoint_monitor:
    training["checkpoint_monitor"] = checkpoint_monitor
if checkpoint_mode:
    training["checkpoint_mode"] = checkpoint_mode
if checkpoint_every_epochs:
    training["checkpoint_every_n_epochs"] = int(checkpoint_every_epochs)
training["gradient_clip_val"] = float(gradient_clip)
training["seed"] = int(seed)
training["cuda_tf32"] = str(cuda_tf32).lower() in {"1", "true", "yes", "y", "on"}
training["float32_matmul_precision"] = matmul_precision
training["export_samples"] = True
training["num_sample_images"] = int(num_sample_images)
training["num_sample_batches"] = int(num_sample_batches)
training["eval_fid"] = str(eval_fid).lower() in {"1", "true", "yes", "y", "on"}
training["eval_efficiency"] = str(eval_efficiency).lower() in {"1", "true", "yes", "y", "on"}
with open(path, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY
}

patch_final_eval_options() {
  local cfg="$1" fid_enabled="$2" efficiency_enabled="$3"
  python - "${cfg}" "${fid_enabled}" "${efficiency_enabled}" <<'PY'
import sys
import yaml

path, fid_enabled, efficiency_enabled = sys.argv[1:]
with open(path, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
training = cfg.setdefault("training", {})
training["eval_fid"] = str(fid_enabled).lower() in {"1", "true", "yes", "y", "on"}
training["eval_efficiency"] = str(efficiency_enabled).lower() in {"1", "true", "yes", "y", "on"}
with open(path, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY
}

log_msg "Repository: ${REPO_ROOT}"
log_msg "Dataset: ${DATASET_NAME}; route=${ROUTE}"
log_msg "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; nvidia-smi physical GPU id=${NVIDIA_SMI_GPU_ID}"
log_msg "Benchmark root: ${BENCH_ROOT}"
log_msg "RGB VAE condition encoder path: ${RGB_VAE_PATH}"

BASE_DIR="psl_flow/configs/experiments"
if [[ "${ROUTE}" == "psl_flow" ]]; then
  TERB_BASE_CFG="${BASE_DIR}/terb/${DATASET_NAME}.yml"
  PSLVAE_BASE_CFG="${BASE_DIR}/psl_vae/${DATASET_NAME}.yml"
  FLOW_BASE_CFG="${BASE_DIR}/psl_flow/${DATASET_NAME}.yml"
  PIPE_ROOT="${ARTIFACT_ROOT}/psl_flow"
  TERB_RUN="${PIPE_ROOT}/terb"
  PSLVAE_RUN="${PIPE_ROOT}/psl_vae"
  FLOW_RUN="${PIPE_ROOT}/sit_steps${SIT_STEP_2}"
  TERB_LAST_CKPT="${PIPE_ROOT}/checkpoints/terb_last.ckpt"
  TERB_BEST_CKPT="${PIPE_ROOT}/checkpoints/terb_best.ckpt"
  TERB_CKPT="${PIPE_ROOT}/checkpoints/terb_selected.ckpt"
  PSLVAE_LAST_CKPT="${PIPE_ROOT}/checkpoints/psl_vae_last.ckpt"
  PSLVAE_BEST_CKPT="${PIPE_ROOT}/checkpoints/psl_vae_best.ckpt"
  PSLVAE_CKPT="${PIPE_ROOT}/checkpoints/psl_vae_selected.ckpt"
  TERB_CFG="${ARTIFACT_ROOT}/configs/terb_${DATASET_NAME}.yml"
  PSLVAE_CFG="${ARTIFACT_ROOT}/configs/psl_vae_${DATASET_NAME}.yml"
  FLOW_CFG="${ARTIFACT_ROOT}/configs/psl_flow_${DATASET_NAME}.yml"
  FLOW_STATS_JSON="${ARTIFACT_ROOT}/configs/psl_flow_${DATASET_NAME}_latent_stats.json"
  TERB_VAL_METRICS_JSON="${ARTIFACT_ROOT}/metrics/terb_final_validation_${DATASET_NAME}.json"
  PSLVAE_VAL_METRICS_JSON="${ARTIFACT_ROOT}/metrics/psl_vae_final_validation_${DATASET_NAME}.json"

  patch_cfg_common "${TERB_BASE_CFG}" "${TERB_CFG}" "${DATASET_NAME}" "" "" "${TERB_NUM_SAMPLES_PER_EPOCH}" "psl_flow"
  patch_training_strategy "${TERB_CFG}" "${TERB_NUM_EPOCHS}" "${TERB_VAL_EVERY_EPOCHS}" "" "val/loss_total" "min" "${TERB_VAL_EVERY_EPOCHS}"
  TERB_EXPECTED_STEPS="$(expected_fit_steps "${TERB_CFG}" "${TERB_MAX_STEPS}")"
  if ! copy_existing_ckpt_if_any "${TERB_RUN}" "${TERB_LAST_CKPT}" "TeR-B last checkpoint" "${TERB_EXPECTED_STEPS}"; then
    TERB_STAGE="terb_train"
    TERB_VAL_ARGS=(--check-val-every-n-epoch "${TERB_VAL_EVERY_EPOCHS}")
    if ! is_truthy "${TRAIN_WITH_VALIDATION}"; then
      TERB_STAGE="terb_train_no_val"
      TERB_VAL_ARGS=(--limit-val-batches 0 --check-val-every-n-epoch 999999)
    fi
    TERB_RESUME_ARGS=()
    if TERB_RESUME_CKPT="$(latest_last_ckpt "${TERB_RUN}" 2>/dev/null)"; then
      TERB_RESUME_ARGS=(--resume-from "${TERB_RESUME_CKPT}")
      log_msg "Resume TeR-B from partial checkpoint: ${TERB_RESUME_CKPT}"
    fi
    TERB_CMD=(python -m psl_flow.training.train_terb --config "${TERB_CFG}" --default-root-dir "${TERB_RUN}" --devices "${PL_DEVICES_DEFAULT}" --accelerator "${PL_ACCELERATOR_DEFAULT}" --strategy "${PL_STRATEGY_DEFAULT}" "${TERB_RESUME_ARGS[@]}" "${TERB_VAL_ARGS[@]}")
    if [[ -n "${TERB_MAX_STEPS}" ]]; then TERB_CMD+=(--max-steps "${TERB_MAX_STEPS}"); fi
    run_stage "${ROUTE}" "${TERB_STAGE}" "" "${TERB_LAST_CKPT}" "${TERB_CMD[@]}"
    copy_latest_last_ckpt "${TERB_RUN}" "${TERB_LAST_CKPT}"
  fi
  copy_latest_best_ckpt "${TERB_RUN}" "${TERB_BEST_CKPT}" || true
  select_best_or_last_ckpt "${TERB_BEST_CKPT}" "${TERB_LAST_CKPT}" "${TERB_CKPT}" "TeR-B selected checkpoint"
  assert_file "${TERB_CKPT}" "TeR-B checkpoint"
  if should_run_validation "${TERB_VAL_METRICS_JSON}"; then
    run_stage "${ROUTE}" "terb_final_validation" "" "${TERB_CKPT}" \
      python -m psl_flow.training.train_terb --config "${TERB_CFG}" --mode validate \
        --ckpt "${TERB_CKPT}" --metrics-json "${TERB_VAL_METRICS_JSON}" \
        --default-root-dir "${TERB_RUN}" --devices "${PL_DEVICES_DEFAULT}" \
        --accelerator "${PL_ACCELERATOR_DEFAULT}" --strategy "${PL_STRATEGY_DEFAULT}"
  fi

  patch_cfg_common "${PSLVAE_BASE_CFG}" "${PSLVAE_CFG}" "${DATASET_NAME}" "${TERB_CKPT}" "" "${PSLVAE_NUM_SAMPLES_PER_EPOCH}" "psl_flow"
  patch_training_strategy "${PSLVAE_CFG}" "${PSLVAE_NUM_EPOCHS}" "${PSLVAE_VAL_EVERY_EPOCHS}" "" "${PSLVAE_CHECKPOINT_MONITOR}" "min" "${PSLVAE_VAL_EVERY_EPOCHS}"
  PSLVAE_EXPECTED_STEPS="$(expected_fit_steps "${PSLVAE_CFG}" "${PSLVAE_MAX_STEPS}")"
  if ! copy_existing_ckpt_if_any "${PSLVAE_RUN}" "${PSLVAE_LAST_CKPT}" "PSL-VAE last checkpoint" "${PSLVAE_EXPECTED_STEPS}"; then
    PSLVAE_STAGE="psl_vae_train"
    PSLVAE_VAL_ARGS=(--check-val-every-n-epoch "${PSLVAE_VAL_EVERY_EPOCHS}")
    if ! is_truthy "${TRAIN_WITH_VALIDATION}"; then
      PSLVAE_STAGE="psl_vae_train_no_val"
      PSLVAE_VAL_ARGS=(--limit-val-batches 0 --check-val-every-n-epoch 999999)
    fi
    PSLVAE_RESUME_ARGS=()
    if PSLVAE_RESUME_CKPT="$(latest_last_ckpt "${PSLVAE_RUN}" 2>/dev/null)"; then
      PSLVAE_RESUME_ARGS=(--resume-from "${PSLVAE_RESUME_CKPT}")
      log_msg "Resume PSL-VAE from partial checkpoint: ${PSLVAE_RESUME_CKPT}"
    fi
    PSLVAE_CMD=(python -m psl_flow.training.train_psl_vae --config "${PSLVAE_CFG}" --default-root-dir "${PSLVAE_RUN}" --devices "${PL_DEVICES_DEFAULT}" --accelerator "${PL_ACCELERATOR_DEFAULT}" --strategy "${PL_STRATEGY_DEFAULT}" "${PSLVAE_RESUME_ARGS[@]}" "${PSLVAE_VAL_ARGS[@]}")
    if [[ -n "${PSLVAE_MAX_STEPS}" ]]; then PSLVAE_CMD+=(--max-steps "${PSLVAE_MAX_STEPS}"); fi
    run_stage "${ROUTE}" "${PSLVAE_STAGE}" "" "${PSLVAE_LAST_CKPT}" "${PSLVAE_CMD[@]}"
    copy_latest_last_ckpt "${PSLVAE_RUN}" "${PSLVAE_LAST_CKPT}"
  fi
  copy_latest_best_ckpt "${PSLVAE_RUN}" "${PSLVAE_BEST_CKPT}" || true
  if is_truthy "${TRAIN_WITH_VALIDATION}"; then
    select_psl_vae_ckpt "${PSLVAE_RUN}" "${PSLVAE_LAST_CKPT}" "${PSLVAE_BEST_CKPT}" "${PSLVAE_CKPT}" "${PSLVAE_SELECT}" "${PSLVAE_EPOCH}"
  else
    select_psl_vae_ckpt "${PSLVAE_RUN}" "${PSLVAE_LAST_CKPT}" "${PSLVAE_BEST_CKPT}" "${PSLVAE_CKPT}" "last" ""
  fi
  assert_file "${PSLVAE_CKPT}" "PSL-VAE checkpoint"
  if should_run_validation "${PSLVAE_VAL_METRICS_JSON}"; then
    run_stage "${ROUTE}" "psl_vae_final_validation" "" "${PSLVAE_CKPT}" \
      python -m psl_flow.training.train_psl_vae --config "${PSLVAE_CFG}" --mode validate \
        --ckpt "${PSLVAE_CKPT}" --metrics-json "${PSLVAE_VAL_METRICS_JSON}" \
        --default-root-dir "${PSLVAE_RUN}" --devices "${PL_DEVICES_DEFAULT}" \
        --accelerator "${PL_ACCELERATOR_DEFAULT}" --strategy "${PL_STRATEGY_DEFAULT}"
  fi

  NORMALIZER_CACHE_OK=0
  NORMALIZER_CACHE_REPORT=""
  if ! is_truthy "${RERUN_NORMALIZER}"; then
    set +e
    NORMALIZER_CACHE_REPORT="$(normalizer_cache_status "${FLOW_STATS_JSON}" "${FLOW_CFG}" "${PSLVAE_CKPT}" "${TERB_CKPT}" "${NORMALIZER_LATENT_SAMPLE_MODE}" "${NORMALIZER_SEED}" "${NORMALIZER_MAX_SAMPLES}" "${PSLVAE_NUM_SAMPLES_PER_EPOCH}" "${TRAIN_BATCH_SIZE_DEFAULT}" 2>&1)"
    NORMALIZER_CACHE_OK="$?"
    set -e
    while IFS= read -r line; do
      [[ -n "${line}" ]] && log_msg "${line}"
    done <<< "${NORMALIZER_CACHE_REPORT}"
  else
    log_msg "NORMALIZER_CACHE_MISMATCH forced_by_RERUN_NORMALIZER=1"
  fi
  if [[ "${NORMALIZER_CACHE_OK}" -eq 0 ]]; then
    log_msg "Skip latent stats; checkpoint-aware cache is valid: ${FLOW_STATS_JSON}"
  else
    patch_cfg_common "${FLOW_BASE_CFG}" "${FLOW_CFG}" "${DATASET_NAME}" "${TERB_CKPT}" "${PSLVAE_CKPT}" "${FLOW_NUM_SAMPLES_PER_EPOCH}" "psl_flow"
    patch_training_strategy "${FLOW_CFG}" "" "0" "${FLOW_VAL_EVERY_STEPS}" "${FLOW_CHECKPOINT_MONITOR}" "min" ""
    run_stage "${ROUTE}" "psl_vae_latent_stats_patch_config" "" "${FLOW_STATS_JSON}" \
      python -m psl_flow.models.psl_vae.prepare_psl_flow_config \
        --flow-config "${FLOW_CFG}" --output-flow-config "${FLOW_CFG}" \
        --psl-vae-config "${PSLVAE_CFG}" --psl-vae-ckpt "${PSLVAE_CKPT}" \
        --teacher-ckpt "${TERB_CKPT}" --rgb-vae-path "${RGB_VAE_PATH}" \
        --rgb-vae-repo "${RGB_VAE_REPO}" --rgb-vae-local-files-only "${RGB_VAE_LOCAL_FILES_ONLY}" \
        --stats-json "${FLOW_STATS_JSON}" --device "${NORMALIZER_DEVICE}" \
        --max-samples "${NORMALIZER_MAX_SAMPLES}" --latent-sample-mode "${NORMALIZER_LATENT_SAMPLE_MODE}" \
        --seed "${NORMALIZER_SEED}"
  fi
  assert_file "${FLOW_STATS_JSON}" "PSL-VAE latent stats"
else
  FLOW_BASE_CFG="${BASE_DIR}/klvae_sit/${DATASET_NAME}.yml"
  PIPE_ROOT="${ARTIFACT_ROOT}/klvae_sit"
  FLOW_RUN="${PIPE_ROOT}/sit_steps${SIT_STEP_2}"
  FLOW_CFG="${ARTIFACT_ROOT}/configs/klvae_sit_${DATASET_NAME}.yml"
  if [[ -z "${THERMAL_KLVAE_CKPT}" ]]; then
    log_msg "ERR route=klvae_sit requires THERMAL_KLVAE_CKPT=/path/to/thermal_klvae.ckpt"
    exit 1
  fi
  patch_cfg_common "${FLOW_BASE_CFG}" "${FLOW_CFG}" "${DATASET_NAME}" "" "" "${FLOW_NUM_SAMPLES_PER_EPOCH}" "klvae_sit"
  patch_training_strategy "${FLOW_CFG}" "" "0" "${FLOW_VAL_EVERY_STEPS}" "${FLOW_CHECKPOINT_MONITOR}" "min" ""
fi

printf -v FLOW_STEP1 "%s/checkpoints/step_%06d.ckpt" "${FLOW_RUN}" "${SIT_STEP_1}"
printf -v FLOW_STEP2 "%s/checkpoints/step_%06d.ckpt" "${FLOW_RUN}" "${SIT_STEP_2}"
FLOW_SELECTED_CKPT="${PIPE_ROOT}/checkpoints/${ROUTE}_selected.ckpt"
FLOW_STEP1_METRICS_JSON="${ARTIFACT_ROOT}/metrics/${ROUTE}_step_${SIT_STEP_1}_${SIT_EVAL_SPLIT}_${DATASET_NAME}.json"
FLOW_VAL_METRICS_JSON="${ARTIFACT_ROOT}/metrics/${ROUTE}_final_${SIT_EVAL_SPLIT}_${DATASET_NAME}.json"

SIT_START_TS="$(date +%s)"
SIT_PEAK=0
if [[ -f "${FLOW_STEP2}" ]]; then
  log_msg "Skip SiT step ${SIT_STEP_1}; step ${SIT_STEP_2} checkpoint already exists: ${FLOW_STEP2}"
elif copy_existing_ckpt_at_or_after_step "${FLOW_RUN}" "${FLOW_STEP1}" "${SIT_STEP_1}" "${ROUTE} step ${SIT_STEP_1} checkpoint"; then
  log_msg "Reuse existing ${ROUTE} step ${SIT_STEP_1} checkpoint: ${FLOW_STEP1}"
else
  FLOW_STAGE1="sit_train_to_${SIT_STEP_1}"
  FLOW_VAL_ARGS=(--check-val-every-n-epoch 0 --val-check-interval "${FLOW_VAL_EVERY_STEPS}")
  if ! is_truthy "${TRAIN_WITH_VALIDATION}"; then
    FLOW_STAGE1="sit_train_to_${SIT_STEP_1}_no_val"
    FLOW_VAL_ARGS=(--limit-val-batches 0 --check-val-every-n-epoch 999999)
  fi
  FLOW_RESUME_ARGS=()
  if FLOW_RESUME_CKPT="$(latest_last_ckpt "${FLOW_RUN}" 2>/dev/null)"; then
    FLOW_RESUME_ARGS=(--resume-from "${FLOW_RESUME_CKPT}")
    log_msg "Resume ${ROUTE} toward step ${SIT_STEP_1} from partial checkpoint: ${FLOW_RESUME_CKPT}"
  fi
  run_stage "${ROUTE}" "${FLOW_STAGE1}" "${SIT_STEP_1}" "${FLOW_STEP1}" \
    python -m psl_flow.training.train_psl_flow --config "${FLOW_CFG}" --mode fit \
      --default-root-dir "${FLOW_RUN}" --devices "${PL_DEVICES_DEFAULT}" \
      --accelerator "${PL_ACCELERATOR_DEFAULT}" --strategy "${PL_STRATEGY_DEFAULT}" \
      "${FLOW_RESUME_ARGS[@]}" \
      --max-steps "${SIT_STEP_1}" --checkpoint-every-n-train-steps "${SIT_CHECKPOINT_EVERY_STEPS}" \
      "${FLOW_VAL_ARGS[@]}"
  copy_latest_last_ckpt "${FLOW_RUN}" "${FLOW_STEP1}"
  SIT_PEAK="${RESULT_PEAK_GPU_MIB}"
  SIT_CUM="$(( $(date +%s) - SIT_START_TS ))"
  append_summary "${ROUTE}" "sit_cumulative_to_${SIT_STEP_1}" "${SIT_STEP_1}" "${SIT_CUM}" "${SIT_PEAK}" "${BENCH_ROOT}/logs/${ROUTE}_${FLOW_STAGE1}.log" "${FLOW_STEP1}"
fi
if [[ ! -f "${FLOW_STEP2}" ]]; then
  assert_file "${FLOW_STEP1}" "${ROUTE} step ${SIT_STEP_1}"
fi

EVAL_MODE="test"
if [[ "${SIT_EVAL_SPLIT}" == "val" ]]; then
  EVAL_MODE="validate"
fi

if [[ -f "${FLOW_STEP1}" && "${SIT_STEP_1}" != "${SIT_STEP_2}" ]] && should_run_validation "${FLOW_STEP1_METRICS_JSON}"; then
  run_stage "${ROUTE}" "validation_${SIT_EVAL_SPLIT}_step_${SIT_STEP_1}" "${SIT_STEP_1}" "${FLOW_STEP1}" \
    python -m psl_flow.training.train_psl_flow --config "${FLOW_CFG}" --mode "${EVAL_MODE}" \
      --ckpt "${FLOW_STEP1}" --metrics-json "${FLOW_STEP1_METRICS_JSON}" \
      --default-root-dir "${FLOW_RUN}" --devices "${PL_DEVICES_DEFAULT}" \
      --accelerator "${PL_ACCELERATOR_DEFAULT}" --strategy "${PL_STRATEGY_DEFAULT}"
fi

if copy_existing_ckpt_at_or_after_step "${FLOW_RUN}" "${FLOW_STEP2}" "${SIT_STEP_2}" "${ROUTE} step ${SIT_STEP_2} checkpoint"; then
  log_msg "Reuse existing ${ROUTE} step ${SIT_STEP_2} checkpoint: ${FLOW_STEP2}"
else
  FLOW_STAGE2="sit_resume_to_${SIT_STEP_2}"
  FLOW_VAL_ARGS=(--check-val-every-n-epoch 0 --val-check-interval "${FLOW_VAL_EVERY_STEPS}")
  if ! is_truthy "${TRAIN_WITH_VALIDATION}"; then
    FLOW_STAGE2="sit_resume_to_${SIT_STEP_2}_no_val"
    FLOW_VAL_ARGS=(--limit-val-batches 0 --check-val-every-n-epoch 999999)
  fi
  FLOW_STAGE2_RESUME="${FLOW_STEP1}"
  if FLOW_RESUME_CKPT="$(latest_last_ckpt "${FLOW_RUN}" 2>/dev/null)" \
    && checkpoint_reaches_step "${FLOW_RESUME_CKPT}" "${SIT_STEP_1}" \
    && ! checkpoint_reaches_step "${FLOW_RESUME_CKPT}" "${SIT_STEP_2}"; then
    FLOW_STAGE2_RESUME="${FLOW_RESUME_CKPT}"
    log_msg "Resume ${ROUTE} toward step ${SIT_STEP_2} from partial checkpoint: ${FLOW_RESUME_CKPT}"
  fi
  run_stage "${ROUTE}" "${FLOW_STAGE2}" "${SIT_STEP_2}" "${FLOW_STEP2}" \
    python -m psl_flow.training.train_psl_flow --config "${FLOW_CFG}" --mode fit \
      --default-root-dir "${FLOW_RUN}" --devices "${PL_DEVICES_DEFAULT}" \
      --accelerator "${PL_ACCELERATOR_DEFAULT}" --strategy "${PL_STRATEGY_DEFAULT}" \
      --resume-from "${FLOW_STAGE2_RESUME}" --max-steps "${SIT_STEP_2}" \
      --checkpoint-every-n-train-steps "${SIT_CHECKPOINT_EVERY_STEPS}" \
      "${FLOW_VAL_ARGS[@]}"
  copy_latest_last_ckpt "${FLOW_RUN}" "${FLOW_STEP2}"
  SIT_PEAK="$(max_int "${SIT_PEAK}" "${RESULT_PEAK_GPU_MIB}")"
  SIT_CUM="$(( $(date +%s) - SIT_START_TS ))"
  append_summary "${ROUTE}" "sit_cumulative_to_${SIT_STEP_2}" "${SIT_STEP_2}" "${SIT_CUM}" "${SIT_PEAK}" "${BENCH_ROOT}/logs/${ROUTE}_${FLOW_STAGE2}.log" "${FLOW_STEP2}"
fi
assert_file "${FLOW_STEP2}" "${ROUTE} step ${SIT_STEP_2}"

FLOW_SELECT_REPORT="$(select_flow_ckpt "${FLOW_RUN}" "${FLOW_STEP1}" "${FLOW_STEP2}" "${FLOW_SELECTED_CKPT}" "${FLOW_SELECT}" 2>&1)"
while IFS= read -r line; do
  [[ -n "${line}" ]] && log_msg "${line}"
done <<< "${FLOW_SELECT_REPORT}"
assert_file "${FLOW_SELECTED_CKPT}" "${ROUTE} selected checkpoint"
append_summary "${ROUTE}" "flow_select_${FLOW_SELECT}" "" 0 0 "NA" "${FLOW_SELECTED_CKPT}"

if should_run_validation "${FLOW_VAL_METRICS_JSON}"; then
  patch_final_eval_options "${FLOW_CFG}" "${FINAL_EVAL_FID}" "${FINAL_EVAL_EFFICIENCY}"
  run_stage "${ROUTE}" "final_validation_${SIT_EVAL_SPLIT}" "" "${FLOW_SELECTED_CKPT}" \
    python -m psl_flow.training.train_psl_flow --config "${FLOW_CFG}" --mode "${EVAL_MODE}" \
      --ckpt "${FLOW_SELECTED_CKPT}" --metrics-json "${FLOW_VAL_METRICS_JSON}" \
      --default-root-dir "${FLOW_RUN}" --devices "${PL_DEVICES_DEFAULT}" \
      --accelerator "${PL_ACCELERATOR_DEFAULT}" --strategy "${PL_STRATEGY_DEFAULT}"
fi
assert_file "${FLOW_VAL_METRICS_JSON}" "${ROUTE} final metrics"

log_msg "Finished. Summary CSV: ${SUMMARY_CSV}"
for metrics_file in "${ARTIFACT_ROOT}"/metrics/*.json; do
  if [[ -f "${metrics_file}" ]]; then
    log_msg "METRICS ${metrics_file}"
    cat "${metrics_file}" | tee -a "${RUN_LOG}"
  fi
done
column -s, -t "${SUMMARY_CSV}" | tee -a "${RUN_LOG}" || cat "${SUMMARY_CSV}" | tee -a "${RUN_LOG}"
