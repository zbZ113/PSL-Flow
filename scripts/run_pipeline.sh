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
TERB_NUM_SAMPLES_PER_EPOCH="${TERB_NUM_SAMPLES_PER_EPOCH:-${NUM_SAMPLES_PER_EPOCH:-2412}}"
PSLVAE_NUM_SAMPLES_PER_EPOCH="${PSLVAE_NUM_SAMPLES_PER_EPOCH:-${NUM_SAMPLES_PER_EPOCH:-2412}}"
FLOW_NUM_SAMPLES_PER_EPOCH="${FLOW_NUM_SAMPLES_PER_EPOCH:-10000}"
MIXED_PRECISION="${MIXED_PRECISION:-false}"
NORMALIZER_MAX_SAMPLES="${NORMALIZER_MAX_SAMPLES:-${PSLVAE_NUM_SAMPLES_PER_EPOCH}}"
NORMALIZER_DEVICE="${NORMALIZER_DEVICE:-auto}"
NORMALIZER_LATENT_SAMPLE_MODE="${NORMALIZER_LATENT_SAMPLE_MODE:-sample}"
NORMALIZER_SEED="${NORMALIZER_SEED:-1234}"

TERB_MAX_STEPS="${TERB_MAX_STEPS:-}"
PSLVAE_MAX_STEPS="${PSLVAE_MAX_STEPS:-}"
SIT_STEP_1="${SIT_STEP_1:-45000}"
SIT_STEP_2="${SIT_STEP_2:-75000}"
SIT_EVAL_SPLIT="${SIT_EVAL_SPLIT:-val}"

RGB_VAE_PATH="${RGB_VAE_PATH:-/root/autodl-fs/sd-vae-ft-ema}"
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
    log_msg "Reuse existing ${label}: ${dst}"
    return 0
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
cfg["route"] = route if route in {"psl_flow", "klvae_sit"} else cfg.get("route")
datasets = cfg.setdefault("datasets", {})
datasets["datasets_folder"] = os.environ.get("DATASETS_PREPROCESS_ROOT", datasets.get("datasets_folder", "./datasets_preprocess"))
datasets["train_datasets"] = [dataset]
datasets["val_datasets"] = [dataset]
datasets["test_datasets"] = [dataset]
datasets["target_val_dataset"] = dataset
training = cfg.setdefault("training", {})
training["train_batch_size"] = int(train_bs)
training["test_batch_size"] = int(test_bs)
training["num_workers"] = int(workers)
training["num_samples_per_epoch"] = int(samples)
training["mixed_precision"] = str(mixed).lower() in {"1", "true", "yes", "y", "on"}
model_cfg = cfg.setdefault("model", {}).setdefault("model_config", {})
if teacher_ckpt:
    cfg.setdefault("training", {}).setdefault("loss", {}).setdefault("config", {}).setdefault("teacher", {})["ckpt"] = teacher_ckpt
    model_cfg.setdefault("teacher", {})["ckpt"] = teacher_ckpt
if psl_vae_ckpt:
    model_cfg["psl_vae_ckpt"] = psl_vae_ckpt
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
  TERB_CKPT="${PIPE_ROOT}/checkpoints/terb_last.ckpt"
  PSLVAE_CKPT="${PIPE_ROOT}/checkpoints/psl_vae_last.ckpt"
  TERB_CFG="${ARTIFACT_ROOT}/configs/terb_${DATASET_NAME}.yml"
  PSLVAE_CFG="${ARTIFACT_ROOT}/configs/psl_vae_${DATASET_NAME}.yml"
  FLOW_CFG="${ARTIFACT_ROOT}/configs/psl_flow_${DATASET_NAME}.yml"
  FLOW_STATS_JSON="${ARTIFACT_ROOT}/configs/psl_flow_${DATASET_NAME}_latent_stats.json"
  TERB_VAL_METRICS_JSON="${ARTIFACT_ROOT}/metrics/terb_final_validation_${DATASET_NAME}.json"
  PSLVAE_VAL_METRICS_JSON="${ARTIFACT_ROOT}/metrics/psl_vae_final_validation_${DATASET_NAME}.json"

  patch_cfg_common "${TERB_BASE_CFG}" "${TERB_CFG}" "${DATASET_NAME}" "" "" "${TERB_NUM_SAMPLES_PER_EPOCH}" "psl_flow"
  TERB_EXPECTED_STEPS="$(expected_fit_steps "${TERB_CFG}" "${TERB_MAX_STEPS}")"
  if ! copy_existing_ckpt_if_any "${TERB_RUN}" "${TERB_CKPT}" "TeR-B checkpoint" "${TERB_EXPECTED_STEPS}"; then
    TERB_CMD=(python -m psl_flow.training.train_terb --config "${TERB_CFG}" --default-root-dir "${TERB_RUN}" --devices "${PL_DEVICES_DEFAULT}" --accelerator "${PL_ACCELERATOR_DEFAULT}" --strategy "${PL_STRATEGY_DEFAULT}" --limit-val-batches 0 --check-val-every-n-epoch 999999)
    if [[ -n "${TERB_MAX_STEPS}" ]]; then TERB_CMD+=(--max-steps "${TERB_MAX_STEPS}"); fi
    run_stage "${ROUTE}" "terb_train_no_val" "" "${TERB_CKPT}" "${TERB_CMD[@]}"
    copy_latest_last_ckpt "${TERB_RUN}" "${TERB_CKPT}"
  fi
  assert_file "${TERB_CKPT}" "TeR-B checkpoint"
  if should_run_validation "${TERB_VAL_METRICS_JSON}"; then
    run_stage "${ROUTE}" "terb_final_validation" "" "${TERB_CKPT}" \
      python -m psl_flow.training.train_terb --config "${TERB_CFG}" --mode validate \
        --ckpt "${TERB_CKPT}" --metrics-json "${TERB_VAL_METRICS_JSON}" \
        --default-root-dir "${TERB_RUN}" --devices "${PL_DEVICES_DEFAULT}" \
        --accelerator "${PL_ACCELERATOR_DEFAULT}" --strategy "${PL_STRATEGY_DEFAULT}"
  fi

  patch_cfg_common "${PSLVAE_BASE_CFG}" "${PSLVAE_CFG}" "${DATASET_NAME}" "${TERB_CKPT}" "" "${PSLVAE_NUM_SAMPLES_PER_EPOCH}" "psl_flow"
  PSLVAE_EXPECTED_STEPS="$(expected_fit_steps "${PSLVAE_CFG}" "${PSLVAE_MAX_STEPS}")"
  if ! copy_existing_ckpt_if_any "${PSLVAE_RUN}" "${PSLVAE_CKPT}" "PSL-VAE checkpoint" "${PSLVAE_EXPECTED_STEPS}"; then
    PSLVAE_CMD=(python -m psl_flow.training.train_psl_vae --config "${PSLVAE_CFG}" --default-root-dir "${PSLVAE_RUN}" --devices "${PL_DEVICES_DEFAULT}" --accelerator "${PL_ACCELERATOR_DEFAULT}" --strategy "${PL_STRATEGY_DEFAULT}" --limit-val-batches 0 --check-val-every-n-epoch 999999)
    if [[ -n "${PSLVAE_MAX_STEPS}" ]]; then PSLVAE_CMD+=(--max-steps "${PSLVAE_MAX_STEPS}"); fi
    run_stage "${ROUTE}" "psl_vae_train_no_val" "" "${PSLVAE_CKPT}" "${PSLVAE_CMD[@]}"
    copy_latest_last_ckpt "${PSLVAE_RUN}" "${PSLVAE_CKPT}"
  fi
  assert_file "${PSLVAE_CKPT}" "PSL-VAE checkpoint"
  if should_run_validation "${PSLVAE_VAL_METRICS_JSON}"; then
    run_stage "${ROUTE}" "psl_vae_final_validation" "" "${PSLVAE_CKPT}" \
      python -m psl_flow.training.train_psl_vae --config "${PSLVAE_CFG}" --mode validate \
        --ckpt "${PSLVAE_CKPT}" --metrics-json "${PSLVAE_VAL_METRICS_JSON}" \
        --default-root-dir "${PSLVAE_RUN}" --devices "${PL_DEVICES_DEFAULT}" \
        --accelerator "${PL_ACCELERATOR_DEFAULT}" --strategy "${PL_STRATEGY_DEFAULT}"
  fi

  if [[ -f "${FLOW_STATS_JSON}" && -f "${FLOW_CFG}" ]] && stats_json_ready "${FLOW_STATS_JSON}" && flow_config_ready "${FLOW_CFG}" && ! is_truthy "${RERUN_NORMALIZER}"; then
    log_msg "Skip latent stats; config already has thermal_normalizer: ${FLOW_CFG}"
  else
    patch_cfg_common "${FLOW_BASE_CFG}" "${FLOW_CFG}" "${DATASET_NAME}" "${TERB_CKPT}" "${PSLVAE_CKPT}" "${FLOW_NUM_SAMPLES_PER_EPOCH}" "psl_flow"
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
fi

printf -v FLOW_STEP1 "%s/checkpoints/step_%06d.ckpt" "${FLOW_RUN}" "${SIT_STEP_1}"
printf -v FLOW_STEP2 "%s/checkpoints/step_%06d.ckpt" "${FLOW_RUN}" "${SIT_STEP_2}"
FLOW_VAL_METRICS_JSON="${ARTIFACT_ROOT}/metrics/${ROUTE}_final_${SIT_EVAL_SPLIT}_${DATASET_NAME}.json"

SIT_START_TS="$(date +%s)"
SIT_PEAK=0
if [[ -f "${FLOW_STEP2}" ]]; then
  log_msg "Skip SiT step ${SIT_STEP_1}; step ${SIT_STEP_2} checkpoint already exists: ${FLOW_STEP2}"
elif copy_existing_ckpt_at_or_after_step "${FLOW_RUN}" "${FLOW_STEP1}" "${SIT_STEP_1}" "${ROUTE} step ${SIT_STEP_1} checkpoint"; then
  log_msg "Reuse existing ${ROUTE} step ${SIT_STEP_1} checkpoint: ${FLOW_STEP1}"
else
  run_stage "${ROUTE}" "sit_train_to_${SIT_STEP_1}_no_val" "${SIT_STEP_1}" "${FLOW_STEP1}" \
    python -m psl_flow.training.train_psl_flow --config "${FLOW_CFG}" --mode fit \
      --default-root-dir "${FLOW_RUN}" --devices "${PL_DEVICES_DEFAULT}" \
      --accelerator "${PL_ACCELERATOR_DEFAULT}" --strategy "${PL_STRATEGY_DEFAULT}" \
      --max-steps "${SIT_STEP_1}" --checkpoint-every-n-train-steps "${SIT_STEP_1}" \
      --limit-val-batches 0 --check-val-every-n-epoch 999999
  copy_latest_last_ckpt "${FLOW_RUN}" "${FLOW_STEP1}"
  SIT_PEAK="${RESULT_PEAK_GPU_MIB}"
  SIT_CUM="$(( $(date +%s) - SIT_START_TS ))"
  append_summary "${ROUTE}" "sit_cumulative_to_${SIT_STEP_1}" "${SIT_STEP_1}" "${SIT_CUM}" "${SIT_PEAK}" "${BENCH_ROOT}/logs/${ROUTE}_sit_train_to_${SIT_STEP_1}_no_val.log" "${FLOW_STEP1}"
fi
if [[ ! -f "${FLOW_STEP2}" ]]; then
  assert_file "${FLOW_STEP1}" "${ROUTE} step ${SIT_STEP_1}"
fi

if copy_existing_ckpt_at_or_after_step "${FLOW_RUN}" "${FLOW_STEP2}" "${SIT_STEP_2}" "${ROUTE} step ${SIT_STEP_2} checkpoint"; then
  log_msg "Reuse existing ${ROUTE} step ${SIT_STEP_2} checkpoint: ${FLOW_STEP2}"
else
  run_stage "${ROUTE}" "sit_resume_to_${SIT_STEP_2}_no_val" "${SIT_STEP_2}" "${FLOW_STEP2}" \
    python -m psl_flow.training.train_psl_flow --config "${FLOW_CFG}" --mode fit \
      --default-root-dir "${FLOW_RUN}" --devices "${PL_DEVICES_DEFAULT}" \
      --accelerator "${PL_ACCELERATOR_DEFAULT}" --strategy "${PL_STRATEGY_DEFAULT}" \
      --resume-from "${FLOW_STEP1}" --max-steps "${SIT_STEP_2}" \
      --limit-val-batches 0 --check-val-every-n-epoch 999999
  copy_latest_last_ckpt "${FLOW_RUN}" "${FLOW_STEP2}"
  SIT_PEAK="$(max_int "${SIT_PEAK}" "${RESULT_PEAK_GPU_MIB}")"
  SIT_CUM="$(( $(date +%s) - SIT_START_TS ))"
  append_summary "${ROUTE}" "sit_cumulative_to_${SIT_STEP_2}" "${SIT_STEP_2}" "${SIT_CUM}" "${SIT_PEAK}" "${BENCH_ROOT}/logs/${ROUTE}_sit_resume_to_${SIT_STEP_2}_no_val.log" "${FLOW_STEP2}"
fi
assert_file "${FLOW_STEP2}" "${ROUTE} step ${SIT_STEP_2}"

EVAL_MODE="test"
if [[ "${SIT_EVAL_SPLIT}" == "val" ]]; then
  EVAL_MODE="validate"
fi

if should_run_validation "${FLOW_VAL_METRICS_JSON}"; then
  run_stage "${ROUTE}" "final_validation_${SIT_EVAL_SPLIT}" "" "${FLOW_STEP2}" \
    python -m psl_flow.training.train_psl_flow --config "${FLOW_CFG}" --mode "${EVAL_MODE}" \
      --ckpt "${FLOW_STEP2}" --metrics-json "${FLOW_VAL_METRICS_JSON}" \
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
