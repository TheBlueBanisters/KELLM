#!/usr/bin/env bash
# KELLM Training Script
# Multi-GPU distributed training with LoRA and Token Translator

set -euo pipefail

########################################
#           Quick Start (Edit here)    #
########################################

# DATASET: dataset name or dataset directory
# - If you provide a short name like "CoDEx-S", the script tries to locate a
#   directory matching this prefix (e.g., CoDEx-S_25) near this script.
# - You can also set DATASET_DIR directly to an absolute or relative path.
DATASET="${DATASET:-CoDEx-S}"
DATASET_DIR="${DATASET_DIR:-${DATASET}}"

# Base model path
BASE_MODEL="${BASE_MODEL:-./models/Qwen2.5-3B}"

# Output root and epochs
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
EPOCHS="${EPOCHS:-1}"

# Model family (auto|llama|qwen); precision mode (auto|bf16|fp16|fp32)
MODEL_FAMILY="${MODEL_FAMILY:-auto}"
PRECISION="${PRECISION:-auto}"

# Module switches
TRAIN_LORA="${TRAIN_LORA:-1}"
TRAIN_KELLM="${TRAIN_KELLM:-1}"

# Extra HF Trainer args (optional)
EXTRA_ARGS="${EXTRA_ARGS:---micro_batch_size 4 --batch_size 16}"

# Validation split size when no external valid file is found (0=disabled)
VAL_SET_SIZE="${VAL_SET_SIZE:-0}"

# Master port for torch.distributed
MASTER_PORT="${MASTER_PORT:-29500}"

########################################
#           Configuration              #
########################################

# Resolve dataset directory
_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Resolve DATASET_DIR across likely bases: as-is, CWD, script dir, script parent, or infer from DATASET
_resolved_dir=""
case "${DATASET_DIR}" in
  /*) [ -d "${DATASET_DIR}" ] && _resolved_dir="${DATASET_DIR}" ;;
  *)
    for base in "." "${_SCRIPT_DIR}" "$(dirname "${_SCRIPT_DIR}")"; do
      cand="${base}/${DATASET_DIR}"
      [ -d "${cand}" ] && { _resolved_dir="$(cd "${cand}" && pwd)"; break; }
    done
  ;;
 esac
if [ -z "${_resolved_dir}" ] && [ -n "${DATASET}" ]; then
  for base in "." "${_SCRIPT_DIR}" "$(dirname "${_SCRIPT_DIR}")"; do
    hit=$(ls -1d "${base}/${DATASET}_"* 2>/dev/null | head -n1 || true)
    if [ -n "${hit}" ] && [ -d "${hit}" ]; then _resolved_dir="$(cd "${hit}" && pwd)"; break; fi
  done
fi
DATASET_DIR="${_resolved_dir:-${DATASET_DIR}}"
if [ ! -d "${DATASET_DIR}" ]; then
  echo "[ERROR] Dataset directory not found: ${DATASET_DIR} (DATASET=${DATASET})" >&2
  exit 1
fi

# Resolve dataset base name (e.g., CoDEx-S_25 -> CoDEx-S)
_DATASET_BASENAME="$(basename "${DATASET_DIR}")"
DATASET_BASE="${_DATASET_BASENAME%%_*}"

# Auto-resolve train/valid file paths unless explicitly provided
if [ -z "${TRAIN_JSON:-}" ]; then
  _CAND1="${DATASET_DIR}/${DATASET_BASE}_train_with_multihop.json"
  if [ -f "${_CAND1}" ]; then
    TRAIN_JSON="${_CAND1}"
  else
    # fallbacks: prefer patterned names, then generic *train*.json
    TRAIN_JSON=""
    for pat in "*_train_with_multihop.json" "*train*.json" "train.json"; do
      _hit=$(ls -1 ${DATASET_DIR}/${pat} 2>/dev/null | head -n1 || true)
      if [ -n "${_hit}" ]; then TRAIN_JSON="${_hit}"; break; fi
    done
  fi
fi

if [ -z "${VALID_JSON:-}" ]; then
  _CAND2="${DATASET_DIR}/${DATASET_BASE}_valid_with_multihop.json"
  if [ -f "${_CAND2}" ]; then
    VALID_JSON="${_CAND2}"
  else
    VALID_JSON=""
    for pat in "*_valid_with_multihop.json" "*dev*.json" "*valid*.json" "valid.json"; do
      _hit=$(ls -1 ${DATASET_DIR}/${pat} 2>/dev/null | head -n1 || true)
      if [ -n "${_hit}" ]; then VALID_JSON="${_hit}"; break; fi
    done
  fi
fi

# Debug mode (use small testflow sample for quick testing)
DEBUG_MODE="${DEBUG_MODE:-1}"
TESTFLOW_JSON_NAME="${TESTFLOW_JSON_NAME:-testflow.json}"

# Early stopping and best model parameters
USE_EARLY_STOPPING="${USE_EARLY_STOPPING:-1}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-3}"
EARLY_STOPPING_THRESHOLD="${EARLY_STOPPING_THRESHOLD:-0.0}"
LOAD_BEST_MODEL_AT_END="${LOAD_BEST_MODEL_AT_END:-1}"
METRIC_FOR_BEST="${METRIC_FOR_BEST:-eval_loss}"
GREATER_IS_BETTER="${GREATER_IS_BETTER:-0}"

# Environment variables
export PRECISION
export PYTHONUNBUFFERED=1
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-info}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export SEED="${SEED:-42}"

echo "[CONFIG] Precision mode: ${PRECISION}"

########################################
#         Environment Setup            #
########################################

# Use system Python or detect available Python
PYBIN="$(command -v python)"
echo "[PYTHON] Using: ${PYBIN} ($("${PYBIN}" -V 2>&1))"

# Create output directory with timestamp
OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}_$(date +%F_%H-%M-%S)"
mkdir -p "${OUTPUT_DIR}"

# Check Python environment
if ! "${PYBIN}" -c "import sys; print(sys.executable)" >/dev/null 2>&1; then
  echo "[ERROR] Cannot run ${PYBIN}" >&2
  exit 1
fi

# Check required packages
if ! "${PYBIN}" -c "import fire" 2>/dev/null; then
  echo "[SETUP] Installing 'fire'..."
  "${PYBIN}" -m pip install fire >/dev/null || {
    echo "[ERROR] Failed to install 'fire'" >&2; exit 1; }
fi

if ! "${PYBIN}" -c "import sentencepiece,google.protobuf" 2>/dev/null; then
  echo "[SETUP] Installing sentencepiece and protobuf..."
  "${PYBIN}" -m pip install 'sentencepiece==0.1.99' 'protobuf>=3.20,<5' >/dev/null || {
    echo "[ERROR] Failed to install sentencepiece/protobuf" >&2; exit 1; }
fi

"${PYBIN}" -c "import torch, transformers; print('[CHECK] torch=',torch.__version__,' transformers=',transformers.__version__)" \
  || { echo "[ERROR] Missing torch/transformers" >&2; exit 1; }

# Display configuration
echo "[CONFIG] Base model: ${BASE_MODEL}"
echo "[CONFIG] Model family: ${MODEL_FAMILY}"
echo "[CONFIG] Train data: ${TRAIN_JSON}"
echo "[CONFIG] Valid data: ${VALID_JSON}"
echo "[CONFIG] Output dir: ${OUTPUT_DIR}"
echo "[CONFIG] Epochs: ${EPOCHS}"

# Check training data existence
if [ "${DEBUG_MODE}" = "1" ]; then
  TESTFLOW_JSON_PATH="${DATASET_DIR}/${TESTFLOW_JSON_NAME}"
  if [ -f "${TESTFLOW_JSON_PATH}" ]; then
    echo "[DEBUG] Using testflow data: ${TESTFLOW_JSON_PATH}"
    TRAIN_JSON="${TESTFLOW_JSON_PATH}"
  else
    echo "[DEBUG] Testflow file not found, will use regular training data"
  fi
fi

if [ ! -f "${TRAIN_JSON}" ]; then
  echo "[ERROR] Training file not found: ${TRAIN_JSON}" >&2
  exit 1
fi

# Check validation data
HAS_EXT_VALID=0
if [ -f "${VALID_JSON}" ]; then
  HAS_EXT_VALID=1
  echo "[CONFIG] Using external validation set: ${VALID_JSON}"
else
  echo "[CONFIG] No external validation set, will split from training data (size: ${VAL_SET_SIZE})"
fi

########################################
#         GPU Detection               #
########################################

# Auto-detect available GPUs
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"
  else
    export CUDA_VISIBLE_DEVICES="0"
  fi
fi

echo "[CONFIG] Initial CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

# Clean up old processes
echo "[CLEANUP] Killing old training processes..."
pkill -9 -f 'torch.distributed.run' || true
pkill -9 -f 'torchrun' || true
pkill -9 -f 'train_kellm.py' || true
sleep 1

# GPU health check
PRECHECK_LOG="${OUTPUT_DIR}/gpu_precheck.log"
"${PYBIN}" - > "${PRECHECK_LOG}" 2>&1 <<'PY'
import os, torch
ids=[s for s in os.environ.get("CUDA_VISIBLE_DEVICES","").split(",") if s.strip()]
print("[PRECHECK] Visible GPU IDs:", ids)
n=torch.cuda.device_count()
print("[PRECHECK] PyTorch sees", n, "devices")
good=[]
for j in range(min(len(ids), n)):
    name=torch.cuda.get_device_name(j)
    try:
        free,total=torch.cuda.mem_get_info(j)
        print("[PRECHECK] GPU {} -> {} | Free: {}MiB Total: {}MiB".format(
            j, name, free//(1024**2), total//(1024**2)))
        good.append(ids[j])
    except Exception as e:
        print("[PRECHECK] GPU {} FAILED:".format(j), e)
print(",".join(good))
PY

GOOD_GPUS=$(tail -n 1 "${PRECHECK_LOG}" | tr -d '\r')
[ -z "${GOOD_GPUS}" ] && { echo "[ERROR] GPU precheck failed:"; cat "${PRECHECK_LOG}"; exit 1; }
export CUDA_VISIBLE_DEVICES="${GOOD_GPUS}"
NGPUS=$(echo "${GOOD_GPUS}" | awk -F',' '{print NF}')
echo "[CONFIG] Using GPUs: ${CUDA_VISIBLE_DEVICES} (Count: ${NGPUS})"
echo "[INFO] GPU precheck details in: ${PRECHECK_LOG}"

########################################
#         Build Training Command      #
########################################

CMD=(
  "${PYBIN}" -u -m torch.distributed.run --nproc_per_node="${NGPUS}" --master_port="${MASTER_PORT}"
  train_kellm.py
  --base_model "${BASE_MODEL}"
  --data_path "${TRAIN_JSON}"
  --output_dir "${OUTPUT_DIR}"
  --num_prefix 1
  --kge_model "${DATASET_DIR}"
  --prompt_template_name alpaca
  --train_lora "$( [ "${TRAIN_LORA}" = "1" ] && echo True || echo False )"
  --train_kellm "$( [ "${TRAIN_KELLM}" = "1" ] && echo True || echo False )"
  --num_epochs "${EPOCHS}"
)

# Add validation data
if [ "${HAS_EXT_VALID}" = "1" ]; then
  CMD+=( --valid_data_path "${VALID_JSON}" )
elif [ "${VAL_SET_SIZE}" != "0" ]; then
  CMD+=( --val_set_size "${VAL_SET_SIZE}" )
fi

# Add early stopping parameters
if [ "${USE_EARLY_STOPPING}" = "1" ]; then
  CMD+=( --use_early_stopping True
         --early_stopping_patience "${EARLY_STOPPING_PATIENCE}"
         --early_stopping_threshold "${EARLY_STOPPING_THRESHOLD}" )
else
  CMD+=( --use_early_stopping False )
fi

if [ "${LOAD_BEST_MODEL_AT_END}" = "1" ]; then
  CMD+=( --load_best_model_at_end True )
else
  CMD+=( --load_best_model_at_end False )
fi

# Add metric configuration
if [ -n "${METRIC_FOR_BEST}" ]; then
  CMD+=( --metric_for_best_model "${METRIC_FOR_BEST}" )
fi
if [ "${GREATER_IS_BETTER}" = "1" ]; then
  CMD+=( --greater_is_better True )
else
  CMD+=( --greater_is_better False )
fi

# Auto-detect model family
if [ "${MODEL_FAMILY}" = "auto" ]; then
  case "${BASE_MODEL,,}" in
    *qwen*) MODEL_FAMILY="qwen" ;;
    *) MODEL_FAMILY="llama" ;;
  esac
fi
CMD+=( --model_family "${MODEL_FAMILY}" )

# Add extra training arguments
if [ -n "${EXTRA_ARGS}" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=( ${EXTRA_ARGS} ); CMD+=( "${EXTRA_ARR[@]}" )
fi

echo "[COMMAND] ${CMD[*]}"

########################################
#         Launch Training             #
########################################

# Launch training in background
nohup bash -lc "${CMD[*]}" > "${OUTPUT_DIR}/train.log" 2>&1 &
LAUNCH_PID=$!
echo "[SUCCESS] Training launched: PID=${LAUNCH_PID} GPUs=${CUDA_VISIBLE_DEVICES}"
echo "[LOG] Monitor with: tail -f ${OUTPUT_DIR}/train.log"