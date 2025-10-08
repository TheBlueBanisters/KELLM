#!/usr/bin/env bash
# KELLM Evaluation Script
# Multi-GPU parallel evaluation with automatic metrics calculation

set -euo pipefail

########################################
#           Quick Start (Edit here)    #
########################################

# DATASET: dataset name or dataset directory (recommended to set)
# - If you give just a name like "CoDEx-S", the script will try to locate a
#   directory whose name starts with this prefix (e.g., CoDEx-S_25) near this script.
# - You can also set DATASET_DIR directly to an absolute or relative path.
DATASET="${DATASET:-CoDEx-S}"
DATASET_DIR="${DATASET_DIR:-${DATASET}}"
# CHECKPOINT: point to a specific training checkpoint directory
# - The script will infer adapter dir from this path.
CHECKPOINT="${CHECKPOINT:-}"

########################################
#           Configuration              #
########################################

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
DATA_PATH="${DATA_PATH:-}"
if [ -z "${DATA_PATH}" ]; then
  _BASE="$(basename "${DATASET_DIR}")"; BASE_NO_SUFFIX="${_BASE%%_*}"
  _CAND_T="${DATASET_DIR}/${BASE_NO_SUFFIX}_test_with_multihop.json"
  if [ -f "${_CAND_T}" ]; then
    DATA_PATH="${_CAND_T}"
  else
    for pat in "*_test_with_multihop.json" "*test*.json" "test.json"; do
      _hit=$(ls -1 ${DATASET_DIR}/${pat} 2>/dev/null | head -n1 || true)
      if [ -n "${_hit}" ]; then DATA_PATH="${_hit}"; break; fi
    done
  fi
fi
if [ -z "${DATA_PATH}" ]; then
  echo "[ERROR] Could not resolve test file under ${DATASET_DIR}. Set DATA_PATH explicitly." >&2
  exit 1
fi
BASE_MODEL="${BASE_MODEL:-models/Qwen2.5-3B}"

# Evaluation parameters
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
DTYPE="${DTYPE:-fp16}"
TEMPLATE="${TEMPLATE:-alpaca}"

# Evaluation scope
EVAL_FIRST_N="${EVAL_FIRST_N:-10}"

# Output configuration
TS=$(date +%F_%H-%M-%S)
RUN_DIR="outputs/eval_logs/eval_${TS}"
mkdir -p "$RUN_DIR"
PRED_FILE="${RUN_DIR}/predictions.jsonl"

# Get adapter directory (parent of checkpoint)
ADAPTER_DIR="$(dirname "$CHECKPOINT")"
# Auto-correct adapter dir: if parent lacks adapter_config.json but checkpoint has it, use checkpoint
if [ ! -f "${ADAPTER_DIR}/adapter_config.json" ] && [ -f "${CHECKPOINT}/adapter_config.json" ]; then
  echo "[FIX] Parent has no adapter_config.json; using checkpoint as adapter dir"
  ADAPTER_DIR="${CHECKPOINT}"
fi

# Set KGE embedding paths
_DATA_DIR="$(dirname "$DATA_PATH")"
if [ -n "$_DATA_DIR" ]; then
  case "$_DATA_DIR" in
    /*) : ;;  # already absolute
    *) _DATA_DIR="$(pwd)/${_DATA_DIR}" ;;
  esac
fi
if [ -f "${_DATA_DIR}/entity_embedding.npy" ] && [ -f "${_DATA_DIR}/relation_embedding.npy" ]; then
  export KGE_ENTITY_NPY="${_DATA_DIR}/entity_embedding.npy"
  export KGE_RELATION_NPY="${_DATA_DIR}/relation_embedding.npy"
fi

echo "[CONFIG] Data path: ${DATA_PATH}"
echo "[CONFIG] Base model: ${BASE_MODEL}"
echo "[CONFIG] Checkpoint: ${CHECKPOINT}"
echo "[CONFIG] Adapter dir: ${ADAPTER_DIR}"
echo "[CONFIG] Output dir: ${RUN_DIR}"

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

echo "[CONFIG] Using GPUs: ${CUDA_VISIBLE_DEVICES}"

# Set environment variables for reduced logging
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

########################################
#         Build Evaluation Command    #
########################################

CMD=(
  python -W ignore evaluation.py
  --data_path "$DATA_PATH"
  --model_path "$BASE_MODEL"
  --adapter_path "$ADAPTER_DIR"
  --template "$TEMPLATE"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --dtype "$DTYPE"
  --use_fast_tokenizer 0
  --eval_first_n "$EVAL_FIRST_N"
  --save_predictions "$PRED_FILE"
  --log_llm_native 0
)

echo "[COMMAND] ${CMD[*]}"

########################################
#         Launch Evaluation           #
########################################

# Launch evaluation in background
nohup bash -lc "${CMD[*]}" > "${RUN_DIR}/eval.log" 2>&1 &
EVAL_PID=$!
echo "[SUCCESS] Evaluation launched: PID=${EVAL_PID} GPUs=${CUDA_VISIBLE_DEVICES}"
echo "[OUTPUT] Predictions will be saved to: ${PRED_FILE}"
echo "[LOG] Monitor with: tail -f ${RUN_DIR}/eval.log"