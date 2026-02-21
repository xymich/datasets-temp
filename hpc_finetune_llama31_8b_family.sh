#!/usr/bin/env bash
set -euo pipefail

# Change to project root so relative paths (data/, splits/, models/) resolve correctly
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Fine-tune Llama-3.1-8B for MOTIF family classification prompts
# Requires:
#   - HF_TOKEN exported and approved access to meta-llama/Llama-3.1-8B
#   - splits generated via hpc_generate_motif_family_prompts.sh
#
# Example:
#   export HF_TOKEN=hf_xxx
#   bash hpc_finetune_llama31_8b_family.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-splits/llm_motif_family_cot}"
OUTPUT_DIR="${OUTPUT_DIR:-models/llm_motif_family_llama31_8b}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-4}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set."
  exit 1
fi

${PYTHON_BIN} finetune_llm.py \
  --data-dir "${DATA_DIR}" \
  --model-name "${MODEL_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --lr "${LR}" \
  --max-length "${MAX_LENGTH}" \
  --lora-r "${LORA_R}" \
  --lora-alpha "${LORA_ALPHA}" \
  --use-4bit \
  --hf-token "${HF_TOKEN}"

echo "Fine-tuning complete: ${OUTPUT_DIR}"
