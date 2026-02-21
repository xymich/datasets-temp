#!/usr/bin/env bash
set -euo pipefail

# Change to project root so relative paths (data/, splits/) resolve correctly
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Generate family-classification prompts from MOTIF Stage-2 feature dataset
# Usage:
#   bash hpc_generate_motif_family_prompts.sh
#   INPUT_JSONL=data/MOTIF/motif_stage2_dataset_full.jsonl OUT_DIR=splits/llm_family bash hpc_generate_motif_family_prompts.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_JSONL="${INPUT_JSONL:-data/datasets-temp/motif_stage2_dataset_full.jsonl}"
BENIGN_JSONL="${BENIGN_JSONL:-data/datasets-temp/benign_features.jsonl}"
OUT_DIR="${OUT_DIR:-splits/llm_motif_family_cot}"
TRAIN_RATIO="${TRAIN_RATIO:-0.7}"
VAL_RATIO="${VAL_RATIO:-0.15}"
SEED="${SEED:-42}"
MIN_FAMILY_COUNT="${MIN_FAMILY_COUNT:-2}"

BENIGN_ARG=""
if [ -f "${BENIGN_JSONL}" ]; then
  BENIGN_ARG="--benign-input ${BENIGN_JSONL}"
  echo "Mixing in benign samples from: ${BENIGN_JSONL}"
fi

${PYTHON_BIN} experiments/motif/generate_family_prompts.py \
  --input "${INPUT_JSONL}" \
  --output-dir "${OUT_DIR}" \
  --train-ratio "${TRAIN_RATIO}" \
  --val-ratio "${VAL_RATIO}" \
  --seed "${SEED}" \
  --min-family-count "${MIN_FAMILY_COUNT}" \
  ${BENIGN_ARG}

echo "Prompt generation complete: ${OUT_DIR}"
