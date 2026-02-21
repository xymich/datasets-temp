#!/usr/bin/env bash
set -euo pipefail

# Change to project root so relative paths (data/, splits/, models/) resolve correctly
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Evaluate fine-tuned family model (exact + alias-match accuracy)
# Usage:
#   bash hpc_eval_llama31_8b_family.sh
#   MODEL_DIR=models/llm_family_llama31_8b/checkpoint-1000 bash hpc_eval_llama31_8b_family.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_DIR:-models/llm_motif_family_llama31_8b}"
TEST_FILE="${TEST_FILE:-splits/llm_motif_family_cot/test.jsonl}"
REPO_DIR="${REPO_DIR:-data/datasets-temp/repo}"
OUT_JSON="${OUT_JSON:-data/datasets-temp/eval_motif_family_llama31_8b.json}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
TEMPERATURE="${TEMPERATURE:-0.0}"

${PYTHON_BIN} experiments/motif/evaluate_finetuned_family_llm.py \
  --model-dir "${MODEL_DIR}" \
  --test-file "${TEST_FILE}" \
  --repo-dir "${REPO_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --output "${OUT_JSON}"

echo "Evaluation complete: ${OUT_JSON}"
