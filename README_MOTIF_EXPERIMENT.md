# MOTIF Family Classification Experiment (Stage 1-4 + LLM Fine-Tuning)

This document tracks the MOTIF-specific experiment for malware family classification using this repository pipeline.

## 1) Experiment Goal

Train and evaluate a model for malware family classification on MOTIF using:
- Stage 1-2 extracted features
- Stage 3 clustering context
- Chain-of-thought (CoT) structured prompt targets
- Fine-tuning with Hugging Face base model `meta-llama/Llama-3.1-8B`

## 2) Dataset and Ground Truth

- Source repository: https://github.com/boozallen/MOTIF
- Total samples processed: 3095
- Ground-truth families loaded from MOTIF metadata
- Family aliases supported via MOTIF family alias table

Primary generated dataset:
- `data/MOTIF/motif_stage2_dataset_full.jsonl`

## 3) Pipeline Status (MOTIF Full Run)

Completed:
- Stage 1 + Stage 2: 3095/3095 succeeded
- Stage 3 clustering: completed on full Stage 2 outputs

Stage 3 evaluation output:
- `data/MOTIF/eval_stage3_full.json`

Observed Stage 3 metrics on full set:
- Samples: 3095
- True families: 502
- Predicted clusters: 198
- ARI: 0.0177
- NMI: 0.6190
- Homogeneity: 0.5364
- Completeness: 0.7317
- V-measure: 0.6190

## 4) MOTIF-Specific Scripts Added

Prompt and training workflow:
- `generate_family_prompts.py`
- `evaluate_finetuned_family_llm.py`

HPC helper scripts:
- `hpc_generate_motif_family_prompts.sh`
- `hpc_finetune_llama31_8b_family.sh`
- `hpc_eval_llama31_8b_family.sh`

## 5) CoT Prompt Dataset (Metric-Friendly)

CoT split directory:
- `splits/llm_family_cot`

Metadata:
- `splits/llm_family_cot/metadata.json`

Current split summary:
- Total prompt records: 2941
- Families: 348
- Train: 1914
- Val: 282
- Test: 745

Schema highlights (structured output target):
- reasoning_chain.identify_claimed_identity
- reasoning_chain.analyze_capabilities
- reasoning_chain.context_from_cluster
- reasoning_chain.verdict
- predicted_family
- confidence
- evidence_refs

This structure is designed to be directly metricised for both correctness and reasoning quality.

## 6) End-to-End Commands

### Step A: Generate CoT prompts

```bash
python generate_family_prompts.py \
  --input data/MOTIF/motif_stage2_dataset_full.jsonl \
  --output-dir splits/llm_family_cot \
  --min-family-count 2
```

### Step B: Hugging Face auth for gated Llama model

```bash
export HF_TOKEN="<your_hf_token>"
huggingface-cli login --token "$HF_TOKEN"
```

You must have approved access to:
- `meta-llama/Llama-3.1-8B`

### Step C: Fine-tune on HPC

```bash
python finetune_llm.py \
  --data-dir splits/llm_family_cot \
  --model-name meta-llama/Llama-3.1-8B \
  --output-dir models/llm_family_llama31_8b \
  --epochs 3 \
  --batch-size 1 \
  --grad-accum 16 \
  --lr 2e-4 \
  --max-length 2048 \
  --lora-r 16 \
  --lora-alpha 32 \
  --use-4bit
```

Notes:
- `finetune_llm.py` now supports Hugging Face token access via `--hf-token` or `HF_TOKEN` env var.
- Adjust batch and grad accumulation to your GPU memory budget.

### Step D: Evaluate family + CoT quality

```bash
python evaluate_finetuned_family_llm.py \
  --model-dir models/llm_family_llama31_8b \
  --test-file splits/llm_family_cot/test.jsonl \
  --repo-dir data/MOTIF/repo \
  --max-new-tokens 220 \
  --output data/MOTIF/eval_family_llm_cot.json
```

## 7) Evaluation Metrics for Fine-Tuned CoT Model

`evaluate_finetuned_family_llm.py` reports:
- exact_accuracy
- alias_accuracy
- unknown_prediction_rate
- json_parse_rate
- full_reasoning_chain_rate
- avg_reasoning_steps_present
- avg_evidence_grounded_ratio

Output report file example:
- `data/MOTIF/eval_family_llm_cot.json`

## 8) Suggested Experiment Record Fields

For reproducibility, log at minimum:
- model name and revision
- LoRA hyperparameters
- train/val/test sizes
- seed
- max_length and max_new_tokens
- exact and alias accuracy
- reasoning structure compliance metrics

## 9) Known Constraints

- Stage 4 family verdict benchmark on full set depends on completion of Stage 4 verdict generation.
- Llama 8B fine-tuning requires suitable GPU resources (recommended HPC environment).
- Hugging Face gated access is mandatory for `meta-llama/Llama-3.1-8B`.
