# MOTIF Experiment Runbook

Canonical experiment README:
- `README_MOTIF_EXPERIMENT.md`

Use this folder as the MOTIF experiment home for configs/log references.

## Core Commands

### 1) Build CoT prompt splits

```bash
python generate_family_prompts.py \
  --input data/MOTIF/motif_stage2_dataset_full.jsonl \
  --output-dir splits/llm_family_cot \
  --min-family-count 2
```

### 2) Fine-tune Llama-3.1-8B

```bash
export HF_TOKEN="<your_hf_token>"
python finetune_llm.py \
  --data-dir splits/llm_family_cot \
  --model-name meta-llama/Llama-3.1-8B \
  --output-dir models/llm_family_llama31_8b \
  --epochs 3 --batch-size 1 --grad-accum 16 --lr 2e-4 \
  --max-length 2048 --lora-r 16 --lora-alpha 32 --use-4bit
```

### 3) Evaluate family + reasoning

```bash
python evaluate_finetuned_family_llm.py \
  --model-dir models/llm_family_llama31_8b \
  --test-file splits/llm_family_cot/test.jsonl \
  --repo-dir data/MOTIF/repo \
  --max-new-tokens 220 \
  --output data/MOTIF/eval_family_llm_cot.json
```

## Shared Reasoning Standard

See:
- `experiments/common/LLM_REASONING_PROMPT_CHANGES.md`
