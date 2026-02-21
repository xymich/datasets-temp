#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=Motif_Llama31_8B
#SBATCH --time=1-23:59:00
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --output=logs/finetune_motif_8b_%j.out
#SBATCH --error=logs/finetune_motif_8b_%j.err

# Exit on any error
set -e

# Create logs directory
mkdir -p logs

echo "=================================="
echo "LLM Fine-tuning - Llama 3.1 8B"
echo "Dataset: MOTIF Family Classification (CoT)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=================================="

# Activate conda environment
source /users/40365657/sharedscratch/miniconda3/bin/activate || { echo "Failed to source conda"; exit 1; }
conda activate /users/40365657/sharedscratch/conda_envs/llm_finetune_v2 || { echo "Failed to activate conda env"; exit 1; }

# Print environment info
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CPUs available: $SLURM_CPUS_PER_TASK"
echo ""

# Check CUDA availability
echo "GPU Information:"
nvidia-smi || { echo "nvidia-smi failed"; exit 1; }
echo ""

# Set HuggingFace token from file
if [ ! -f /users/40365657/sharedscratch/huggingface_cache/token ]; then
    echo "ERROR: HuggingFace token file not found at /users/40365657/sharedscratch/huggingface_cache/token"
    exit 1
fi
export HF_TOKEN=$(cat /users/40365657/sharedscratch/huggingface_cache/token)
echo "✓ HuggingFace token loaded"
echo ""

# Disable DeepSpeed auto-detection and set cache directories
export ACCELERATE_USE_DEEPSPEED=false
export DS_SKIP_CUDA_CHECK=1
export TRITON_CACHE_DIR=/tmp/triton_cache_$$
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Prevent CUDA initialization hangs (critical for bitsandbytes in SLURM)
export CUDA_LAUNCH_BLOCKING=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

echo "✓ DeepSpeed disabled (using native PyTorch)"
echo ""

# Change to project root
cd /users/40365657/sharedscratch/malware-classification/capa-malware || { echo "Failed to cd to project root"; exit 1; }

# ── Step 1: Generate CoT prompt splits (if not already done) ────────────────
DATA_DIR="splits/llm_motif_family_cot"

if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "Generating MOTIF CoT prompt splits..."
    MOTIF_INPUT="${MOTIF_INPUT:-data/motif_stage2_dataset_full.jsonl}"

    if [ ! -f "$MOTIF_INPUT" ]; then
        echo "ERROR: MOTIF input dataset not found: $MOTIF_INPUT"
        echo "Transfer it first:  scp data/MOTIF/motif_stage2_dataset_full.jsonl <hpc>:~/sharedscratch/malware-classification/capa-malware/"
        exit 1
    fi

    python experiments/motif/generate_family_prompts.py \
        --input "$MOTIF_INPUT" \
        --output-dir "$DATA_DIR" \
        --min-family-count 2 \
        --train-ratio 0.70 \
        --val-ratio 0.15 \
        --seed 42

    echo "✓ Splits generated in $DATA_DIR"
else
    echo "✓ Prompt splits already exist in $DATA_DIR — skipping generation"
fi

echo ""
echo "Dataset statistics:"
echo "  Train: $(wc -l < $DATA_DIR/train.jsonl 2>/dev/null || echo 'N/A') samples"
echo "  Val:   $(wc -l < $DATA_DIR/val.jsonl   2>/dev/null || echo 'N/A') samples"
echo "  Test:  $(wc -l < $DATA_DIR/test.jsonl  2>/dev/null || echo 'N/A') samples"
echo ""

# ── Step 2: Fine-tune ─────────────────────────────────────────────────────────
OUTPUT_DIR="models/llm_motif_family_llama31_8b"

echo "Starting fine-tuning..."
echo "  Model:      meta-llama/Llama-3.1-8B"
echo "  Output dir: $OUTPUT_DIR"
echo ""

srun python finetune_llm.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --model-name meta-llama/Llama-3.1-8B \
    --epochs 3 \
    --batch-size 1 \
    --grad-accum 4 \
    --lr 2e-4 \
    --max-length 2048 \
    --lora-r 16 \
    --lora-alpha 32 \
    --use-4bit

TRAIN_EXIT_CODE=$?

echo ""
echo "=================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ Training Complete Successfully!"
    echo "Model saved to: $OUTPUT_DIR"
else
    echo "✗ Training Failed with exit code: $TRAIN_EXIT_CODE"
fi
echo "Date: $(date)"
echo "=================================="

exit $TRAIN_EXIT_CODE
