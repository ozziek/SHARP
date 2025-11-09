#!/bin/bash

set -e  # Exit on any error

echo "🚀 Starting HuggingFace SFT training..."

# setup initial environment
uv sync

source .venv/bin/activate

# Install training dependencies
pip install transformers peft trl accelerate wandb

# Run training
python pluckability/train_hf_sft.py \
    --base_model "Qwen/Qwen3-32B" \
    --base_instruction_file "pluckability/instructions/sft_instruction.txt" \
    --wandb_project "sharp-pluckability" \
    --lora_rank 32 \
    --lora_alpha 32 \
    --lora_dropout 0 \
    --gradient_accumulation_steps 1 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --epochs 1 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --save_steps 50 \
    --save_total_limit 3 \
    --push_to_hub \
    --hub_model_id "ozziek/SHARP-Qwen3-32B-Pluck"

echo "✅ HuggingFace SFT training complete!"

