#!/bin/bash

set -e  # Exit on any error

# Get the script's directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# setup initial environment
uv sync

uv pip install 'ms-swift[all]' -U

source "$PROJECT_ROOT/.venv/bin/activate"

uv run pluckability/form_dataset_sft.py \
  --base_instruction "pluckability/instructions/sft_instruction.txt" \
  --balance \
  --split "train" \
  -o "train_ms_sft.jsonl"

uv run pluckability/form_dataset_sft.py \
  --base_instruction "pluckability/instructions/sft_instruction.txt" \
  --balance \
  --split "test" \
  -o "test_ms_sft.jsonl"

echo "✅ Dataset created!"

# use `--loss_scale ignore_empty_think`
# Avoid losing the think capability by ignoring the loss of empty `<think>\n\n</think>\n\n`
# This method is also applicable to the Deepseek-R1 series of models.

pip install deepspeed
pip install wandb

nproc_per_node=4

CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=$nproc_per_node swift sft \
    --model "Qwen/Qwen3-32B" \
    --use_hf true \
    --train_type lora \
    --dataset './train_ms_sft.jsonl' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps $(expr 8 / $nproc_per_node) \
    --eval_steps 10 \
    --save_steps 10 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --label_names labels \
    --load_from_cache_file false \
    --loss_scale ignore_empty_think \
    --model_author laddermedia \
    --model_name SHARP-Qwen3-32B-Pluck \
    --report_to wandb \
    --deepspeed zero2

pip install vllm

CUDA_VISIBLE_DEVICES=0,1,2,3 swift infer \
    --adapters "./output/v4-20251109-061722/checkpoint-61" \
    --infer_backend vllm \
    --val_dataset './eval_train_ms_sft.jsonl' 'eval_test_ms_sft.jsonl' \
    --stream true \
    --use_hf true \
    --gpu_memory_utilization 0.9 \
    --tensor_parallel_size 2 \
    --temperature 0 \
    --max_new_tokens 64