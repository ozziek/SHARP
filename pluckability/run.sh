#!/bin/bash

# Pluckability Evaluation Script
# Tests GPT-5 and GPT-4.1 with zero-shot and multi-shot instructions (no temperature specified)

set -e  # Exit on any error

echo "🚀 Starting pluckability evaluation runs..."

# Create data directory if it doesn't exist
mkdir -p pluckability/data

# # # Run 1: Zero-shot GPT-5 (no temperature specified)
# echo "📊 Running zero-shot GPT-5 evaluation (default temperature)..."
# uv run pluckability/generate_completions.py \
#     --model "gpt-5" \
#     --instruction-file "pluckability/instructions/zero_shot.txt" \
#     --max-concurrency 18

# echo "✅ Zero-shot GPT-5 evaluation complete!"

# Run 2: Zero-shot GPT-4.1 (no temperature specified)
# echo "📊 Running zero-shot GPT-4.1 evaluation (default temperature)..."
# uv run pluckability/generate_completions.py \
#     --model "gpt-4.1" \
#     --instruction-file "pluckability/instructions/zero_shot.txt" \
#     --max-concurrency 18

# echo "✅ Zero-shot GPT-4.1 evaluation complete!"

# Run 3: Multi-shot GPT-5 (no temperature specified)
# echo "📊 Running multi-shot GPT-5 evaluation (default temperature)..."
# uv run pluckability/generate_completions.py \
#     --model "gpt-5" \
#     --instruction-file "pluckability/instructions/multi_shot.txt" \
#     --max-concurrency 18

# echo "✅ Multi-shot GPT-5 evaluation complete!"

# Run 4: Multi-shot GPT-4.1 (no temperature specified)
# echo "📊 Running multi-shot GPT-4.1 evaluation (default temperature)..."
# uv run pluckability/generate_completions.py \
#     --model "gpt-4.1" \
#     --instruction-file "pluckability/instructions/multi_shot.txt" \
#     --max-concurrency 18

# echo "✅ Multi-shot GPT-4.1 evaluation complete!"

# uv run pluckability/generate_completions.py \
#     --model "ft:gpt-4.1-2025-04-14:personal:sharp-pluckability:CZtsdFlR:ckpt-step-16" \
#     --instruction-file "pluckability/instructions/sft_instruction_slim.txt" \
#     --max-concurrency 18

# Using Cerebras API with qwen3-32b to get baseline performance (before FT)
# Requires CEREBRAS_API_KEY environment variable to be set
echo "📊 Running evaluation with qwen3-32b via Cerebras..."
uv run pluckability/generate_completions.py \
    --model "qwen-3-32b" \
    --instruction-file "pluckability/instructions/zero_shot.txt" \
    --base-url "https://api.cerebras.ai/v1" \
    --auth-token "$CEREBRAS_API_KEY" \
    --max-concurrency 10


