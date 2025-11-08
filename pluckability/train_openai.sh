
set -e  # Exit on any error

echo "🚀 Starting OpenAI SFT training..."

uv run pluckability/train_openai_sft.py \
    --dataset "ozziek/SHARP-Card" \
    --model "gpt-4.1-2025-04-14" \
    --base-instruction-file "pluckability/instructions/sft_instruction.txt"

echo "✅ OpenAI SFT training kickoff complete!"