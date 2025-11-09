from argparse import ArgumentParser, Namespace
import json
import logging
import random
from typing import Literal, cast
import uuid

from datasets import Dataset, DatasetDict, load_dataset
from openai.types.chat import ChatCompletionMessageParam

import wandb
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer
from accelerate import PartialState

from src._format import format_user_message
from src._types import SHARPCard


class Args(Namespace):
    base_model: str
    dataset: str
    base_instruction_file: str

    # LoRA specific args
    lora_rank: int
    lora_alpha: int
    lora_dropout: float

    # General training args
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    weight_decay: float
    warmup_ratio: float
    wandb_project: str
    lr_scheduler_type: Literal[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "constant",
        "constant_with_warmup",
    ]
    max_steps: int
    save_steps: int
    save_total_limit: int
    dataset_shuffle_seed: int
    push_to_hub: bool
    hub_model_id: str | None


def train_sft(cli_args: Args):
    logging.info("Loading tokenizer for model: %s", cli_args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(cli_args.base_model)

    with open(cli_args.base_instruction_file, "r") as f:
        base_instruction = f.read()

    def _apply_chat_template(row: dict) -> dict:
        messages = _build_messages(cast(SHARPCard, row), base_instruction)

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        return {"text": text}

    logging.info("Loading dataset: %s", cli_args.dataset)
    sft_dataset = load_dataset(cli_args.dataset)
    assert isinstance(sft_dataset, DatasetDict), "Dataset is not a DatasetDict"
    assert "train" in sft_dataset, "Train dataset is not in the dataset"

    logging.info("Balancing dataset")
    sft_dataset = DatasetDict(
        {
            "train": _balance_dataset(sft_dataset["train"], seed=cli_args.dataset_shuffle_seed),
            "test": _balance_dataset(sft_dataset["test"], seed=cli_args.dataset_shuffle_seed),
        }
    )

    sft_dataset = sft_dataset.map(_apply_chat_template, remove_columns=sft_dataset["train"].column_names)

    # Create output_dir before wandb init (all processes need the same path)
    run_id = str(uuid.uuid4())[:6]
    output_dir = f"./checkpoints/{run_id}"

    if PartialState().is_main_process:
        wandb.init(project=cli_args.wandb_project, name=run_id)
        assert wandb.run is not None, "wandb.run is not initialized"

    sft_training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=cli_args.batch_size,
        gradient_accumulation_steps=cli_args.gradient_accumulation_steps,
        dataset_text_field="text",
        # in practice the completions for this step are much shorter then our theoretical max (max_prompt_length + max_completion_length ~= 4096)
        max_seq_length=4096,
        num_train_epochs=cli_args.epochs,
        learning_rate=cli_args.learning_rate,
        weight_decay=cli_args.weight_decay,
        warmup_ratio=cli_args.warmup_ratio,
        lr_scheduler_type=cli_args.lr_scheduler_type,
        bf16=True,
        fp16=False,
        logging_steps=4,
        max_steps=cli_args.max_steps,
        save_steps=cli_args.save_steps,
        save_total_limit=cli_args.save_total_limit,
        report_to=["wandb"] if PartialState().is_main_process else [],
    )

    lora_config = LoraConfig(
        r=cli_args.lora_rank,
        lora_alpha=cli_args.lora_alpha,
        lora_dropout=cli_args.lora_dropout,
        target_modules=[
            "embed_tokens",
            "q_proj",
            "v_proj",
            "o_proj",
            "k_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_trainer = SFTTrainer(
        model=cli_args.base_model,
        processing_class=tokenizer,
        train_dataset=sft_dataset["train"],
        eval_dataset=sft_dataset["test"],
        peft_config=lora_config,
        args=sft_training_args,
    )

    # Print number of trainable parameters
    if PartialState().is_main_process:
        sft_trainer.model.print_trainable_parameters()

    try:
        sft_trainer.train()
    except Exception as e:
        if PartialState().is_main_process:
            wandb.finish(exit_code=1)
        raise e

    # Save the final adapter weights
    final_adapter_path = f"{output_dir}/final_adapter"
    if PartialState().is_main_process:
        logging.info(f"Saving final adapter to {final_adapter_path}")
        sft_trainer.model.save_pretrained(final_adapter_path)
        tokenizer.save_pretrained(final_adapter_path)
        logging.info(f"✅ Adapter saved to {final_adapter_path}")

        # Push to hub if requested (pushes adapter weights only, not merged weights)
        if cli_args.push_to_hub:
            hub_model_id = cli_args.hub_model_id or f"sharp-pluckability-{run_id}"
            logging.info(f"Pushing adapter to Hugging Face Hub as {hub_model_id}")
            sft_trainer.model.push_to_hub(hub_model_id)
            tokenizer.push_to_hub(hub_model_id)
            logging.info(f"✅ Adapter pushed to Hugging Face Hub: {hub_model_id}")

        wandb.finish(exit_code=0)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--base_instruction_file", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, required=True, help="Wandb project name")

    # LoRA specific args
    parser.add_argument("--lora_rank", type=int, required=True)
    parser.add_argument("--lora_alpha", type=int, required=True)
    parser.add_argument("--lora_dropout", type=float, required=True)

    parser.add_argument("--dataset", type=str, default="ozziek/SHARP-Card")
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.2)
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="constant_with_warmup",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument("--max_steps", type=int, required=False, default=-1)
    parser.add_argument("--save_steps", type=int, required=False, default=8)
    parser.add_argument("--save_total_limit", type=int, required=False, default=2)
    parser.add_argument("--dataset_shuffle_seed", type=int, required=False, default=67413)
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Push the model to Hugging Face Hub after training"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        required=False,
        default=None,
        help="Model ID on Hugging Face Hub (defaults to sharp-pluckability-{run_id})",
    )
    args: Args = parser.parse_args(namespace=Args())

    train_sft(args)
