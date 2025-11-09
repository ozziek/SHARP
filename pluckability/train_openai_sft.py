import argparse
import json
import logging
import os
import random
import tempfile
import pandas as pd
from typing import cast
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from src._types import SHARPCard
from src._format import format_user_message
from src._helpers import identify_challenging_flashcards, load_jsonl_data
from openai.types.chat import ChatCompletionMessageParam


def _format_assistant_message(row: SHARPCard) -> ChatCompletionMessageParam:
    pluckable = row["pluckable"]
    assert isinstance(pluckable, bool), "Pluckable is not a boolean"
    return {
        "role": "assistant",
        "content": json.dumps(
            {
                "pluckable": pluckable,
            }
        ),
    }


def _analyze_dataset(dataset: Dataset):
    # outline the breakdown
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)
    logging.info("Dataset breakdown: %s", df["pluckable"].value_counts())

    logging.info("Unique source_urls: %s", len(df["source_url"].unique()))


def _balance_dataset(dataset: Dataset, seed: int, max_rows: int | None = None):
    dataset_pluckable = dataset.filter(lambda x: x["pluckable"])
    dataset_unpluckable = dataset.filter(lambda x: not x["pluckable"])

    # choose an equal number of each type of challenging example
    base_length = min(len(dataset_pluckable), len(dataset_unpluckable))
    if max_rows is not None:
        # choose the maximum number of rows for each type of challenging example
        max_length = min(base_length, max_rows)

        def _select_random_row(subset: Dataset):
            # choose a random unique URL to avoid overfitting to a single source
            unique_urls = subset.unique("source_url")
            assert len(unique_urls) > 0, "No unique URLs found"
            url = random.choice(unique_urls)
            row = subset.filter(lambda x: x["source_url"] == url).shuffle(seed=seed).select(range(1))
            assert len(row) > 0, "No row found"

            row = row[0]

            # drop the row by id from the subset
            subset = subset.filter(lambda x: x["id"] != row["id"])

            return row, subset

        rows: list[dict] = []
        while len(rows) < max_length:
            # select one from each row
            pluckable_row, dataset_pluckable = _select_random_row(dataset_pluckable)
            rows.append(pluckable_row)

            unpluckable_row, dataset_unpluckable = _select_random_row(dataset_unpluckable)
            rows.append(unpluckable_row)

        balanced_dataset = Dataset.from_list(rows)

    else:
        # no max rows, so use the base length
        min_length = base_length

        # create a balanced dataset of each type
        balanced_dataset = concatenate_datasets(
            [dataset_pluckable.select(range(min_length)), dataset_unpluckable.select(range(min_length))]
        )

    # ensure there are no duplicate rows
    flashcard_ids = balanced_dataset.unique("id")
    assert isinstance(flashcard_ids, list), "Flashcard IDs is not a list"
    assert len(flashcard_ids) == len(balanced_dataset), "There are duplicate rows in the dataset"

    return balanced_dataset


def build_dataset(dataset: Dataset, base_instruction: str):
    training_completions: list[dict[str, list[ChatCompletionMessageParam]]] = []
    for example in dataset:
        row = cast(SHARPCard, example)
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": base_instruction},
            {"role": "user", "content": format_user_message(row)},
            _format_assistant_message(row),
        ]
        training_completions.append(
            {
                "messages": messages,
            }
        )

    return training_completions


def build_training_dataset(
    dataset: Dataset, base_instruction: str, challenge: bool, seed: int, max_rows: int | None = None
):
    assert seed is not None and isinstance(seed, int), "Seed is not an integer"
    random.seed(seed)

    logging.info("Dataset: %s", dataset)

    # identifying challenging examples
    if challenge:
        # filter the dataset for only challenging exampels (examples the model gets wrong consistently)
        result_set = load_jsonl_data("./pluckability/results")

        # filter for only the zero-shot and multi-shot results
        result_set = {k: v for k, v in result_set.items() if "zero_shot" in k or "multi_shot" in k}

        logging.info("Identifying challenging examples from: %s", result_set.keys())
        challenging_flashcard_ids = identify_challenging_flashcards(result_set)

        challenging_examples = dataset.filter(lambda x: x["id"] in challenging_flashcard_ids)
        challenging_examples = challenging_examples.shuffle(seed=seed)

        examples = _balance_dataset(challenging_examples, max_rows=max_rows, seed=seed)
        del challenging_examples
    else:
        examples = _balance_dataset(dataset.shuffle(seed=seed), max_rows=max_rows, seed=seed)

    _analyze_dataset(examples)

    logging.info("Building training completions")
    training_completions = build_dataset(examples, base_instruction)
    assert isinstance(training_completions, list), "Training completions is not a list"

    return training_completions


def _upload_openai_file(client: OpenAI, data: list[dict]):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as temp_file:
        with open(temp_file.name, "w") as f:
            for completion in data:
                f.write(json.dumps(completion) + "\n")

        logging.info(
            "Uploading training data to OpenAI",
        )
        openai_file = client.files.create(
            file=open(temp_file.name, "rb"),
            purpose="fine-tune",
            expires_after={"anchor": "created_at", "seconds": 2592000},  # 30 days
        )
        logging.info("Uploaded training data to OpenAI: %s", openai_file.id)

    return openai_file


def main(args):
    with open(args.base_instruction_file, "r") as f:
        base_instruction = f.read()

    dataset = load_dataset(args.dataset)
    assert isinstance(dataset, DatasetDict), "Dataset is not a DatasetDict"

    train_completions = build_training_dataset(
        dataset=dataset["train"],
        base_instruction=base_instruction,
        challenge=True,  # use the challenging examples for training
        seed=args.seed,
        max_rows=args.max_rows,
    )
    test_completions = build_training_dataset(
        dataset=dataset["test"],
        base_instruction=base_instruction,
        challenge=False,
        seed=args.seed,
        max_rows=args.max_rows,
    )
    assert isinstance(test_completions, list), "Eval completions is not a list"

    if args.dry_run:
        logging.info("Dry run, skipping upload and fine-tuning job creation")
        exit(0)

    # save the file to a temporary file
    train_file = _upload_openai_file(client, train_completions)
    test_file = _upload_openai_file(client, test_completions)

    # use a batch size of 1 for the online fine-tuning job (this is the default when you create a job, however I decreased the n_epochs to avoid overfitting)
    online_fine_tuning_job = client.fine_tuning.jobs.create(
        model=args.model,
        hyperparameters={
            "batch_size": 1,
            "n_epochs": 1,
        },
        training_file=train_file.id,
        validation_file=test_file.id,
        suffix="sharp-pluckability-online",
    )
    logging.info("Created online fine-tuning job: %s", online_fine_tuning_job.model_dump_json())

    # use a batch size of 4 for the offline fine-tuning job
    fine_tuning_job = client.fine_tuning.jobs.create(
        model=args.model,
        hyperparameters={
            "batch_size": 4,
        },
        training_file=train_file.id,
        validation_file=test_file.id,
        suffix="sharp-pluckability",
    )
    logging.info("Created online fine-tuning job: %s", fine_tuning_job.model_dump_json())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    load_dotenv()
    assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"
    client = OpenAI()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ozziek/SHARP-Card")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base-instruction-file", type=str, required=True)
    parser.add_argument("--max-rows", type=int, default=64, help="Maximum number of rows to use for training")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed used for sampling and shuffling the dataset"
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not upload the training data to OpenAI")
    args = parser.parse_args()

    main(args)
