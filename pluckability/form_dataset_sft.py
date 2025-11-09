from argparse import ArgumentParser
from collections import defaultdict
import json
import random
from typing import cast
from datasets import Dataset, DatasetDict, load_dataset
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from src._format import format_user_message
from src._types import SHARPCard


def _balance_dataset(dataset: Dataset, seed: int):
    """Balances the dataset by randomly selecting one row from each unique source URL, round robin."""

    pluckable = 0
    pluckable_dataset_by_url: dict[str, list[dict]] = defaultdict(list)
    for row in dataset.filter(lambda x: x["pluckable"]):
        pluckable += 1
        assert isinstance(row, dict), "Row is not a SHARPCard"
        source_url = row["source_url"]
        assert isinstance(source_url, str), "Source URL is not a string"
        pluckable_dataset_by_url[source_url].append(row)

    unpluckable = 0
    unpluckable_dataset_by_url: dict[str, list[dict]] = defaultdict(list)
    for row in dataset.filter(lambda x: not x["pluckable"]):
        unpluckable += 1
        assert isinstance(row, dict), "Row is not a SHARPCard"
        source_url = row["source_url"]
        assert isinstance(source_url, str), "Source URL is not a string"
        unpluckable_dataset_by_url[source_url].append(row)

    def _select_random_row_mutating(subset: dict[str, list[dict]]):
        # choose a random unique URL to avoid overfitting to a single source
        key = random.choice(list(subset.keys()))

        rows = subset[key]
        assert len(rows) > 0, "No rows found"

        # choose a random index from the list
        index = random.randint(0, len(rows) - 1)
        row = rows[index]
        # remove the index from the list
        rows.pop(index)

        if len(rows) == 0:
            # remove the key from the dictionary
            subset.pop(key)

        return row

    rows: list[dict] = []
    while len(rows) < min(pluckable, unpluckable):
        # choose one random row from each dataset, round robin
        pluckable_row = _select_random_row_mutating(pluckable_dataset_by_url)
        rows.append(pluckable_row)

        unpluckable_row = _select_random_row_mutating(unpluckable_dataset_by_url)
        rows.append(unpluckable_row)

    dataset = Dataset.from_list(rows)
    dataset = dataset.shuffle(seed=seed)

    # sanity check that there are no duplicate rows
    flashcard_ids = dataset.unique("id")
    assert isinstance(flashcard_ids, list), "Flashcard IDs is not a list"
    assert len(flashcard_ids) == len(dataset), "There are duplicate rows in the dataset"

    return dataset


def _build_messages(row: SHARPCard, base_instruction: str) -> list[ChatCompletionMessageParam]:
    pluckable = row["pluckable"]
    assert isinstance(pluckable, bool), "Pluckable is not a boolean"

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": base_instruction},
        {"role": "user", "content": format_user_message(row)},
        {
            "role": "assistant",
            # single JSON object with the pluckable value
            "content": json.dumps(
                {
                    "pluckable": pluckable,
                }
            ),
        },
    ]
    return messages


def build_dataset(dataset: Dataset, base_instruction: str) -> list[dict[str, list[ChatCompletionMessageParam]]]:
    balanced = _balance_dataset(dataset, seed=42)
    rows = []
    for row in balanced:
        row = cast(SHARPCard, row)
        messages = _build_messages(row, base_instruction)
        rows.append(
            {
                "messages": messages,
            }
        )
    return rows


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ozziek/SHARP-Card")
    parser.add_argument("--base_instruction", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("-o", type=str, required=True)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.base_instruction, "r") as f:
        base_instruction = f.read()

    sft_dataset = load_dataset(args.dataset)
    assert isinstance(sft_dataset, DatasetDict), "Dataset is not a DatasetDict"
    assert args.split in sft_dataset, f"{args.split} dataset is not in the dataset"

    assert args.o.endswith(".jsonl"), "Output file must end with .jsonl"

    sft_dataset = build_dataset(sft_dataset[args.split], base_instruction)
    assert isinstance(sft_dataset, list), "SFT dataset is not a list"
    with open(args.o, "w") as f:
        for completion in sft_dataset:
            f.write(json.dumps(completion) + "\n")
