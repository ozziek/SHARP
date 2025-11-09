import json
import random
from datasets import Dataset
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from pluckability.src._types import SHARPCard


def _balance_dataset(dataset: Dataset, seed: int):
    """Balances the dataset by randomly selecting one row from each unique source URL, round robin."""
    pluckable_dataset = dataset.filter(lambda x: x["pluckable"])
    unpluckable_dataset = dataset.filter(lambda x: not x["pluckable"])

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
    base_length = min(len(pluckable_dataset), len(unpluckable_dataset))
    while len(rows) < base_length:
        # choose one random row from each dataset, round robin
        pluckable_row, pluckable_dataset = _select_random_row(pluckable_dataset)
        rows.append(pluckable_row)

        unpluckable_row, unpluckable_dataset = _select_random_row(unpluckable_dataset)
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
    sft_dataset = load_dataset(cli_args.dataset)
    assert isinstance(sft_dataset, DatasetDict), "Dataset is not a DatasetDict"
    assert "train" in sft_dataset, "Train dataset is not in the dataset"
