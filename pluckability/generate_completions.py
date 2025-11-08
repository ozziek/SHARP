import argparse
import asyncio
from dataclasses import dataclass
import json
import logging
import os
from typing import Any, Tuple, cast
from openai import AsyncOpenAI, Omit
from datasets import DatasetDict, load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from src._format import format_user_message
from src._types import SHARPCard


@dataclass
class _SamplingParams:
    model: str
    temperature: float | Omit
    system_instruction: str


async def _call_openai_judge(
    client: AsyncOpenAI, row: SHARPCard, params: _SamplingParams
) -> Tuple[bool | None, str]:
    """Call OpenAI API to judge pluckability. Returns (pluckable, reason)."""
    try:
        response = await client.chat.completions.create(
            model=params.model,
            messages=[
                {"role": "system", "content": params.system_instruction},
                {"role": "user", "content": format_user_message(row)},
            ],
            temperature=params.temperature,
            response_format={"type": "json_object"},
        )

        result_text = response.choices[0].message.content
        if result_text is None:
            return None, "Empty response from API"
        result_text = result_text.strip()

        # Parse JSON response
        try:
            result_json = json.loads(result_text)
            pluckable = result_json.get("pluckable", False)
            reason = result_json.get("reason", "No reason provided")

            # Ensure pluckable is boolean
            if isinstance(pluckable, str):
                pluckable = pluckable.lower() == "true"

            assert isinstance(pluckable, bool), "Pluckable is not a boolean"
            assert isinstance(reason, str), "Reason is not a string"

            return bool(pluckable), str(reason)

        except json.JSONDecodeError:
            logging.warning(f"Failed to parse JSON response: {result_text}")
            return None, "JSON parsing error"
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return None, f"API error: {str(e)}"


async def evaluate_pluckability(
    client: AsyncOpenAI, dataset: DatasetDict, params: _SamplingParams, max_concurrency: int = 10
):
    """Evaluate pluckability using LLM judge with parallel processing."""

    results = []
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_example(split_name: str, i: int, example: dict) -> dict[str, Any]:
        """Process a single example with concurrency control."""
        async with semaphore:
            assert isinstance(example, dict)
            row = cast(SHARPCard, example)

            judge_prediction, judge_reason = await _call_openai_judge(client, row, params)

            # Always include the row in output, even if judge_prediction is None
            if judge_prediction is None:
                logging.warning(f"Got None prediction for example {i} from {split_name}: {judge_reason}")
                correct = None  # Can't determine correctness with None prediction
            else:
                correct = judge_prediction == row["pluckable"]

            return {
                "split": split_name,
                "index": i,
                **row,
                "judge_prediction": judge_prediction,
                "judge_reason": judge_reason,
                "correct": correct,
            }

    for split_name in dataset.keys():
        split_data = dataset[split_name]

        logging.info(
            f"Evaluating {len(split_data)} examples from {split_name} split with max concurrency: {max_concurrency}"
        )

        # Create tasks for all examples in this split
        tasks = [process_example(str(split_name), i, dict(example)) for i, example in enumerate(split_data)]

        # Process all tasks with progress tracking
        with tqdm(total=len(tasks), desc=f"Evaluating {split_name}") as pbar:
            # Process in batches to update progress
            batch_size = max_concurrency * 2  # Process in batches of 2x concurrency limit
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i : i + batch_size]
                batch_results = await asyncio.gather(*batch_tasks)
                # Include all results (even those with None predictions) in the output
                results.extend(batch_results)
                pbar.update(len(batch_tasks))

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ozziek/SHARP-Card")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--instruction-file", type=str, required=True)
    parser.add_argument("--max-concurrency", type=int, default=10)
    args = parser.parse_args()

    instruction_name = args.instruction_file.split("/")[-1].split(".")[0]
    assert instruction_name.strip() != "", "Instruction name is empty"
    logging.info(f"Using instruction name: {instruction_name}")

    with open(args.instruction_file, "r") as f:
        system_instruction = f.read()

    assert system_instruction.strip() != "", "System instruction file is empty"

    # Use NOT_GIVEN if temperature not specified
    temperature = args.temperature if args.temperature is not None else Omit()

    sampling_params = _SamplingParams(
        model=args.model,
        temperature=temperature,
        system_instruction=system_instruction,
    )
    logging.info(f"Samplig with model: {args.model} and temperature: {temperature}")

    load_dotenv()
    assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"

    client = AsyncOpenAI()

    logging.info(f"Loading dataset from {args.dataset}")
    dataset = load_dataset(args.dataset)
    assert isinstance(dataset, DatasetDict), "Dataset is not a DatasetDict"

    results = asyncio.run(
        evaluate_pluckability(client, dataset, sampling_params, max_concurrency=args.max_concurrency)
    )

    # Build output filename
    temp_suffix = f"_{args.temperature}" if args.temperature is not None else ""
    output_file = f"results_{instruction_name}_{args.model}{temp_suffix}.jsonl"

    # check if results directory exists
    results_dir = "./pluckability/results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    output_file = os.path.join(results_dir, output_file)

    logging.info(f"Writing results to {output_file}")
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
