import glob
import os
import json
from typing import Dict
import pandas as pd


def extract_run_name(filename: str) -> str:
    """Extract a clean run name from the JSONL filename."""
    basename = os.path.basename(filename)
    # Remove .jsonl extension
    name = basename.replace(".jsonl", "")
    # Remove common prefixes
    name = name.replace("results_", "")
    return name


def load_jsonl_data(results_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all JSONL files from the results directory."""
    jsonl_files = glob.glob(os.path.join(results_dir, "*.jsonl"))
    data = {}

    for jsonl_file in jsonl_files:
        run_name = extract_run_name(jsonl_file)
        try:
            # Read JSONL file
            records = []
            with open(jsonl_file, "r") as f:
                for line in f:
                    records.append(json.loads(line))

            df = pd.DataFrame(records)
            data[run_name] = df
            print(f"Loaded {len(df)} rows from {run_name}")
        except Exception as e:
            print(f"Error loading {jsonl_file}: {e}")

    return data
