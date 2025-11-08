#!/usr/bin/env python3
"""
Analyze pluckability evaluation results from JSONL files.

This script scans the pluckability/results folder for JSONL files and
displays performance metrics including accuracy, precision, recall, and F1.

Usage:
    python3 analyze.py
"""

import os
import warnings
from typing import Dict

import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from _helpers import load_jsonl_data

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def calculate_metrics(df: pd.DataFrame) -> tuple[float, float, float, float]:
    """Calculate accuracy, precision, recall, and F1 score."""
    # Filter out rows where judge_prediction is None
    df_filtered = df[~pd.isna(df["judge_prediction"])].copy()

    if len(df_filtered) == 0:
        return 0.0, 0.0, 0.0, 0.0

    y_true = df_filtered["pluckable"].astype(int)
    y_pred = df_filtered["judge_prediction"].astype(int)

    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division="warn")
        recall = recall_score(y_true, y_pred, zero_division="warn")
        f1 = f1_score(y_true, y_pred, zero_division="warn")
    except Exception:
        accuracy = precision = recall = f1 = 0.0

    return accuracy, precision, recall, f1


def create_overall_performance_summary(data: Dict[str, pd.DataFrame], console: Console) -> None:
    """Create Overall Performance Summary table."""
    console.print("\n[bold blue]📊 Overall Performance Summary[/bold blue]")

    table = Table(
        title="Overall Performance Summary",
        show_header=True,
        header_style="bold magenta",
        box=None,
        padding=(0, 1),
        title_justify="left",
    )

    table.add_column("Run", style="cyan", no_wrap=True)
    table.add_column("n", justify="right", style="green", no_wrap=True)
    table.add_column("accuracy", justify="right", style="yellow", no_wrap=True)
    table.add_column("precision", justify="right", style="yellow", no_wrap=True)
    table.add_column("recall", justify="right", style="yellow", no_wrap=True)
    table.add_column("f1", justify="right", style="yellow", no_wrap=True)

    for run_name, df in data.items():
        # Filter out None predictions for counting
        df_filtered = df[~pd.isna(df["judge_prediction"])]
        total_n = len(df_filtered)

        accuracy, precision, recall, f1 = calculate_metrics(df)

        table.add_row(
            run_name,
            str(total_n),
            f"{accuracy:.3f}",
            f"{precision:.3f}",
            f"{recall:.3f}",
            f"{f1:.3f}",
        )

    console.print(table)


def create_detailed_breakdown(data: Dict[str, pd.DataFrame], console: Console) -> None:
    """Create Detailed Breakdown by Run and Split."""
    console.print("\n[bold blue]🔍 Detailed Breakdown (Run × Split)[/bold blue]")

    table = Table(
        title="Detailed Breakdown (Run × Split)",
        show_header=True,
        header_style="bold magenta",
        box=None,
        padding=(0, 1),
        title_justify="left",
    )

    table.add_column("Run", style="cyan", no_wrap=True)
    table.add_column("Split", style="orange3", no_wrap=True)
    table.add_column("n", justify="right", style="green", no_wrap=True)
    table.add_column("accuracy", justify="right", style="yellow", no_wrap=True)
    table.add_column("precision", justify="right", style="yellow", no_wrap=True)
    table.add_column("recall", justify="right", style="yellow", no_wrap=True)
    table.add_column("f1", justify="right", style="yellow", no_wrap=True)

    for run_name, df in data.items():
        if "split" in df.columns:
            splits = sorted(df["split"].unique())

            for split in splits:
                split_df = df[df["split"] == split].copy()
                # Filter out None predictions
                split_df_filtered = split_df[~pd.isna(split_df["judge_prediction"])]

                if len(split_df_filtered) > 0:
                    total = len(split_df_filtered)
                    accuracy, precision, recall, f1 = calculate_metrics(pd.DataFrame(split_df))

                    table.add_row(
                        run_name,
                        split,
                        str(total),
                        f"{accuracy:.3f}",
                        f"{precision:.3f}",
                        f"{recall:.3f}",
                        f"{f1:.3f}",
                    )
        else:
            # If no split column, show as single entry
            df_filtered = df[~pd.isna(df["judge_prediction"])]
            if len(df_filtered) > 0:
                total = len(df_filtered)
                accuracy, precision, recall, f1 = calculate_metrics(df)

                table.add_row(
                    run_name,
                    "all",
                    str(total),
                    f"{accuracy:.3f}",
                    f"{precision:.3f}",
                    f"{recall:.3f}",
                    f"{f1:.3f}",
                )

    console.print(table)


def main():
    """Main analysis function."""
    console = Console()

    # Get the results directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")

    if not os.path.exists(results_dir):
        console.print(f"[red]Error: Results directory not found at {results_dir}[/red]")
        return

    console.print(f"[blue]Loading JSONL files from {results_dir}[/blue]")

    # Load all JSONL data
    data = load_jsonl_data(results_dir)

    if not data:
        console.print("[red]No JSONL files found in results directory[/red]")
        return

    console.print(f"[green]Loaded {len(data)} JSONL files[/green]")

    # Create summary tables
    create_overall_performance_summary(data, console)
    create_detailed_breakdown(data, console)

    console.print("\n[bold green]✅ Analysis complete![/bold green]")


if __name__ == "__main__":
    main()
