import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from src._helpers import load_jsonl_data


def _create_heatmap(df: pd.DataFrame, split: str, output_dir: str, rows_per_chunk: int = 400):
    """Creates a heatmap where each run_name is a column and each row is a specific flashcard id. The cells are colored according to whether the judge prediction is correct or incorrect."""

    # Pivot the data: rows=flashcard id, columns=run_name, values=correct
    pivot_df = df.pivot_table(
        index="id",
        columns="run_name",
        values="correct",
        aggfunc="first",  # Use first in case of duplicates
    )

    # Convert boolean to int for better visualization (1=correct, 0=incorrect)
    pivot_df = pivot_df.astype(int)

    # Split into chunks
    num_rows = len(pivot_df)
    num_chunks = (num_rows + rows_per_chunk - 1) // rows_per_chunk  # Ceiling division

    # Create figure with subplots side by side
    fig, axes = plt.subplots(1, num_chunks, figsize=(12 * num_chunks, 10))

    # Handle case where there's only one chunk
    if num_chunks == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        start_idx = i * rows_per_chunk
        end_idx = min((i + 1) * rows_per_chunk, num_rows)
        chunk_df = pivot_df.iloc[start_idx:end_idx]

        # Create heatmap for this chunk
        sns.heatmap(
            chunk_df,
            cmap=["#ff6b6b", "#51cf66"],  # Red for incorrect (0), Green for correct (1)
            cbar=True,  # Show colorbar on all subplots for consistent coloring
            cbar_kws={"label": "Correct", "ticks": [0.25, 0.75]},
            yticklabels=False,  # Don't show flashcard IDs (too many)
            xticklabels=True,
            linewidths=0.5,
            linecolor="white",
            vmin=0,
            vmax=1,
            ax=ax,
        )

        # Customize colorbar labels
        if ax.collections:
            cbar = ax.collections[0].colorbar
            if cbar:
                cbar.set_ticklabels(["Incorrect", "Correct"])

        # Add subtitle for each chunk
        ax.set_title(f"IDs {start_idx + 1}-{end_idx}", fontsize=10, pad=10)
        ax.set_xlabel("Run Name", fontsize=10)
        if i == 0:
            ax.set_ylabel("Flashcard ID", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=9)

    # Overall title
    fig.suptitle(
        f"Flashcard Performance by Run - {split.capitalize()} Split (n={num_rows})", fontsize=16, y=0.98
    )
    plt.tight_layout()

    # Add spacing between subplots
    plt.subplots_adjust(wspace=0.3)

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"heatmap_{split}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved heatmap to {output_path} ({num_chunks} columns, {num_rows} total flashcards)")
    plt.close()


def main(results_dir: str, output_dir: str = "./pluckability/viz"):
    results = load_jsonl_data(results_dir)

    for run_name, df in results.items():
        df["run_name"] = run_name

    df = pd.concat(results.values())

    for split, split_df in df.groupby("split"):
        _create_heatmap(split_df, str(split), output_dir)


if __name__ == "__main__":
    main(results_dir="./pluckability/results")
