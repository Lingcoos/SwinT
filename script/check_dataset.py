import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect dataset splits, class balance, and sample visuals."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("./datasets/Butterfly-100"),
        help="Dataset root containing train/valid/test subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output/dataset"),
        help="Directory to store plots and CSV summaries.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of classes to highlight in the class-count bar charts.",
    )
    parser.add_argument(
        "--classes-in-grid",
        type=int,
        default=4,
        help="How many random classes to visualize per split.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=3,
        help="Number of sample images to show per class in the visualization grid.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def collect_split_stats(split_dir: Path) -> pd.DataFrame:
    rows = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        count = sum(1 for img_path in class_dir.iterdir() if img_path.is_file())
        rows.append({"class": class_dir.name, "count": count})
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"[WARN] No images found under {split_dir}.")
    return df


def summarize_split(df: pd.DataFrame, split: str) -> dict:
    if df.empty:
        return {}
    counts = df["count"].astype(float)
    return {
        "split": split,
        "num_classes": int(len(df)),
        "num_images": int(counts.sum()),
        "min_per_class": int(counts.min()),
        "max_per_class": int(counts.max()),
        "median_per_class": float(counts.median()),
        "mean_per_class": float(counts.mean()),
        "std_per_class": float(counts.std(ddof=0)),
        "imbalance_ratio": float(counts.max() / max(counts.min(), 1.0)),
    }


def plot_class_bar(df: pd.DataFrame, split: str, output_dir: Path, top_k: int) -> None:
    if df.empty:
        return
    display_df = df.sort_values("count", ascending=False).head(top_k)
    fig_height = max(4, 0.35 * len(display_df))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(display_df["class"], display_df["count"], color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("Images")
    ax.set_title(f"Top {len(display_df)} classes in {split}")
    fig.tight_layout()
    bar_path = output_dir / f"{split}_top_{len(display_df)}_classes.png"
    fig.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bar chart to {bar_path}")


def plot_histogram(df: pd.DataFrame, split: str, output_dir: Path) -> None:
    if df.empty:
        return
    bins = min(25, max(5, df["count"].nunique()))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["count"], bins=bins, color="darkorange", edgecolor="black")
    ax.set_title(f"{split.title()} class-count histogram")
    ax.set_xlabel("Images per class")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    hist_path = output_dir / f"{split}_hist.png"
    fig.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved histogram to {hist_path}")


def plot_sample_grid(
    dataset_root: Path,
    split: str,
    output_dir: Path,
    classes_in_grid: int,
    samples_per_class: int,
    rng: random.Random,
) -> None:
    split_dir = dataset_root / split
    class_dirs = [d for d in sorted(split_dir.iterdir()) if d.is_dir()]
    if not class_dirs:
        return

    selected = rng.sample(class_dirs, min(classes_in_grid, len(class_dirs)))
    n_rows = len(selected)
    n_cols = samples_per_class
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = np.atleast_2d(axes)

    for row, class_dir in enumerate(selected):
        images = [p for p in class_dir.iterdir() if p.is_file()]
        if not images:
            for col in range(n_cols):
                axes[row, col].axis("off")
            continue
        chosen = rng.sample(images, min(n_cols, len(images)))
        for col in range(n_cols):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(chosen):
                continue
            try:
                with Image.open(chosen[col]) as img:
                    ax.imshow(img.convert("RGB"))
            except OSError:
                continue
            if col == 0:
                ax.set_title(class_dir.name, fontsize=10)

    fig.suptitle(f"Random samples from {split}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    grid_path = output_dir / f"{split}_sample_grid.png"
    fig.savefig(grid_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sample grid to {grid_path}")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    split_dfs = []
    summaries = []
    for split in ["train", "valid", "test"]:
        split_dir = args.root / split
        if not split_dir.exists():
            print(f"[WARN] Missing split directory: {split_dir}. Skipping.")
            continue
        df = collect_split_stats(split_dir)
        if df.empty:
            continue
        df["split"] = split
        split_dfs.append(df)
        summary = summarize_split(df, split)
        if summary:
            summaries.append(summary)
            print(summary)
        csv_path = args.output_dir / f"{split}_class_counts.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved class counts to {csv_path}")
        plot_class_bar(df, split, args.output_dir, args.top_k)
        plot_histogram(df, split, args.output_dir)
        plot_sample_grid(
            args.root,
            split,
            args.output_dir,
            args.classes_in_grid,
            args.samples_per_class,
            rng,
        )

    if split_dfs:
        combined_df = pd.concat(split_dfs, ignore_index=True)
        combined_csv = args.output_dir / "all_splits_class_counts.csv"
        combined_df.to_csv(combined_csv, index=False)
        print(f"Saved combined class counts to {combined_csv}")

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_csv = args.output_dir / "split_level_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"Saved split-level summary to {summary_csv}")
        print("\nSplit overview:\n", summary_df)


if __name__ == "__main__":
    main()
