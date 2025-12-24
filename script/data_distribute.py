"""
Generate a bar chart showing per-class counts for the training split.
Saves outputs to `output/dataset/` and prints a short analysis.

Usage:
    python data_distribute.py --csv output/dataset/train_class_counts.csv
    OR
    python data_distribute.py --root datasets/Butterfly-100

"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap


def parse_args():
    p = argparse.ArgumentParser(description="Plot per-class distribution for training split")
    p.add_argument("--csv", type=Path, default=Path("output/dataset/train_class_counts.csv"),
                   help="Path to CSV with columns: class,count (optional). If missing, script will scan dataset root.")
    p.add_argument("--root", type=Path, default=Path("datasets/Butterfly-100"),
                   help="Dataset root with train/valid/test subfolders (used if --csv missing)")
    p.add_argument("--out-dir", type=Path, default=Path("output/dataset"),
                   help="Output directory for plots and summary")
    p.add_argument("--save-name", type=str, default="train_all_classes_bar.png",
                   help="Filename for saved bar chart")
    return p.parse_args()


def read_counts_from_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    # accept variants
    if 'count' not in df.columns:
        raise ValueError(f"CSV {csv_path} missing 'count' column")
    if 'class' not in df.columns and 'label' in df.columns:
        df = df.rename(columns={'label':'class'})
    return df[['class','count']]


def scan_train_folder(root: Path):
    train_dir = root / 'train'
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    rows = []
    for d in sorted(train_dir.iterdir()):
        if d.is_dir():
            n = sum(1 for _ in d.iterdir() if _.is_file())
            rows.append({'class': d.name, 'count': n})
    return pd.DataFrame(rows)


def plot_bar_all_classes(df: pd.DataFrame, out_path: Path):
    df_plot = df.copy()
    # ensure consistent ordering (alphabetical or original), but keep as-is
    labels = df_plot['class'].tolist()
    counts = df_plot['count'].astype(int).tolist()

    n = len(labels)
    # figure size scaled by number of classes
    fig_w = max(12, n * 0.2)
    fig_h = 8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    bars = ax.bar(range(n), counts, color='C0')
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of images')
    ax.set_title('Training set: images per class (all classes)')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # annotate min/max on plot
    max_idx = int(np.argmax(counts))
    min_idx = int(np.argmin(counts))
    ax.annotate(f"max: {counts[max_idx]}", xy=(max_idx, counts[max_idx]), xytext=(max_idx, counts[max_idx]+5),
                ha='center', color='green', fontsize=9)
    ax.annotate(f"min: {counts[min_idx]}", xy=(min_idx, counts[min_idx]), xytext=(min_idx, counts[min_idx]+5),
                ha='center', color='red', fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path


def analyze_and_save(df: pd.DataFrame, out_dir: Path):
    counts = df['count'].astype(int)
    stats = {
        'num_classes': int(len(df)),
        'num_images': int(counts.sum()),
        'min_per_class': int(counts.min()),
        'max_per_class': int(counts.max()),
        'mean_per_class': float(counts.mean()),
        'median_per_class': float(counts.median()),
        'std_per_class': float(counts.std(ddof=0)),
        'imbalance_ratio': float(counts.max() / max(counts.min(), 1)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / 'train_data_distribution_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('Training set per-class distribution summary\n')
        f.write('----------------------------------------\n')
        for k,v in stats.items():
            f.write(f"{k}: {v}\n")

        f.write('\nQuick observations:\n')
        # simple observations
        if stats['std_per_class'] < 0.2 * stats['mean_per_class']:
            f.write('- The dataset is relatively balanced across classes (low relative std).\n')
        else:
            f.write('- Noticeable variance across classes; consider balancing strategies if needed.\n')

        if stats['imbalance_ratio'] > 2.0:
            f.write('- Imbalance ratio > 2.0: some classes have significantly more samples.\n')
        else:
            f.write('- Imbalance ratio is moderate (<2.0).\n')

        # list top 5 and bottom 5 classes
        f.write('\nTop 5 classes by count:\n')
        top5 = df.sort_values('count', ascending=False).head(5)
        for _, row in top5.iterrows():
            f.write(f"  {row['class']}: {int(row['count'])}\n")
        f.write('\nBottom 5 classes by count:\n')
        bot5 = df.sort_values('count', ascending=True).head(5)
        for _, row in bot5.iterrows():
            f.write(f"  {row['class']}: {int(row['count'])}\n")
    return summary_path


def main():
    args = parse_args()
    out_dir = args.out_dir

    if args.csv.exists():
        df = read_counts_from_csv(args.csv)
    else:
        # scan dataset
        try:
            df = scan_train_folder(args.root)
        except Exception as e:
            print('Failed to read CSV or scan dataset root:', e)
            return

    # ensure deterministic order (alphabetical) for plotting if needed
    df = df.sort_values('class').reset_index(drop=True)

    out_path = out_dir / args.save_name
    print('Saving bar chart to', out_path)
    plot_bar_all_classes(df, out_path)

    print('Saving analysis summary...')
    summary_path = analyze_and_save(df, out_dir)
    print('Saved summary to', summary_path)
    print('\nDone. Plots and summary saved to', out_dir)

if __name__ == '__main__':
    main()
