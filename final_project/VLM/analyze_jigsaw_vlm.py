import ast
import re
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# models and CSV paths
MODEL_FILES = {
    "InternVL3-8B": "./puzzle_transparent_results_InternVL3_8B.csv",
    "Qwen3-VL-8B-Instruct": "./puzzle_transparent_results_Qwen3-VL-8B-Instruct.csv",
    "Visual-Jigsaw-Image-7B": "./puzzle_transparent_results_visual_jigsaw_image_7B.csv",
}

K = 9  # 3x3 puzzle

# Parsing utilities
def parse_list_from_str(s: str) -> List[int]:
    """Parse '[0, 1, 2, ...]' or '0,1,2,...' into a list of ints."""
    if pd.isna(s):
        return []
    s = str(s).strip()
    if not s:
        return []
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [int(x) for x in val]
    except Exception:
        pass
    nums = re.findall(r"-?\d+", s)
    return [int(x) for x in nums]


def parse_dict_from_str(s: str) -> Dict[int, int]:
    """Parse '{0: 2, 1: 5, ...}' into dict[int, int]."""
    if pd.isna(s):
        return {}
    s = str(s).strip()
    if not s:
        return {}
    d = ast.literal_eval(s)
    if isinstance(d, dict):
        return {int(k): int(v) for k, v in d.items()}
    return {}


def is_valid_perm(perm: List[int], k: int = K) -> bool:
    """Check if perm is a permutation of 0..k-1."""
    return len(perm) == k and set(perm) == set(range(k))


# Neighbor accuracy
def get_neighbor_position_pairs(n_rows: int = 3, n_cols: int = 3):
    """All horizontal/vertical neighbor pairs (pos_a, pos_b) with pos_a < pos_b."""
    pairs = []
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            if c + 1 < n_cols:           # right
                pairs.append((idx, idx + 1))
            if r + 1 < n_rows:           # down
                pairs.append((idx, idx + n_cols))
    return pairs


NEIGHBOR_PAIRS = get_neighbor_position_pairs()


def build_board_pred(pred_perm: List[int], mapping_new2old: Dict[int, int]) -> List[int]:
    """
    board_pred[pos] = original tile index at position pos.
      - pos: position in original 3x3 grid (0..8)
      - pred_perm[pos]: tile index (0..8) placed here
      - mapping_new2old[tile_idx]: original index of that tile
    """
    board = []
    for pos in range(len(pred_perm)):
        tile_idx = pred_perm[pos]
        orig_idx = mapping_new2old.get(tile_idx, -1)
        board.append(orig_idx)
    return board


def neighbor_accuracy(board_pred: List[int]) -> float:
    """Adjacency accuracy based on unordered neighbor pairs of original indices."""
    if len(board_pred) != K:
        return 0.0

    # ground truth: each position holds its own index
    board_gt = list(range(K))

    def board_to_pair_set(board: List[int]):
        s = set()
        for a, b in NEIGHBOR_PAIRS:
            x, y = board[a], board[b]
            if x == -1 or y == -1:
                continue
            s.add(tuple(sorted((x, y))))
        return s

    gt_pairs = board_to_pair_set(board_gt)
    pred_pairs = board_to_pair_set(board_pred)

    if not gt_pairs:
        return 0.0
    return len(gt_pairs & pred_pairs) / len(gt_pairs)


# Per-model analysis
def analyze_single_model(name: str, path: str) -> pd.DataFrame:
    """
    Load one model's CSV and compute:
      - pred_list, gt_list
      - valid_perm
      - neighbor_acc
    """
    print(f"\n=== Model [{name}] from {path} ===")
    df = pd.read_csv(path)

    df["pred_list"] = df["pred_perm"].apply(parse_list_from_str)
    df["gt_list"] = df["gt_perm"].apply(parse_list_from_str)
    df["mapping_new2old_dict"] = df["mapping_new2old"].apply(parse_dict_from_str)

    df["valid_perm"] = df["pred_list"].apply(is_valid_perm)

    def compute_neighbor(row):
        p = row["pred_list"]
        mapping = row["mapping_new2old_dict"]
        if not is_valid_perm(p):
            return 0.0
        board = build_board_pred(p, mapping)
        return neighbor_accuracy(board)

    df["neighbor_acc"] = df.apply(compute_neighbor, axis=1)
    df["model"] = name

    print(
        f"  n={len(df)}, "
        f"mean acc={df['accuracy'].mean():.3f}±{df['accuracy'].std():.3f}, "
        f"mean neighbor={df['neighbor_acc'].mean():.3f}±{df['neighbor_acc'].std():.3f}, "
        f"valid_perm={df['valid_perm'].mean():.3f}"
    )
    return df


# Piece type performance
def compute_piece_type_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-puzzle accuracy by piece type:
      - Corner / Edge / Interior
    """
    corner_idx = {0, 2, 6, 8}
    edge_idx = {1, 3, 5, 7}
    center_idx = {4}

    rows = []

    for model_name, df_model in df.groupby("model"):
        corner_acc, edge_acc, center_acc = [], [], []

        for _, row in df_model.iterrows():
            pred = row["pred_list"]
            gt = row["gt_list"]
            if not (isinstance(pred, list) and isinstance(gt, list) and len(pred) == K):
                continue

            c = sum(pred[pos] == gt[pos] for pos in corner_idx) / len(corner_idx)
            e = sum(pred[pos] == gt[pos] for pos in edge_idx) / len(edge_idx)
            cent = 1.0 if pred[4] == gt[4] else 0.0

            corner_acc.append(c)
            edge_acc.append(e)
            center_acc.append(cent)

        def add(piece_type: str, acc_list: list):
            arr = np.array(acc_list, dtype=float)
            rows.append(
                {
                    "model": model_name,
                    "piece_type": piece_type,
                    "mean_acc": float(arr.mean()) if len(arr) else np.nan,
                    "std_acc": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "n_puzzles": len(arr),
                }
            )

        add("Corner", corner_acc)
        add("Edge", edge_acc)
        add("Interior", center_acc)

    result = pd.DataFrame(rows)
    print("\n=== Piece type performance (mean ± std) ===")
    print(result.to_string(index=False))
    return result


# Plotting
def plot_model_summary(summary_df: pd.DataFrame, out_path: str = "VLM_accuracy_summary.png"):
    """Bar plot: Direct Accuracy / Neighbor Accuracy / Valid Perm Rate."""
    models = summary_df["model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x - width, summary_df["mean_acc"], width, label="Direct Accuracy")
    bars2 = ax.bar(x, summary_df["mean_neighbor_acc"], width, label="Neighbor Accuracy")
    bars3 = ax.bar(x + width, summary_df["valid_perm_rate"], width, label="Valid Permutation Rate")

    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("VLM Model Accuracy Performance")
    ax.legend()

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def plot_piece_type_bar(piece_df: pd.DataFrame, out_path: str = "VLM_piece_type_performance.png"):
    """Bar plot: accuracy by piece type for each model."""
    piece_types = ["Corner", "Edge", "Interior"]
    models = piece_df["model"].unique().tolist()
    x = np.arange(len(piece_types))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, model in enumerate(models):
        sub = piece_df[piece_df["model"] == model].set_index("piece_type")
        means = [sub.loc[pt, "mean_acc"] for pt in piece_types]
        bars = ax.bar(x + (i - len(models) / 2) * width + width / 2, means, width, label=model)

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(piece_types)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("VLM Accuracy by Piece Type")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


# Main
def main():
    # Load and analyze each model
    all_dfs = []
    summary_rows = []

    for name, path in MODEL_FILES.items():
        df = analyze_single_model(name, path)
        all_dfs.append(df)

        summary_rows.append(
            {
                "model": name,
                "mean_acc": df["accuracy"].mean(),
                "std_acc": df["accuracy"].std(),
                "mean_neighbor_acc": df["neighbor_acc"].mean(),
                "std_neighbor_acc": df["neighbor_acc"].std(),
                "valid_perm_rate": df["valid_perm"].mean(),
            }
        )

    all_df = pd.concat(all_dfs, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    print("\n=== Overall summary (mean ± std) ===")
    print(summary_df.to_string(index=False))

    # Piece-type stats + plots
    piece_df = compute_piece_type_performance(all_df)

    plot_model_summary(summary_df)
    plot_piece_type_bar(piece_df)


if __name__ == "__main__":
    main()