"""
visualize_csv.py - Generate attention and TAM visualizations from CSV data.

Reads all_token_data_*.csv files produced by eval_clevr.py and generates
visualizations using pre-existing functions from vis_attention_new and vis_tam,
without needing a model loaded.

Modes:
  --item N:     Per-item plots for item N
  --item all:   Per-item plots for every item
  (no --item):  Aggregate layer-level comparison plots (instruct vs thinking)

Usage:
    python visualize_csv.py \\
        --instruct_csv path/to/all_token_data_instruct.csv \\
        --thinking_csv path/to/all_token_data_thinking.csv \\
        --output_dir path/to/output/ \\
        --item 0

    python visualize_csv.py \\
        --instruct_csv path/to/all_token_data_instruct.csv \\
        --thinking_csv path/to/all_token_data_thinking.csv \\
        --output_dir path/to/output/
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "vis_attention_new"))
sys.path.insert(0, os.path.join(REPO_ROOT, "vis_tam"))

import matplotlib
matplotlib.use("Agg")

from attention_vis import (
    plot_attention_ratio_over_generation,
    plot_entropy_over_generation,
    plot_entropy_by_layer,
    plot_image_attention_ratio_by_layer,
)
from tam_vis import (
    plot_activation_ratio_over_generation,
    plot_activation_entropy_over_generation,
    plot_activation_entropy_by_layer,
    plot_image_activation_ratio_by_layer,
)
from grade_answers import (
    load_attention_csv,
    plot_attention_over_generation,
)


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    """Load a token-level CSV, handling both old and new column formats."""
    df = pd.read_csv(path)

    # Older CSVs used attn_reasoning / tam_reasoning instead of
    # attn_generated / tam_generated and had no system columns.
    renames = {}
    if "attn_reasoning" in df.columns and "attn_generated" not in df.columns:
        renames["attn_reasoning"] = "attn_generated"
    if "tam_reasoning" in df.columns and "tam_generated" not in df.columns:
        renames["tam_reasoning"] = "tam_generated"
    if renames:
        df = df.rename(columns=renames)

    for col in ("attn_system", "tam_system", "attn_generated", "tam_generated"):
        if col not in df.columns:
            df[col] = 0.0

    return df


def get_item_keys(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique item keys in the DataFrame."""
    return sorted(df["item"].unique(), key=lambda x: int(x.split("_")[1]))


# ------------------------------------------------------------------
# Summary / metrics construction from CSV data
# ------------------------------------------------------------------

def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy (nats) of a discrete distribution."""
    p = probs[probs > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def build_attention_summary(df_item: pd.DataFrame) -> list[dict]:
    """Build summary dicts (one per generated token) for attention plots.

    Each dict: {token_idx, token_str, image_attn_ratio, prompt_attn_ratio,
                generated_attn_ratio, entropy}
    """
    summary = []
    for t_idx, grp in df_item.groupby("token_idx", sort=True):
        img = grp["attn_image"].mean()
        prompt = grp["attn_prompt"].mean()
        gen = grp["attn_generated"].mean()
        total = img + prompt + gen + 1e-12
        img_r, prompt_r, gen_r = img / total, prompt / total, gen / total
        summary.append({
            "token_idx": int(t_idx),
            "token_str": str(grp["token"].iloc[0]),
            "image_attn_ratio": float(img_r),
            "prompt_attn_ratio": float(prompt_r),
            "generated_attn_ratio": float(gen_r),
            "entropy": _entropy(np.array([img_r, prompt_r, gen_r])),
        })
    return summary


def build_tam_summary(df_item: pd.DataFrame) -> list[dict]:
    """Build summary dicts (one per generated token) for TAM plots.

    Each dict: {token_idx, token_str, image_activation_ratio,
                prompt_activation_ratio, generated_activation_ratio, entropy}
    """
    summary = []
    for t_idx, grp in df_item.groupby("token_idx", sort=True):
        img = grp["tam_image"].mean()
        prompt = grp["tam_prompt"].mean()
        gen = grp["tam_generated"].mean()
        total = img + prompt + gen + 1e-12
        img_r, prompt_r, gen_r = img / total, prompt / total, gen / total
        summary.append({
            "token_idx": int(t_idx),
            "token_str": str(grp["token"].iloc[0]),
            "image_activation_ratio": float(img_r),
            "prompt_activation_ratio": float(prompt_r),
            "generated_activation_ratio": float(gen_r),
            "entropy": _entropy(np.array([img_r, prompt_r, gen_r])),
        })
    return summary


def build_attention_metrics(df: pd.DataFrame) -> dict:
    """Build per-layer metrics dict for attention from a full DataFrame.

    Returns: {entropy_per_layer: (L,), image_attn_ratio_per_layer: (L,)}
    """
    entropy_list = []
    image_ratio_list = []
    for _, grp in df.groupby("layer", sort=True):
        img = grp["attn_image"].values
        prompt = grp["attn_prompt"].values
        gen = grp["attn_generated"].values
        total = img + prompt + gen + 1e-12
        ratios = np.stack([img / total, prompt / total, gen / total], axis=1)
        image_ratio_list.append(float(ratios[:, 0].mean()))
        ent = np.array([_entropy(ratios[i]) for i in range(len(ratios))])
        entropy_list.append(float(ent.mean()))
    return {
        "entropy_per_layer": np.array(entropy_list),
        "image_attn_ratio_per_layer": np.array(image_ratio_list),
    }


def build_tam_metrics(df: pd.DataFrame) -> dict:
    """Build per-layer metrics dict for TAM from a full DataFrame.

    Returns: {entropy_per_layer: (L,), image_activation_ratio_per_layer: (L,)}
    """
    entropy_list = []
    image_ratio_list = []
    for _, grp in df.groupby("layer", sort=True):
        img = grp["tam_image"].values
        prompt = grp["tam_prompt"].values
        gen = grp["tam_generated"].values
        total = img + prompt + gen + 1e-12
        ratios = np.stack([img / total, prompt / total, gen / total], axis=1)
        image_ratio_list.append(float(ratios[:, 0].mean()))
        ent = np.array([_entropy(ratios[i]) for i in range(len(ratios))])
        entropy_list.append(float(ent.mean()))
    return {
        "entropy_per_layer": np.array(entropy_list),
        "image_activation_ratio_per_layer": np.array(image_ratio_list),
    }


# ------------------------------------------------------------------
# Plot generation
# ------------------------------------------------------------------

def generate_per_item_plots(
    df_instruct: pd.DataFrame | None,
    df_thinking: pd.DataFrame | None,
    item_key: str,
    output_dir: str,
    instruct_csv_path: str | None = None,
    thinking_csv_path: str | None = None,
):
    """Generate all per-item plots for a single item."""
    item_dir = os.path.join(output_dir, item_key)
    os.makedirs(item_dir, exist_ok=True)
    item_index = int(item_key.split("_")[1])

    sub_inst = sub_think = None
    if df_instruct is not None:
        sub_inst = df_instruct[df_instruct["item"] == item_key]
        if sub_inst.empty:
            sub_inst = None
    if df_thinking is not None:
        sub_think = df_thinking[df_thinking["item"] == item_key]
        if sub_think.empty:
            sub_think = None

    if sub_inst is None and sub_think is None:
        print(f"[Skip] No data for {item_key}")
        return

    # -- Attention ratio stacked area (per model) --
    if sub_inst is not None:
        plot_attention_ratio_over_generation(
            build_attention_summary(sub_inst),
            title=f"{item_key} (Instruct)",
            save_path=os.path.join(item_dir, "attn_ratio_instruct.png"),
        )
    if sub_think is not None:
        plot_attention_ratio_over_generation(
            build_attention_summary(sub_think),
            title=f"{item_key} (Thinking)",
            save_path=os.path.join(item_dir, "attn_ratio_thinking.png"),
        )

    # -- TAM ratio stacked area (per model) --
    if sub_inst is not None:
        plot_activation_ratio_over_generation(
            build_tam_summary(sub_inst),
            title=f"{item_key} (Instruct)",
            save_path=os.path.join(item_dir, "tam_ratio_instruct.png"),
        )
    if sub_think is not None:
        plot_activation_ratio_over_generation(
            build_tam_summary(sub_think),
            title=f"{item_key} (Thinking)",
            save_path=os.path.join(item_dir, "tam_ratio_thinking.png"),
        )

    # -- Attention entropy over generation (comparative) --
    s_inst = build_attention_summary(sub_inst) if sub_inst is not None else None
    s_think = build_attention_summary(sub_think) if sub_think is not None else None
    if s_inst or s_think:
        plot_entropy_over_generation(
            s_inst, s_think, title=item_key,
            save_path=os.path.join(item_dir, "attn_entropy_generation.png"),
        )

    # -- TAM entropy over generation (comparative) --
    ts_inst = build_tam_summary(sub_inst) if sub_inst is not None else None
    ts_think = build_tam_summary(sub_think) if sub_think is not None else None
    if ts_inst or ts_think:
        plot_activation_entropy_over_generation(
            ts_inst, ts_think, title=item_key,
            save_path=os.path.join(item_dir, "tam_entropy_generation.png"),
        )

    # -- Raw attention + TAM line plot (grade_answers style) --
    if instruct_csv_path and sub_inst is not None:
        try:
            attn_data = load_attention_csv(instruct_csv_path, item_index)
            plot_attention_over_generation(
                attn_data, item_index,
                save_path=os.path.join(item_dir, "raw_attn_tam_instruct.png"),
            )
        except (ValueError, KeyError):
            pass
    if thinking_csv_path and sub_think is not None:
        try:
            attn_data = load_attention_csv(thinking_csv_path, item_index)
            plot_attention_over_generation(
                attn_data, item_index,
                save_path=os.path.join(item_dir, "raw_attn_tam_thinking.png"),
            )
        except (ValueError, KeyError):
            pass

    print(f"[Done] {item_key} -> {item_dir}")


def generate_aggregate_plots(
    df_instruct: pd.DataFrame | None,
    df_thinking: pd.DataFrame | None,
    output_dir: str,
):
    """Generate aggregate layer-level comparison plots across all items."""
    agg_dir = os.path.join(output_dir, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)

    m_inst_attn = m_inst_tam = None
    m_think_attn = m_think_tam = None

    if df_instruct is not None and not df_instruct.empty:
        m_inst_attn = build_attention_metrics(df_instruct)
        m_inst_tam = build_tam_metrics(df_instruct)
    if df_thinking is not None and not df_thinking.empty:
        m_think_attn = build_attention_metrics(df_thinking)
        m_think_tam = build_tam_metrics(df_thinking)

    plot_entropy_by_layer(
        m_inst_attn, m_think_attn,
        title="All Items (Aggregate)",
        save_path=os.path.join(agg_dir, "attn_entropy_by_layer.png"),
    )
    plot_image_attention_ratio_by_layer(
        m_inst_attn, m_think_attn,
        title="All Items (Aggregate)",
        save_path=os.path.join(agg_dir, "attn_image_ratio_by_layer.png"),
    )
    plot_activation_entropy_by_layer(
        m_inst_tam, m_think_tam,
        title="All Items (Aggregate)",
        save_path=os.path.join(agg_dir, "tam_entropy_by_layer.png"),
    )
    plot_image_activation_ratio_by_layer(
        m_inst_tam, m_think_tam,
        title="All Items (Aggregate)",
        save_path=os.path.join(agg_dir, "tam_image_ratio_by_layer.png"),
    )

    print(f"[Done] Aggregate plots -> {agg_dir}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate attention and TAM visualizations from CSV data.",
    )
    parser.add_argument("--instruct_csv", type=str, default="",
                        help="Path to all_token_data_instruct.csv")
    parser.add_argument("--thinking_csv", type=str, default="",
                        help="Path to all_token_data_thinking.csv")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for output plots.")
    parser.add_argument("--item", type=str, default=None,
                        help='Item index (int) or "all". '
                             "Omit for aggregate-only plots.")
    args = parser.parse_args()

    if not args.instruct_csv and not args.thinking_csv:
        parser.error("Provide at least one of --instruct_csv or --thinking_csv.")

    df_instruct = load_csv(args.instruct_csv) if args.instruct_csv else None
    df_thinking = load_csv(args.thinking_csv) if args.thinking_csv else None

    os.makedirs(args.output_dir, exist_ok=True)

    if args.item is not None:
        # Per-item mode
        if args.item.lower() == "all":
            items = set()
            if df_instruct is not None:
                items.update(get_item_keys(df_instruct))
            if df_thinking is not None:
                items.update(get_item_keys(df_thinking))
            items = sorted(items, key=lambda x: int(x.split("_")[1]))
        else:
            items = [f"item_{args.item}"]

        for item_key in items:
            generate_per_item_plots(
                df_instruct, df_thinking, item_key, args.output_dir,
                instruct_csv_path=args.instruct_csv or None,
                thinking_csv_path=args.thinking_csv or None,
            )
    else:
        # Aggregate mode
        generate_aggregate_plots(df_instruct, df_thinking, args.output_dir)

    print(f"\n[All done] Output in {args.output_dir}")


if __name__ == "__main__":
    main()
