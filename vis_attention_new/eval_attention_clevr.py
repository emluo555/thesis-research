"""
eval_attention_clevr.py - Run causal attention analysis over a CLEVR dataset.

Loads a CLEVR questions JSON, iterates over each item, and runs the full
attention analysis pipeline (from demo_attention.py) per item.  Models are
loaded once and reused across all items.

Image directory is derived from the JSON path following CLEVR conventions:
    <dataset_root>/images/<type>/
where <type> comes from the filename (e.g. CLEVR_valA_questions.json -> valA).

Usage:
    python eval_attention_clevr.py \\
        --dataset_path /path/to/CLEVR_valA_questions.json \\
        --save_dir ../results/clevr_attention \\
        --model_instruct /path/to/Qwen3-VL-8B-Instruct \\
        --data_range "0,5" \\
        --local_files_only \\
        --image_attn_ratio_by_layer
"""

import csv
import gc
import os
import json
import random
import argparse
import traceback
import torch
import tqdm

from demo_attention import load_model_and_processor, prepare_inputs, run_analysis
from attention_map import compute_per_token_attention_ratios


# ------------------------------------------------------------------
# Dataset helpers
# ------------------------------------------------------------------

def load_dataset(dataset_path, random_n=None, data_range=None):
    """Load CLEVR questions JSON and optionally subsample."""
    with open(dataset_path, "r") as f:
        data_json = json.load(f)

    dataset = data_json["questions"]
    if random_n is not None:
        return random.sample(dataset, random_n)
    if data_range is not None:
        start, end = data_range
        return dataset[start:end]
    return dataset


def derive_image_dir(dataset_path):
    """Derive the CLEVR image directory from the questions JSON path.

    CLEVR_valA_questions.json  ->  ../images/valA/
    """
    dataset_root = os.path.abspath(
        os.path.join(os.path.dirname(dataset_path), "..")
    )
    dataset_type = os.path.basename(dataset_path).split("_")[1]
    return os.path.join(dataset_root, "images", dataset_type)


# ------------------------------------------------------------------
# Plot toggle helpers (mirrors demo_attention._build_enabled_plots)
# ------------------------------------------------------------------

ALL_PLOT_NAMES = [
    "causal_bar", "image_heatmap", "image_heatmap_gif", "layer_head_grid",
    "rollout_to_input", "attn_ratios", "top_attended", "reasoning_phase",
    "attention_matrix", "entropy_by_layer", "head_specialization",
    "image_attn_ratio_by_layer",
    "entropy_by_layer_comparison", "image_attn_ratio_comparison",
    "entropy_over_generation", "rollout_comparison", "report",
]


def build_enabled_plots(args):
    """Return the set of enabled plot names, or None for all.

    Returns ``set()`` (empty) when --no_plots is given, which disables all
    plotting and directory creation.  Returns ``None`` when no individual
    plot flags are given, meaning all plots are enabled.
    """
    if getattr(args, "no_plots", False):
        return set()
    selected = {name for name in ALL_PLOT_NAMES if getattr(args, name, False)}
    return selected or None


# ------------------------------------------------------------------
# Memory / resume helpers
# ------------------------------------------------------------------

def _flush_memory():
    """Force-free Python garbage and CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _item_already_done(item_dir, model_label, no_plots=False,
                       prev_keys=None):
    """Check whether this item+model was already processed.

    With plots: checks for .png files in *item_dir/model_label/*.
    Without plots (--no_plots): checks if the item key is present in
    *prev_keys* (loaded from a previous consolidated JSON).
    """
    if no_plots:
        item_key = os.path.basename(item_dir)
        return prev_keys is not None and item_key in prev_keys
    out_dir = os.path.join(item_dir, model_label)
    return os.path.isdir(out_dir) and any(
        f.endswith(".png") for f in os.listdir(out_dir)
    )


def _load_previous_keys(save_dir, model_label):
    """Load item keys from a previously written metrics JSON for resume."""
    path = os.path.join(save_dir, f"metrics_{model_label}.json")
    if not os.path.isfile(path):
        return set()
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return {k for k, v in data.items() if "error" not in v}
    except (json.JSONDecodeError, KeyError):
        return set()


# ------------------------------------------------------------------
# Per-item record helpers
# ------------------------------------------------------------------

def _build_item_record(item, prompt, json_metrics=None,
                       per_token_data=None, error=None):
    """Build a consolidated record for one dataset item.

    Only per_answer_token_metrics goes into the JSON (small).
    all_per_token_metrics is written to CSV separately for thinking models.
    """
    record = {
        "prompt": prompt,
        "correct_answer": item.get("answer", ""),
        "generated_answer": "",
        "thinking_trace": "",
    }
    if json_metrics is not None:
        record["generated_answer"] = json_metrics.get("generated_answer", "")
        record["thinking_trace"] = json_metrics.get("thinking_trace", "")
    if per_token_data is not None:
        record["per_answer_token_metrics"] = per_token_data["per_answer_token_metrics"]
    if error is not None:
        record["error"] = error
    return record


def _per_token_to_csv_rows(item_key, per_token_data):
    """Convert all_per_token_metrics into long-format CSV rows.

    Yields tuples of (item, token_idx, token, layer,
    image_attn, prompt_attn, reasoning_attn).
    """
    m = per_token_data["all_per_token_metrics"]
    tokens = m["generated_tokens"]
    image = m["image_attn_per_token"]
    prompt = m["prompt_attn_per_token"]
    reasoning = m["reasoning_attn_per_token"]
    has_reasoning = len(reasoning) > 0

    for t_idx, token_str in enumerate(tokens):
        num_layers = len(image[t_idx])
        for layer in range(num_layers):
            yield (
                item_key,
                t_idx,
                token_str,
                layer,
                f"{image[t_idx][layer]:.6f}",
                f"{prompt[t_idx][layer]:.6f}",
                f"{reasoning[t_idx][layer]:.6f}" if has_reasoning else "",
            )


# ------------------------------------------------------------------
# Prompt template
# ------------------------------------------------------------------

DEFAULT_PROMPT_TEMPLATE = (
    "{question} Please respond with ONE word or number, you must wrap "
    "your final answer with <answer> and </answer> tags."
)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run causal attention analysis over a CLEVR dataset.",
    )

    # ---- Dataset arguments ----
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to CLEVR questions JSON.")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Root output directory for results.")
    parser.add_argument("--data_range", type=str, default=None,
                        help='Subset range as "start,end" (e.g. "0,10").')
    parser.add_argument("--random_n", type=int, default=None,
                        help="Randomly sample N items from the dataset.")
    parser.add_argument("--prompt_template", type=str,
                        default=DEFAULT_PROMPT_TEMPLATE,
                        help="Prompt template with {question} placeholder.")

    # ---- Model arguments ----
    parser.add_argument("--model_instruct", type=str,
                        default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Instruct (non-reasoning) model path.")
    parser.add_argument("--model_thinking", type=str, default="",
                        help="Thinking (reasoning) model path. Empty to skip.")

    # ---- Analysis arguments ----
    parser.add_argument("--mode", type=str, default="both",
                        choices=["deep", "summary", "both"],
                        help="Analysis mode: deep, summary, or both.")
    parser.add_argument("--token_idx", type=int, default=-1,
                        help="Token index for deep inspection (-1 = last).")
    parser.add_argument("--layers", type=str, default="",
                        help="Comma-separated layer indices (e.g. 0,8,16,24,35). "
                             "Empty = all layers.")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum tokens to generate per item.")
    parser.add_argument("--local_files_only", action="store_true",
                        help="Only use locally cached models.")

    # ---- Model-type selection flags ----
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--instruct_only", action="store_true",
                             help="Run only the Instruct model.")
    model_group.add_argument("--thinking_only", action="store_true",
                             help="Run only the Thinking model.")

    # ---- Per-plot toggle flags (same as demo_attention.py) ----
    parser.add_argument("--no_plots", action="store_true",
                        help="Disable all visualizations and per-item folders. "
                             "Only metrics JSON and CSV are produced.")
    plot_group = parser.add_argument_group(
        "Plot toggles",
        "If NONE given, ALL plots are generated. If ANY given, ONLY those.",
    )
    plot_group.add_argument("--causal_bar", action="store_true")
    plot_group.add_argument("--image_heatmap", action="store_true")
    plot_group.add_argument("--image_heatmap_gif", action="store_true")
    plot_group.add_argument("--layer_head_grid", action="store_true")
    plot_group.add_argument("--rollout_to_input", action="store_true")
    plot_group.add_argument("--attn_ratios", action="store_true")
    plot_group.add_argument("--top_attended", action="store_true")
    plot_group.add_argument("--reasoning_phase", action="store_true")
    plot_group.add_argument("--attention_matrix", action="store_true")
    plot_group.add_argument("--entropy_by_layer", action="store_true")
    plot_group.add_argument("--head_specialization", action="store_true")
    plot_group.add_argument("--image_attn_ratio_by_layer", action="store_true")
    plot_group.add_argument("--entropy_by_layer_comparison", action="store_true")
    plot_group.add_argument("--image_attn_ratio_comparison", action="store_true")
    plot_group.add_argument("--entropy_over_generation", action="store_true")
    plot_group.add_argument("--rollout_comparison", action="store_true")
    plot_group.add_argument("--report", action="store_true")

    args = parser.parse_args()

    # ---- Parse dataset subset ----
    data_range = None
    if args.data_range:
        start, end = map(int, args.data_range.split(","))
        data_range = (start, end)

    data = load_dataset(args.dataset_path, random_n=args.random_n,
                        data_range=data_range)
    img_dir = derive_image_dir(args.dataset_path)

    # Offset so item labels match original dataset indices (e.g. range 6,11
    # produces item_6 .. item_10 rather than item_0 .. item_4).
    data_start = data_range[0] if data_range is not None else 0

    print(f"[Dataset] {len(data)} items from {args.dataset_path}")
    print(f"[Dataset] Image dir: {img_dir}")
    if data_start:
        print(f"[Dataset] Index offset: {data_start}")

    # ---- Parse layers ----
    selected_layers = None
    if args.layers:
        selected_layers = [int(x.strip()) for x in args.layers.split(",")]
        print(f"[Config] Selected layers: {selected_layers}")

    no_plots = getattr(args, "no_plots", False)
    enabled_plots = build_enabled_plots(args)
    if no_plots:
        print("[Config] All plots disabled (--no_plots)")
    elif enabled_plots is not None:
        print(f"[Config] Enabled plots: {sorted(enabled_plots)}")
    else:
        print("[Config] All plots enabled (no individual plot flags given)")

    run_instruct = bool(args.model_instruct) and not args.thinking_only
    run_thinking = bool(args.model_thinking) and not args.instruct_only

    os.makedirs(args.save_dir, exist_ok=True)

    failed_items: list[tuple[int, str]] = []

    # Consolidated results keyed by model label, accumulated during the loops
    # and written to a single metrics.json at the end.
    consolidated: dict[str, dict] = {}

    # ---- Run Instruct model over all items ----
    if run_instruct:
        print(f"\n[Instruct] Loading model: {args.model_instruct}")
        model, processor = load_model_and_processor(
            args.model_instruct, args.local_files_only,
        )

        prev_instruct = _load_previous_keys(args.save_dir, "instruct") if no_plots else None
        instruct_results = {}
        for idx, item in enumerate(tqdm.tqdm(data, desc="Instruct")):
            real_idx = data_start + idx
            item_key = f"item_{real_idx}"
            item_dir = os.path.join(args.save_dir, item_key)
            prompt = args.prompt_template.format(question=item["question"])

            if _item_already_done(item_dir, "instruct", no_plots=no_plots,
                                  prev_keys=prev_instruct):
                print(f"[Instruct] Skipping item {real_idx} (already processed)")
                instruct_results[item_key] = _build_item_record(item, prompt)
                continue

            result = inputs = image = json_metrics = per_token_data = None
            try:
                image_path = os.path.join(img_dir, item["image_filename"])

                inputs, image = prepare_inputs(processor, image_path, prompt)
                result, _, _, _, json_metrics = run_analysis(
                    model, processor, inputs, image,
                    model_label="instruct",
                    output_dir=item_dir,
                    mode=args.mode,
                    token_idx=args.token_idx,
                    selected_layers=selected_layers,
                    max_new_tokens=args.max_new_tokens,
                    enabled_plots=enabled_plots,
                    save_metrics_json=False,
                )
                per_token_data = compute_per_token_attention_ratios(result)
                instruct_results[item_key] = _build_item_record(
                    item, prompt, json_metrics, per_token_data,
                )
            except Exception as exc:
                print(f"[Instruct] ERROR on item {real_idx}: {exc}")
                traceback.print_exc()
                failed_items.append((real_idx, f"instruct: {exc}"))
                instruct_results[item_key] = _build_item_record(
                    item, prompt, error=str(exc),
                )
            finally:
                del result, inputs, image, json_metrics, per_token_data
                _flush_memory()

        consolidated["instruct"] = instruct_results

        del model, processor
        _flush_memory()
        print("[Memory] Instruct model unloaded, CUDA cache cleared.\n")

    # ---- Run Thinking model over all items ----
    if run_thinking:
        print(f"\n[Thinking] Loading model: {args.model_thinking}")
        model, processor = load_model_and_processor(
            args.model_thinking, args.local_files_only,
        )

        prev_thinking = _load_previous_keys(args.save_dir, "thinking") if no_plots else None
        thinking_results = {}
        csv_path = os.path.join(args.save_dir, "all_token_attn_thinking.csv")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "item", "token_idx", "token", "layer",
            "image_attn", "prompt_attn", "reasoning_attn",
        ])

        for idx, item in enumerate(tqdm.tqdm(data, desc="Thinking")):
            real_idx = data_start + idx
            item_key = f"item_{real_idx}"
            item_dir = os.path.join(args.save_dir, item_key)
            prompt = args.prompt_template.format(question=item["question"])

            if _item_already_done(item_dir, "thinking", no_plots=no_plots,
                                  prev_keys=prev_thinking):
                print(f"[Thinking] Skipping item {real_idx} (already processed)")
                thinking_results[item_key] = _build_item_record(item, prompt)
                continue

            result = inputs = image = json_metrics = per_token_data = None
            try:
                image_path = os.path.join(img_dir, item["image_filename"])

                inputs, image = prepare_inputs(
                    processor, image_path, prompt, enable_thinking=True,
                )
                result, _, _, _, json_metrics = run_analysis(
                    model, processor, inputs, image,
                    model_label="thinking",
                    output_dir=item_dir,
                    mode=args.mode,
                    token_idx=args.token_idx,
                    selected_layers=selected_layers,
                    max_new_tokens=args.max_new_tokens,
                    enabled_plots=enabled_plots,
                    save_metrics_json=False,
                )
                per_token_data = compute_per_token_attention_ratios(result)
                thinking_results[item_key] = _build_item_record(
                    item, prompt, json_metrics, per_token_data,
                )
                csv_writer.writerows(
                    _per_token_to_csv_rows(item_key, per_token_data)
                )
            except Exception as exc:
                print(f"[Thinking] ERROR on item {real_idx}: {exc}")
                traceback.print_exc()
                failed_items.append((real_idx, f"thinking: {exc}"))
                thinking_results[item_key] = _build_item_record(
                    item, prompt, error=str(exc),
                )
            finally:
                del result, inputs, image, json_metrics, per_token_data
                _flush_memory()

        csv_file.close()
        print(f"[Saved] {csv_path}")

        consolidated["thinking"] = thinking_results

        del model, processor
        _flush_memory()
        print("[Memory] Thinking model unloaded, CUDA cache cleared.\n")

    # ---- Write consolidated metrics.json ----
    for model_label, results_dict in consolidated.items():
        out_path = os.path.join(args.save_dir, f"metrics_{model_label}.json")
        with open(out_path, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"[Saved] {out_path}  ({len(results_dict)} items)")

    # ---- Summary ----
    if failed_items:
        print(f"\n[Warning] {len(failed_items)} item(s) failed:")
        for item_idx, reason in failed_items:
            print(f"  item {item_idx}: {reason}")

    print(f"\n[Done] Results for {len(data)} items saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
