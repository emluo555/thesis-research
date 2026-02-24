"""
grade_answers.py - Grade model-generated answers against correct answers using OpenAI.

Reads one or both metrics_*.json files produced by eval_attention_clevr.py and uses
the OpenAI API to semantically judge whether each generated answer is correct.

Usage:
    python grade_answers.py \\
        --instruct_path /path/to/metrics_instruct.json \\
        --thinking_path /path/to/metrics_thinking.json \\
        --output_path /path/to/correctness_results.json

    # Single mode:
    python grade_answers.py --instruct_path /path/to/metrics_instruct.json

    # Examine only items 0-99:
    python grade_answers.py \\
        --instruct_path /path/to/metrics_instruct.json \\
        --data_range "0,100"

Output:
    correctness_results.json with per-mode accuracy stats, lists of incorrect item
    indices, and a cross-mode comparison block (when both modes are provided).
"""

import argparse
import asyncio
import csv
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone


try:
    from openai import AsyncOpenAI
except ImportError:
    print("[Error] openai package not found. Install with: pip install openai")
    sys.exit(1)


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def load_metrics(path: str, data_range: tuple[int, int] | None = None) -> dict:
    """Load a metrics JSON, optionally filtering to items within [start, end)."""
    with open(path, "r") as f:
        data = json.load(f)
    if data_range is None:
        return data
    start, end = data_range
    return {
        k: v for k, v in data.items()
        if start <= extract_item_index(k) < end
    }


def extract_item_index(item_key: str) -> int:
    """Parse the integer index from a key like 'item_42'."""
    m = re.search(r"(\d+)$", item_key)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse index from item key: {item_key!r}")


# ------------------------------------------------------------------
# OpenAI grading
# ------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are evaluating whether a model's answer to a visual question is correct. "
    "Consider semantic equivalence: 'one' and '1' are the same, 'metallic' and "
    "'metal' are the same, 'grey' and 'gray' are the same, etc. "
    "Reply with only 'yes' or 'no'."
)

USER_PROMPT_TEMPLATE = (
    "Correct answer: {correct}\n"
    "Model answer: {generated}\n"
    "Is the model answer semantically correct?"
)


async def grade_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    item_key: str,
    correct: str,
    generated: str,
) -> tuple[str, bool]:
    """Return (item_key, is_correct). Retries once on transient errors."""
    user_msg = USER_PROMPT_TEMPLATE.format(correct=correct, generated=generated)
    for attempt in range(2):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=5,
                    temperature=0,
                )
            answer = response.choices[0].message.content.strip().lower()
            return item_key, answer.startswith("yes")
        except Exception as exc:
            if attempt == 0:
                await asyncio.sleep(2)
            else:
                print(f"[Warning] OpenAI error for {item_key}: {exc}. Marking incorrect.")
                return item_key, False


async def grade_all(
    client: AsyncOpenAI,
    tasks: list[tuple[str, str, str]],
    model: str,
    concurrency: int,
) -> dict[str, bool]:
    """Grade a list of (item_key, correct_answer, generated_answer) tuples.

    Returns a dict mapping item_key -> is_correct.
    """
    semaphore = asyncio.Semaphore(concurrency)
    jobs = [
        grade_one(client, semaphore, model, key, correct, generated)
        for key, correct, generated in tasks
    ]
    results = {}
    total = len(jobs)
    done = 0
    for coro in asyncio.as_completed(jobs):
        item_key, is_correct = await coro
        results[item_key] = is_correct
        done += 1
        if done % 10 == 0 or done == total:
            print(f"  graded {done}/{total}", end="\r", flush=True)
    print()
    return results


# ------------------------------------------------------------------
# Per-mode processing
# ------------------------------------------------------------------

def process_mode(data: dict, detect_truncated: bool = False) -> tuple[list, list, list]:
    """Split items into (to_grade, skipped, truncated).

    to_grade:  list of (item_key, correct_answer, generated_answer)
    skipped:   list of item_key strings for items with empty generated_answer
    truncated: list of item_key strings where thinking_trace is empty but
               generated_answer is non-empty (model ran out of tokens during
               thinking and never produced a final answer). Only populated
               when detect_truncated=True.
    """
    to_grade = []
    skipped = []
    truncated = []
    for item_key, record in data.items():
        if isinstance(record, dict) and record.get("generated_answer", "") != "":
            to_grade.append((
                item_key,
                record.get("correct_answer", ""),
                record["generated_answer"],
            ))
            if detect_truncated and record.get("thinking_trace", "") == "":
                truncated.append(item_key)
        else:
            skipped.append(item_key)
    return to_grade, skipped, truncated


def build_mode_block(
    data: dict,
    graded: dict[str, bool],
    skipped_keys: list[str],
    truncated_keys: list[str] | None = None,
) -> dict:
    """Build the per-mode output block."""
    total = len(data)
    skipped_indices = sorted(extract_item_index(k) for k in skipped_keys)
    evaluated = total - len(skipped_keys)

    incorrect_keys = [k for k, correct in graded.items() if not correct]
    correct_count = sum(1 for v in graded.values() if v)
    incorrect_count = len(incorrect_keys)

    accuracy = correct_count / evaluated if evaluated > 0 else 0.0
    incorrect_indices = sorted(extract_item_index(k) for k in incorrect_keys)

    block = {
        "total_items": total,
        "evaluated_items": evaluated,
        "skipped_items": skipped_indices,
        "accuracy": round(accuracy, 6),
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "incorrect_items": incorrect_indices,
    }

    if truncated_keys is not None:
        truncated_indices = sorted(extract_item_index(k) for k in truncated_keys)
        block["truncated_thinking_items"] = truncated_indices
        block["truncated_thinking_count"] = len(truncated_indices)

    return block


def build_comparison_block(
    instruct_graded: dict[str, bool],
    thinking_graded: dict[str, bool],
) -> dict:
    """Build cross-mode comparison block using only items graded in both modes."""
    common_keys = set(instruct_graded) & set(thinking_graded)

    both_incorrect = sorted(
        extract_item_index(k)
        for k in common_keys
        if not instruct_graded[k] and not thinking_graded[k]
    )
    instruct_only_incorrect = sorted(
        extract_item_index(k)
        for k in common_keys
        if not instruct_graded[k] and thinking_graded[k]
    )
    thinking_only_incorrect = sorted(
        extract_item_index(k)
        for k in common_keys
        if instruct_graded[k] and not thinking_graded[k]
    )

    return {
        "both_incorrect_items": both_incorrect,
        "instruct_only_incorrect_items": instruct_only_incorrect,
        "thinking_only_incorrect_items": thinking_only_incorrect,
    }


_ATTN_CATEGORIES = ("system", "image", "prompt", "generated")

# ------------------------------------------------------------------
# Attention analysis for individual items
# ------------------------------------------------------------------

_CSV_ATTN_COLS = [f"attn_{c}" for c in _ATTN_CATEGORIES]
_CSV_TAM_COLS = [f"tam_{c}" for c in _ATTN_CATEGORIES]
_CSV_ALL_COLS = _CSV_ATTN_COLS + _CSV_TAM_COLS


def load_attention_csv(csv_path: str, item_index: int) -> dict:
    """Load attention data for a single item from a token-level CSV.

    Returns a dict with keys:
        tokens:        list[str]   – token string per generation step
        attn_system:   list[float] – mean across layers per step
        attn_image:    list[float]
        attn_prompt:   list[float]
        attn_generated:list[float]
        tam_system:    list[float]
        tam_image:     list[float]
        tam_prompt:    list[float]
        tam_generated: list[float]
    """
    item_key = f"item_{item_index}"

    def _make_step():
        d = {"count": 0, "token": ""}
        for c in _CSV_ALL_COLS:
            d[c] = 0.0
        return d

    step_data: dict[int, dict] = defaultdict(_make_step)

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["item"] != item_key:
                continue
            t_idx = int(row["token_idx"])
            d = step_data[t_idx]
            d["token"] = row["token"]
            for col in _CSV_ALL_COLS:
                d[col] += float(row[col]) if row.get(col) else 0.0
            d["count"] += 1

    if not step_data:
        raise ValueError(f"No data found for {item_key} in {csv_path}")

    sorted_steps = sorted(step_data.keys())
    result: dict = {"tokens": [step_data[s]["token"] for s in sorted_steps]}
    for col in _CSV_ALL_COLS:
        result[col] = [step_data[s][col] / step_data[s]["count"] for s in sorted_steps]
    return result


def summarize_attention(attn_data: dict) -> dict:
    """Compute overall averages across all generation steps."""
    n = len(attn_data["attn_image"])
    summary: dict = {"num_tokens": n}
    for col in _CSV_ALL_COLS:
        summary[f"mean_{col}"] = sum(attn_data[col]) / n if n else 0.0
    return summary


def plot_attention_over_generation(attn_data: dict, item_index: int,
                                   save_path: str | None = None):
    """Plot attention and TAM across generation steps (two subplots)."""
    import matplotlib.pyplot as plt

    steps = list(range(len(attn_data["attn_image"])))

    fig, (ax_attn, ax_tam) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for cat in _ATTN_CATEGORIES:
        ax_attn.plot(steps, attn_data[f"attn_{cat}"], label=cat.title(), linewidth=1.2)
    ax_attn.set_ylabel("Raw Attention (mean across layers)")
    ax_attn.set_title(f"Attention Over Generation — item_{item_index}")
    ax_attn.legend()
    ax_attn.grid(True, alpha=0.3)

    for cat in _ATTN_CATEGORIES:
        ax_tam.plot(steps, attn_data[f"tam_{cat}"], label=cat.title(), linewidth=1.2)
    ax_tam.set_xlabel("Generation Step (token index)")
    ax_tam.set_ylabel("TAM (mean across layers)")
    ax_tam.set_title(f"TAM Over Generation — item_{item_index}")
    ax_tam.legend()
    ax_tam.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[Saved] {save_path}")
    else:
        plt.show()
    plt.close(fig)


def analyze_item_attention(csv_path: str, item_index: int,
                           save_path: str | None = None) -> dict:
    """Load, summarize, and optionally plot attention for a single item.

    Returns the summary dict with mean attention values.
    """
    attn_data = load_attention_csv(csv_path, item_index)
    summary = summarize_attention(attn_data)
    print(f"[item_{item_index}] {summary['num_tokens']} generation steps")
    for col in _CSV_ALL_COLS:
        print(f"  mean {col:20s}: {summary[f'mean_{col}']:.6f}")
    plot_attention_over_generation(attn_data, item_index, save_path=save_path)
    return summary


# ------------------------------------------------------------------
# Incorrect-item attention report
# ------------------------------------------------------------------


def _mean_attn_from_json(record: dict) -> dict | None:
    """Extract mean attention values from the attention and tam blocks.

    Averages across all answer tokens and all layers.
    Returns a dict with keys like ``attn_system``, ``attn_image``, …,
    ``tam_system``, ``tam_image``, …  or None if no attention data exists.
    """
    result: dict[str, float] = {}

    for block_key, col_suffix in [("attention", "attn"), ("tam", "tam")]:
        block = record.get(block_key)
        if not block:
            continue
        for cat in _ATTN_CATEGORIES:
            field = f"{cat}_{col_suffix}_per_ans_token"
            rows = block.get(field, [])
            flat = [v for row in rows for v in row]
            result[f"{col_suffix}_{cat}"] = sum(flat) / len(flat) if flat else 0.0

    return result if result else None


def _image_ratio(vals: dict, prefix: str) -> float:
    """Compute image / (system + image + prompt + generated) for a given prefix."""
    total = sum(vals.get(f"{prefix}_{c}", 0.0) for c in _ATTN_CATEGORIES)
    return vals.get(f"{prefix}_image", 0.0) / total if total > 0 else 0.0


_REPORT_ATTN_COLS = [f"attn_{c}" for c in _ATTN_CATEGORIES]
_REPORT_TAM_COLS = [f"tam_{c}" for c in _ATTN_CATEGORIES]
_REPORT_DETAIL_COLS = _REPORT_ATTN_COLS + ["attn_image_ratio"] + _REPORT_TAM_COLS + ["tam_image_ratio"]


def generate_incorrect_attention_csv(
    correctness_path: str,
    instruct_metrics_path: str | None,
    thinking_metrics_path: str | None,
    output_csv_path: str,
) -> None:
    """Build a CSV of attention stats for every incorrect item per mode.

    Per-item columns cover all four attention categories (system, image,
    prompt, generated) for both raw attention and TAM, plus image ratios.
    Aggregate summary rows are appended at the bottom.
    """
    with open(correctness_path, "r") as f:
        correctness = json.load(f)

    metrics_by_mode: dict[str, dict] = {}
    if instruct_metrics_path and "instruct" in correctness:
        with open(instruct_metrics_path, "r") as f:
            metrics_by_mode["instruct"] = json.load(f)
    if thinking_metrics_path and "thinking" in correctness:
        with open(thinking_metrics_path, "r") as f:
            metrics_by_mode["thinking"] = json.load(f)

    rows: list[list] = []
    mode_aggregates: dict[str, dict] = {}

    for mode, metrics_data in metrics_by_mode.items():
        incorrect_indices = correctness[mode].get("incorrect_items", [])
        if not incorrect_indices:
            print(f"[{mode}] No incorrect items.")
            continue

        accum: dict[str, list[float]] = defaultdict(list)

        for idx in sorted(incorrect_indices):
            item_key = f"item_{idx}"
            record = metrics_data.get(item_key)
            if not record:
                print(f"[{mode}] Warning: {item_key} not found in metrics JSON, skipping.")
                continue

            vals = _mean_attn_from_json(record)
            if vals is None:
                print(f"[{mode}] Warning: {item_key} has no attention data, skipping.")
                continue

            attn_ratio = _image_ratio(vals, "attn")
            tam_ratio = _image_ratio(vals, "tam")

            row = [idx, mode]
            for col in _REPORT_ATTN_COLS:
                row.append(f"{vals.get(col, 0.0):.6f}")
            row.append(f"{attn_ratio:.6f}")
            for col in _REPORT_TAM_COLS:
                row.append(f"{vals.get(col, 0.0):.6f}")
            row.append(f"{tam_ratio:.6f}")
            rows.append(row)

            for k, v in vals.items():
                accum[k].append(v)

        if accum:
            agg: dict[str, float] = {
                k: sum(vs) / len(vs) for k, vs in accum.items()
            }
            agg["attn_image_ratio"] = _image_ratio(agg, "attn")
            agg["tam_image_ratio"] = _image_ratio(agg, "tam")
            mode_aggregates[mode] = {"count": len(accum[next(iter(accum))]), **agg}

    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_index", "mode"] + _REPORT_DETAIL_COLS)
        writer.writerows(rows)

        writer.writerow([])
        writer.writerow(["# Aggregate attention stats for all incorrect items per mode"])
        writer.writerow(["mode", "num_incorrect"] + _REPORT_DETAIL_COLS)
        for mode, agg in mode_aggregates.items():
            agg_row = [mode, agg["count"]]
            for col in _REPORT_DETAIL_COLS:
                agg_row.append(f"{agg.get(col, 0.0):.6f}")
            writer.writerow(agg_row)

    print(f"[Saved] {output_csv_path}")
    for mode, agg in mode_aggregates.items():
        print(f"  {mode}: {int(agg['count'])} incorrect items, "
              f"attn_image_ratio={agg['attn_image_ratio']:.4f}, "
              f"tam_image_ratio={agg['tam_image_ratio']:.4f}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> None:
    if not args.instruct_path and not args.thinking_path:
        print("[Error] Provide at least one of --instruct_path or --thinking_path.")
        sys.exit(1)

    client = AsyncOpenAI()  # reads OPENAI_API_KEY from environment

    output: dict = {
        "model": args.model,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    instruct_graded: dict[str, bool] | None = None
    thinking_graded: dict[str, bool] | None = None

    if args.instruct_path:
        print(f"[Instruct] Loading {args.instruct_path}")
        instruct_data = load_metrics(args.instruct_path, args.data_range)
        to_grade, skipped, _ = process_mode(instruct_data)
        print(f"[Instruct] {len(to_grade)} to grade, {len(skipped)} skipped (empty answer)")
        instruct_graded = await grade_all(client, to_grade, args.model, args.concurrency)
        output["instruct"] = build_mode_block(instruct_data, instruct_graded, skipped)
        acc = output["instruct"]["accuracy"]
        print(f"[Instruct] Accuracy: {acc:.2%}  "
              f"({output['instruct']['correct_count']}/{output['instruct']['evaluated_items']})\n")

    if args.thinking_path:
        print(f"[Thinking] Loading {args.thinking_path}")
        thinking_data = load_metrics(args.thinking_path, args.data_range)
        to_grade, skipped, truncated = process_mode(thinking_data, detect_truncated=True)
        print(f"[Thinking] {len(to_grade)} to grade, {len(skipped)} skipped (empty answer), "
              f"{len(truncated)} truncated (no answer produced)")
        thinking_graded = await grade_all(client, to_grade, args.model, args.concurrency)
        output["thinking"] = build_mode_block(
            thinking_data, thinking_graded, skipped, truncated_keys=truncated,
        )
        acc = output["thinking"]["accuracy"]
        print(f"[Thinking] Accuracy: {acc:.2%}  "
              f"({output['thinking']['correct_count']}/{output['thinking']['evaluated_items']})\n")

    if instruct_graded is not None and thinking_graded is not None:
        output["comparison"] = build_comparison_block(instruct_graded, thinking_graded)

    # Determine output path
    if args.output_path:
        out_path = args.output_path
    else:
        first_input = args.instruct_path or args.thinking_path
        out_path = os.path.join(os.path.dirname(os.path.abspath(first_input)),
                                "correctness_results.json")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[Saved] {out_path}")

    # Console summary
    print("\n=== Summary ===")
    for mode in ("instruct", "thinking"):
        if mode in output:
            b = output[mode]
            line = (f"  {mode:10s}  accuracy={b['accuracy']:.2%}  "
                    f"correct={b['correct_count']}  incorrect={b['incorrect_count']}  "
                    f"skipped={len(b['skipped_items'])}")
            if "truncated_thinking_count" in b:
                line += f"  truncated={b['truncated_thinking_count']}"
            print(line)
    if "comparison" in output:
        c = output["comparison"]
        print(f"\n  Both incorrect:            {len(c['both_incorrect_items'])}")
        print(f"  Instruct only incorrect:   {len(c['instruct_only_incorrect_items'])}")
        print(f"  Thinking only incorrect:   {len(c['thinking_only_incorrect_items'])}")

    # Auto-generate incorrect-item attention report
    has_incorrect = any(
        output.get(m, {}).get("incorrect_count", 0) > 0
        for m in ("instruct", "thinking")
    )
    if has_incorrect and (args.instruct_path or args.thinking_path):
        report_csv = os.path.join(
            os.path.dirname(os.path.abspath(out_path)), "incorrect_attention.csv",
        )
        print(f"\n[Report] Auto-generating incorrect-item attention CSV...")
        try:
            generate_incorrect_attention_csv(
                correctness_path=out_path,
                instruct_metrics_path=args.instruct_path,
                thinking_metrics_path=args.thinking_path,
                output_csv_path=report_csv,
            )
        except Exception as exc:
            print(f"[Report] Warning: could not generate attention report: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grade model answers against correct answers using OpenAI.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ---- grade subcommand (default behaviour) ----
    grade_parser = subparsers.add_parser("grade", help="Grade answers using OpenAI.")
    grade_parser.add_argument("--instruct_path", type=str, default="",
                              help="Path to metrics_instruct.json.")
    grade_parser.add_argument("--thinking_path", type=str, default="",
                              help="Path to metrics_thinking.json.")
    grade_parser.add_argument("--output_path", type=str, default="",
                              help="Output JSON path. Defaults to correctness_results.json "
                                   "in the same directory as the first input file.")
    grade_parser.add_argument("--model", type=str, default="gpt-4o-mini",
                              help="OpenAI model to use for grading (default: gpt-4o-mini).")
    grade_parser.add_argument("--concurrency", type=int, default=20,
                              help="Number of concurrent OpenAI API calls (default: 20).")
    grade_parser.add_argument("--data_range", type=str, default="",
                              help='Restrict grading to items within index range "start,end" '
                                   '(exclusive end, e.g. "0,100" grades item_0 .. item_99).')

    # ---- analyze subcommand ----
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze attention for a specific item from the thinking CSV.",
    )
    analyze_parser.add_argument("--csv_path", type=str, required=True,
                                help="Path to all_token_attn_thinking.csv.")
    analyze_parser.add_argument("--item_index", type=int, required=True,
                                help="Item index to analyze (e.g. 42 for item_42).")
    analyze_parser.add_argument("--save_path", type=str, default="",
                                help="Path to save the plot (PNG). If omitted, displays interactively.")

    # ---- report subcommand ----
    report_parser = subparsers.add_parser(
        "report",
        help="Generate CSV of attention stats for all incorrect items.",
    )
    report_parser.add_argument("--correctness_path", type=str, required=True,
                               help="Path to correctness_results.json.")
    report_parser.add_argument("--instruct_metrics", type=str, default="",
                               help="Path to metrics_instruct.json.")
    report_parser.add_argument("--thinking_metrics", type=str, default="",
                               help="Path to metrics_thinking.json.")
    report_parser.add_argument("--output_csv", type=str, default="",
                               help="Output CSV path. Defaults to "
                                    "incorrect_attention.csv next to correctness JSON.")

    # Default to "grade" when no subcommand given (backward compatible)
    if len(sys.argv) > 1 and sys.argv[1] not in (
        "grade", "analyze", "report", "-h", "--help",
    ):
        sys.argv.insert(1, "grade")

    args = parser.parse_args()

    if args.command == "grade":
        args.instruct_path = args.instruct_path or None
        args.thinking_path = args.thinking_path or None
        args.output_path = args.output_path or None
        if args.data_range:
            start, end = map(int, args.data_range.split(","))
            args.data_range = (start, end)
        else:
            args.data_range = None
        asyncio.run(main_async(args))

    elif args.command == "analyze":
        analyze_item_attention(
            csv_path=args.csv_path,
            item_index=args.item_index,
            save_path=args.save_path or None,
        )

    elif args.command == "report":
        output_csv = args.output_csv or os.path.join(
            os.path.dirname(os.path.abspath(args.correctness_path)),
            "incorrect_attention.csv",
        )
        generate_incorrect_attention_csv(
            correctness_path=args.correctness_path,
            instruct_metrics_path=args.instruct_metrics or None,
            thinking_metrics_path=args.thinking_metrics or None,
            output_csv_path=output_csv,
        )


if __name__ == "__main__":
    main()
