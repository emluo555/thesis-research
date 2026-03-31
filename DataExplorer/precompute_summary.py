#!/usr/bin/env python3
"""Generate summary_metrics.json for DataExplorer Summary mode."""

import json
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RESULTS_CSV_8B = os.path.join(os.path.dirname(__file__), "..", "DriVQA", "outputs", "trial_8_full_10", "results.csv")
RESULTS_CSV_30B = os.path.join(os.path.dirname(__file__), "..", "DriVQA", "outputs", "trial_4_full_30", "results.csv")
CORRECTNESS_JSON = os.path.join(os.path.dirname(__file__), "..", "DriVQA", "outputs", "cot_attribution_full", "correctness.json")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "data", "summary_metrics.json")

MODELS = ["instruct", "thinking", "instruct_30b", "thinking_30b", "center_bias", "dino"]
METRICS = ["CC", "KLD", "NSS", "AUC_Judd"]
METRIC_INFO = {
    "CC":       {"higher_better": True,  "range": "[-1, 1]"},
    "KLD":      {"higher_better": False, "range": "[0, inf)"},
    "NSS":      {"higher_better": True,  "range": "(-inf, inf)"},
    "AUC_Judd": {"higher_better": True,  "range": "[0, 1]"},
}

WILCOXON_PAIRS = [
    ("thinking_vs_instruct",         "thinking",      "instruct"),
    ("thinking_30b_vs_instruct_30b", "thinking_30b",  "instruct_30b"),
    ("thinking_30b_vs_thinking",     "thinking_30b",  "thinking"),
    ("instruct_30b_vs_instruct",     "instruct_30b",  "instruct"),
    ("thinking_vs_center_bias",      "thinking",      "center_bias"),
    ("instruct_vs_center_bias",      "instruct",      "center_bias"),
    ("thinking_vs_dino",             "thinking",      "dino"),
    ("instruct_vs_dino",             "instruct",      "dino"),
]


def main():
    df_8b = pd.read_csv(RESULTS_CSV_8B)

    # Load 30B CSV and rename models to *_30b (keep only instruct/thinking rows)
    df_30b = pd.read_csv(RESULTS_CSV_30B)
    df_30b = df_30b[df_30b["model"].isin(["instruct", "thinking"])].copy()
    df_30b["model"] = df_30b["model"].map({"instruct": "instruct_30b", "thinking": "thinking_30b"})

    df = pd.concat([df_8b, df_30b], ignore_index=True)

    with open(CORRECTNESS_JSON) as f:
        correctness = json.load(f)

    # Build per-question order (preserve CSV order for unique image_ids)
    question_order = list(dict.fromkeys(zip(df["image_id"], df["question"])))

    # Overall stats
    overall = {}
    for model in MODELS:
        mdf = df[df["model"] == model]
        overall[model] = {}
        for metric in METRICS:
            vals = mdf[metric].values
            overall[model][metric] = {
                "mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals, ddof=1)), 4),
            }

    # Wilcoxon signed-rank tests
    wilcoxon_results = {}
    for label, model_a, model_b in WILCOXON_PAIRS:
        wilcoxon_results[label] = {}
        for metric in METRICS:
            vals_a = df[df["model"] == model_a].sort_values("image_id")[metric].values
            vals_b = df[df["model"] == model_b].sort_values("image_id")[metric].values
            try:
                stat, p = wilcoxon(vals_a, vals_b)
                wilcoxon_results[label][metric] = {
                    "W": round(float(stat), 4),
                    "p": round(float(p), 6),
                }
            except Exception:
                wilcoxon_results[label][metric] = {"W": None, "p": None}

    # Per-question
    per_question = []
    for image_id, question in question_order:
        entry = {
            "image_id": image_id,
            "question": question,
            "correctness": {
                m: correctness.get(m, {}).get(image_id)
                for m in MODELS if m not in ("center_bias", "dino")
            },
            "metrics": {},
        }
        for model in MODELS:
            row = df[(df["image_id"] == image_id) & (df["model"] == model)]
            if len(row) == 1:
                entry["metrics"][model] = {
                    metric: round(float(row[metric].values[0]), 4)
                    for metric in METRICS
                }
        per_question.append(entry)

    output = {
        "models": MODELS,
        "metrics": METRICS,
        "metric_info": METRIC_INFO,
        "overall": overall,
        "wilcoxon": wilcoxon_results,
        "per_question": per_question,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {OUTPUT_JSON} ({len(per_question)} questions)")


if __name__ == "__main__":
    main()
