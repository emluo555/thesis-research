"""
analyze_prompt_keyword_attention.py - Prompt keyword attention experiment.

Research question: When a keyword from the question appears in the model's
generated text, does the model attend more to the textual mention (keyword
in the prompt) or to the actual visual object in the image?

For each DriVQA question x model, decomposes attention of keyword-matching
generated tokens into 5 regions:
  1. Keyword prompt tokens
  2. Other prompt tokens
  3. Object bbox image patches (when OVD detection available)
  4. Other image patches
  5. Prior generated tokens

Computes both raw attention % and RAPT-normalized values.
Also reports TAM (Token Attribution Map) decomposition.
Includes Soft IoU metric measuring overlap between attention and object bbox.

References:
  - RAPT metric: Liu et al. "Seeing but Not Believing" (2025), arXiv:2510.17771
  - EAGLE: Chen et al. (CVPR 2026), arXiv:2509.22496
  - Visual Attention Sink: Kang et al. (ICLR 2025), arXiv:2503.03321
  - Soft IoU: Σ(A·M) / [Σ A + Σ M − Σ(A·M)], differentiable IoU for continuous attention

Usage:
    conda activate tam3
    python analyze_prompt_keyword_attention.py [--output_dir results/prompt_keyword_attention]
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import from sibling modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from analyze_bbox_tokens import SYNONYM_MAP, match_token_to_detections, build_box_mask
from scripts.analyze_cot_attribution import load_contributions

# ── Configuration ─────────────────────────────────────────────────────

QUESTION_KEYWORDS = {
    "q0001": ["vehicles"],
    "q0002": ["lane"],
    "q0003": ["pedestrians", "sidewalk"],
    "q0004": ["lane"],
    "q0005": ["street"],
    "q0006": ["turn"],
    "q0007": ["country", "building"],
    "q0008": ["stop"],
    "q0009": ["park", "right"],
    "q0010": ["trees"],
    "q0011": ["restaurant"],
    "q0012": ["snow", "road"],
    "q0013": ["pedestrians"],
    "q0014": ["store", "right"],
    "q0015": ["stop", "pedestrians", "turn", "left"],
    "q0016": ["name", "building"],
    "q0017": ["name", "lane"],
    "q0018": ["park"],
    "q0019": ["park", "right"],
    "q0020": ["pedestrians"],
}

MODEL_KEYS = ["instruct", "thinking", "instruct_30b", "thinking_30b"]
MODEL_LABELS = {
    "instruct": "Instruct 8B",
    "thinking": "Thinking 8B",
    "instruct_30b": "Instruct 30B",
    "thinking_30b": "Thinking 30B",
}

PT_BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "DriVQA", "outputs", "cot_attribution_full",
)

FLOAT_PREC = 6


# ── Helpers ───────────────────────────────────────────────────────────

def get_phase(abs_idx, tr):
    """Return 'thinking' or 'answer' for a generated token index."""
    if tr["think_start"] is not None and tr["think_end"] is not None:
        if tr["think_start"] <= abs_idx <= tr["think_end"]:
            return "thinking"
    if tr["answer_start"] is not None and abs_idx >= tr["answer_start"]:
        return "answer"
    # Instruct model: all generated tokens are "answer"
    if tr["think_start"] is None:
        return "answer"
    return "unknown"


def get_attn_vector(data, gen_idx, layer_idx=None):
    """Get attention vector for generated token gen_idx."""
    attn_rows = data["attn_gen_rows"]  # (L, num_gen, S)
    abs_idx = data["input_len"] + gen_idx
    if gen_idx >= attn_rows.shape[1]:
        return None
    if layer_idx is not None:
        vec = attn_rows[layer_idx, gen_idx, :abs_idx + 1].float().numpy()
    else:
        vec = attn_rows[:, gen_idx, :abs_idx + 1].float().mean(dim=0).numpy()
    return vec


def get_tam_vector(data, gen_idx, layer_idx=None):
    """Get TAM contribution vector for generated token gen_idx."""
    tam_scores = data["tam_scores"]
    if gen_idx >= len(tam_scores):
        return None
    if layer_idx is not None:
        vec = np.array(tam_scores[gen_idx][layer_idx], dtype=np.float64)
    else:
        vec = np.mean(tam_scores[gen_idx], axis=0).astype(np.float64)
    return vec


def decompose_vector(vec, tr, kw_prompt_indices, bbox_mask, abs_idx):
    """Decompose an attention/TAM vector into region sums.

    Returns dict with raw sums for each region.
    """
    total = float(vec.sum())
    if total == 0:
        return {k: 0.0 for k in [
            "keyword_prompt", "other_prompt", "object_bbox",
            "other_image", "full_image", "prior_gen", "system", "total"
        ]}

    img_s, img_e = tr["image_start"], tr["image_end"]
    prompt_s, prompt_e = tr["prompt_start"], tr["prompt_end"]
    gen_s = tr["generation_start"]

    # System: everything before image
    system_sum = float(vec[:img_s].sum()) if img_s > 0 else 0.0

    # Full image
    img_vec = vec[img_s:img_e]
    full_image = float(img_vec.sum())

    # Object bbox vs other image
    if bbox_mask is not None:
        object_bbox = float((img_vec * bbox_mask).sum())
        other_image = full_image - object_bbox
    else:
        object_bbox = float("nan")
        other_image = full_image

    # Prompt: keyword vs other
    # Only count prompt tokens that are within the vector length
    prompt_end_clamp = min(prompt_e, len(vec))
    prompt_vec = vec[prompt_s:prompt_end_clamp] if prompt_s < len(vec) else np.array([])
    kw_sum = 0.0
    for ki in kw_prompt_indices:
        if ki < len(vec):
            kw_sum += float(vec[ki])
    other_prompt = float(prompt_vec.sum()) - kw_sum

    # Prior generated tokens (gen_start to abs_idx, exclusive)
    gen_end = min(abs_idx, len(vec))
    prior_gen = float(vec[gen_s:gen_end].sum()) if gen_s < gen_end else 0.0

    return {
        "keyword_prompt": kw_sum,
        "other_prompt": other_prompt,
        "object_bbox": object_bbox,
        "other_image": other_image,
        "full_image": full_image,
        "prior_gen": prior_gen,
        "system": system_sum,
        "total": total,
    }


def compute_rapt(vec, tr, kw_prompt_indices, bbox_mask, abs_idx):
    """Compute RAPT (Relative Attention Per Token) for each region.

    RAPT = (mean attention in section) / (mean attention across all input tokens).
    Values > 1 mean disproportionately high attention; < 1 means lower than average.
    """
    if vec is None or len(vec) == 0:
        return {}

    img_s, img_e = tr["image_start"], tr["image_end"]
    prompt_s, prompt_e = tr["prompt_start"], tr["prompt_end"]
    gen_s = tr["generation_start"]

    # Mean attention across all input tokens (system + image + prompt)
    input_end = min(gen_s, len(vec))
    input_vec = vec[:input_end]
    n_input = len(input_vec)
    if n_input == 0:
        return {}
    global_mean = float(input_vec.mean())
    if global_mean == 0:
        return {}

    result = {}

    # Keyword prompt RAPT
    n_kw = len(kw_prompt_indices)
    if n_kw > 0:
        kw_attn = sum(float(vec[ki]) for ki in kw_prompt_indices if ki < len(vec))
        result["keyword_prompt_rapt"] = (kw_attn / n_kw) / global_mean
    else:
        result["keyword_prompt_rapt"] = 0.0

    # Full image RAPT
    img_vec = vec[img_s:img_e]
    n_img = len(img_vec)
    if n_img > 0:
        result["full_image_rapt"] = float(img_vec.mean()) / global_mean

    # Object bbox RAPT
    if bbox_mask is not None:
        n_bbox = int(bbox_mask.sum())
        if n_bbox > 0:
            result["object_bbox_rapt"] = float((img_vec * bbox_mask).sum() / n_bbox) / global_mean
        else:
            result["object_bbox_rapt"] = float("nan")
    else:
        result["object_bbox_rapt"] = float("nan")

    # Prior generated RAPT (use total_mean including generated for fairness)
    gen_end = min(abs_idx, len(vec))
    n_gen = gen_end - gen_s
    if n_gen > 0:
        gen_vec = vec[gen_s:gen_end]
        # For prior_gen, normalize against the global mean of all tokens up to abs_idx
        all_mean = float(vec[:gen_end].mean()) if gen_end > 0 else global_mean
        result["prior_gen_rapt"] = (float(gen_vec.mean()) / all_mean) if all_mean > 0 else 0.0
    else:
        result["prior_gen_rapt"] = 0.0

    return result


def compute_soft_iou(vec, tr, bbox_mask):
    """Compute Soft IoU between image attention and object bbox mask.

    Soft IoU = Σ(A · M) / [Σ A + Σ M − Σ(A · M)]
    Both A and M are normalized to sum to 1 (probability distributions over
    image patches), so this measures distributional overlap.
    Returns NaN if no bbox or all-zero attention.
    """
    if vec is None or bbox_mask is None:
        return float("nan")

    img_s, img_e = tr["image_start"], tr["image_end"]
    img_vec = vec[img_s:img_e].astype(np.float64)

    if len(img_vec) != len(bbox_mask) or len(img_vec) == 0:
        return float("nan")

    # Normalize attention to probability distribution (sum to 1)
    a_sum = float(img_vec.sum())
    if a_sum < 1e-12:
        return float("nan")
    A = img_vec / a_sum

    # Normalize mask to probability distribution (uniform over bbox patches)
    M = bbox_mask.astype(np.float64)
    m_sum = float(M.sum())
    if m_sum < 1e-12:
        return float("nan")
    M = M / m_sum

    intersection = float(np.sum(A * M))
    union = float(np.sum(A) + np.sum(M) - intersection)
    if union < 1e-12:
        return float("nan")
    return intersection / union


def find_keyword_in_prompt(token_strings, tr, keyword):
    """Find indices of prompt tokens matching the keyword."""
    prompt_s, prompt_e = tr["prompt_start"], tr["prompt_end"]
    indices = []
    for i in range(prompt_s, prompt_e):
        tok = token_strings[i].strip().lower()
        # Match if keyword is substring of token or token is substring of keyword
        # Handle tokenizer artifacts like " vehicles" -> "vehicles"
        if keyword.lower() in tok or tok in keyword.lower():
            # Avoid matching empty or single-char special tokens
            if len(tok) >= 2 or tok == keyword.lower():
                indices.append(i)
    return indices


def find_keyword_in_generated(token_strings, tr, keyword):
    """Find generated token indices matching the keyword. Returns list of (gen_idx, text)."""
    gen_s = tr["generation_start"]
    matches = []
    kw = keyword.lower()
    for i in range(gen_s, len(token_strings)):
        tok = token_strings[i].strip().lower()
        if kw in tok and len(tok) >= 2:
            matches.append((i - gen_s, token_strings[i]))
    return matches


# ── Main analysis ─────────────────────────────────────────────────────

def analyze_all(manifest_path, output_dir):
    with open(manifest_path) as f:
        manifest = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots", "per_keyword_breakdown"), exist_ok=True)

    all_rows = []
    baseline_rows = []  # non-keyword generated tokens for comparison
    position_rows = []  # ALL generated tokens with position info for sliding window
    skipped = []

    for qi, question in enumerate(manifest["questions"]):
        qid = question["id"]
        image_id = question["image_id"]
        detections = question.get("detections", [])
        grid_h, grid_w = question["patch_grid"]
        keywords = QUESTION_KEYWORDS.get(qid, [])

        if not keywords:
            continue

        for model_key in MODEL_KEYS:
            # Load contributions.pt
            pt_path = os.path.join(PT_BASE, model_key, image_id, "contributions.pt")
            if not os.path.exists(pt_path):
                skipped.append((qid, model_key, "no contributions.pt"))
                continue

            data = load_contributions(pt_path)
            tr = data["token_regions"]
            token_strings = data["token_strings"]

            # Collect all keyword gen indices for baseline exclusion
            all_kw_gen_indices = set()

            for keyword in keywords:
                # Find keyword in prompt
                kw_prompt_indices = find_keyword_in_prompt(token_strings, tr, keyword)
                if not kw_prompt_indices:
                    skipped.append((qid, model_key, f"keyword '{keyword}' not in prompt"))
                    continue

                # Try to build bbox mask
                matched_dets = match_token_to_detections(keyword, detections)
                bbox_mask = build_box_mask(matched_dets, grid_h, grid_w) if matched_dets else None
                has_bbox = bbox_mask is not None and bbox_mask.sum() > 0
                if bbox_mask is not None and bbox_mask.sum() == 0:
                    bbox_mask = None
                    has_bbox = False

                # Find keyword in generated tokens
                gen_matches = find_keyword_in_generated(token_strings, tr, keyword)
                if not gen_matches:
                    skipped.append((qid, model_key, f"keyword '{keyword}' not in generated text"))
                    continue

                all_kw_gen_indices.update(gi for gi, _ in gen_matches)

                print(f"[{qi+1}/{len(manifest['questions'])}] {qid}/{model_key}: "
                      f"keyword='{keyword}', prompt_indices={kw_prompt_indices}, "
                      f"has_bbox={has_bbox}, gen_matches={len(gen_matches)}")

                for gen_idx, gen_text in gen_matches:
                    abs_idx = tr["generation_start"] + gen_idx
                    phase = get_phase(abs_idx, tr)

                    # Process layer='all' (averaged) + each individual layer
                    layers_to_process = [None] + list(range(36))
                    for layer_idx in layers_to_process:
                        layer_label = "all" if layer_idx is None else str(layer_idx)

                        attn_vec = get_attn_vector(data, gen_idx, layer_idx)
                        tam_vec = get_tam_vector(data, gen_idx, layer_idx)

                        if attn_vec is None:
                            continue

                        # Decompose attention
                        attn_decomp = decompose_vector(
                            attn_vec, tr, kw_prompt_indices, bbox_mask, abs_idx
                        )
                        attn_total = attn_decomp["total"]

                        # Decompose TAM
                        tam_decomp = None
                        if tam_vec is not None:
                            # TAM vector covers full sequence, trim to abs_idx+1
                            tam_trimmed = tam_vec[:abs_idx + 1] if len(tam_vec) > abs_idx + 1 else tam_vec
                            tam_decomp = decompose_vector(
                                tam_trimmed, tr, kw_prompt_indices, bbox_mask, abs_idx
                            )

                        # RAPT
                        rapt = compute_rapt(attn_vec, tr, kw_prompt_indices, bbox_mask, abs_idx)

                        # Soft IoU (attention vs bbox)
                        soft_iou = compute_soft_iou(attn_vec, tr, bbox_mask) if has_bbox else float("nan")

                        # Build row
                        row = {
                            "qid": qid,
                            "image_id": image_id,
                            "model": model_key,
                            "keyword": keyword,
                            "gen_token_idx": gen_idx,
                            "gen_token_text": gen_text.strip(),
                            "phase": phase,
                            "layer": layer_label,
                            "has_bbox": has_bbox,
                            # Raw attention %
                            "keyword_prompt_attn": round(attn_decomp["keyword_prompt"] / attn_total * 100, FLOAT_PREC) if attn_total > 0 else 0,
                            "other_prompt_attn": round(attn_decomp["other_prompt"] / attn_total * 100, FLOAT_PREC) if attn_total > 0 else 0,
                            "object_bbox_attn": round(attn_decomp["object_bbox"] / attn_total * 100, FLOAT_PREC) if attn_total > 0 and has_bbox else float("nan"),
                            "other_image_attn": round(attn_decomp["other_image"] / attn_total * 100, FLOAT_PREC) if attn_total > 0 else 0,
                            "full_image_attn": round(attn_decomp["full_image"] / attn_total * 100, FLOAT_PREC) if attn_total > 0 else 0,
                            "prior_gen_attn": round(attn_decomp["prior_gen"] / attn_total * 100, FLOAT_PREC) if attn_total > 0 else 0,
                            "system_attn": round(attn_decomp["system"] / attn_total * 100, FLOAT_PREC) if attn_total > 0 else 0,
                            # RAPT
                            "keyword_prompt_rapt": round(rapt.get("keyword_prompt_rapt", 0), FLOAT_PREC),
                            "object_bbox_rapt": round(rapt.get("object_bbox_rapt", float("nan")), FLOAT_PREC),
                            "full_image_rapt": round(rapt.get("full_image_rapt", 0), FLOAT_PREC),
                            "prior_gen_rapt": round(rapt.get("prior_gen_rapt", 0), FLOAT_PREC),
                            # Soft IoU
                            "soft_iou": round(soft_iou, FLOAT_PREC),
                            # TAM
                            "keyword_prompt_tam": round(tam_decomp["keyword_prompt"] / tam_decomp["total"] * 100, FLOAT_PREC) if tam_decomp and tam_decomp["total"] > 0 else float("nan"),
                            "object_bbox_tam": round(tam_decomp["object_bbox"] / tam_decomp["total"] * 100, FLOAT_PREC) if tam_decomp and tam_decomp["total"] > 0 and has_bbox else float("nan"),
                            "full_image_tam": round(tam_decomp["full_image"] / tam_decomp["total"] * 100, FLOAT_PREC) if tam_decomp and tam_decomp["total"] > 0 else float("nan"),
                            "prior_gen_tam": round(tam_decomp["prior_gen"] / tam_decomp["total"] * 100, FLOAT_PREC) if tam_decomp and tam_decomp["total"] > 0 else float("nan"),
                        }
                        all_rows.append(row)

            # ── Baseline: non-keyword generated tokens (layer-averaged only) ──
            # For each generated token, check if it matches any prompt token.
            # This lets us compare:
            #   - keyword tokens → attention to keyword in prompt
            #   - non-keyword tokens that match a prompt token → attention to that match
            #   - non-keyword tokens with no prompt match → no self-referencing
            first_kw = keywords[0]
            kw_prompt_indices = find_keyword_in_prompt(token_strings, tr, first_kw)
            if not kw_prompt_indices:
                kw_prompt_indices = []

            # Build prompt token lookup: lowercase stripped text → list of absolute indices
            prompt_s, prompt_e = tr["prompt_start"], tr["prompt_end"]
            prompt_token_map = {}
            for pi in range(prompt_s, prompt_e):
                ptok = token_strings[pi].strip().lower()
                if len(ptok) >= 2:  # skip single-char tokens
                    prompt_token_map.setdefault(ptok, []).append(pi)

            n_gen = data["attn_gen_rows"].shape[1]
            for gen_idx in range(n_gen):
                if gen_idx in all_kw_gen_indices:
                    continue
                abs_idx = tr["generation_start"] + gen_idx
                phase = get_phase(abs_idx, tr)
                attn_vec = get_attn_vector(data, gen_idx, None)  # layer-averaged
                if attn_vec is None:
                    continue
                attn_decomp = decompose_vector(
                    attn_vec, tr, kw_prompt_indices, None, abs_idx
                )
                attn_total = attn_decomp["total"]
                if attn_total <= 0:
                    continue

                # Check if this generated token matches any prompt token
                gen_tok_text = token_strings[tr["generation_start"] + gen_idx].strip().lower()
                matched_prompt_indices = prompt_token_map.get(gen_tok_text, [])
                # Also try substring matching (same logic as find_keyword_in_prompt)
                if not matched_prompt_indices:
                    for ptok_text, ptok_indices in prompt_token_map.items():
                        if gen_tok_text in ptok_text or ptok_text in gen_tok_text:
                            matched_prompt_indices = ptok_indices
                            break
                has_prompt_match = len(matched_prompt_indices) > 0

                # Compute attention to the matching prompt token(s)
                matched_prompt_attn = 0.0
                if has_prompt_match:
                    for mpi in matched_prompt_indices:
                        if mpi < len(attn_vec):
                            matched_prompt_attn += float(attn_vec[mpi])

                baseline_rows.append({
                    "qid": qid,
                    "model": model_key,
                    "phase": phase,
                    "gen_token_text": gen_tok_text,
                    "has_prompt_match": has_prompt_match,
                    "matched_prompt_attn": round(matched_prompt_attn / attn_total * 100, FLOAT_PREC) if attn_total > 0 else 0,
                    "keyword_prompt_attn": round(attn_decomp["keyword_prompt"] / attn_total * 100, FLOAT_PREC),
                    "other_prompt_attn": round(attn_decomp["other_prompt"] / attn_total * 100, FLOAT_PREC),
                    "full_image_attn": round(attn_decomp["full_image"] / attn_total * 100, FLOAT_PREC),
                    "prior_gen_attn": round(attn_decomp["prior_gen"] / attn_total * 100, FLOAT_PREC),
                    "system_attn": round(attn_decomp["system"] / attn_total * 100, FLOAT_PREC),
                })

            # ── Position tracking: all generated tokens for sliding window ──
            n_gen_pos = data["attn_gen_rows"].shape[1]
            gen_s_pos = tr["generation_start"]
            img_s_pos, img_e_pos = tr["image_start"], tr["image_end"]
            ps_pos, pe_pos = tr["prompt_start"], tr["prompt_end"]

            for gen_idx in range(n_gen_pos):
                abs_idx = gen_s_pos + gen_idx
                phase = get_phase(abs_idx, tr)
                attn_vec = get_attn_vector(data, gen_idx, None)
                if attn_vec is None:
                    continue
                attn_total = float(attn_vec.sum())
                if attn_total <= 0:
                    continue

                pe_c = min(pe_pos, len(attn_vec))
                gen_end_c = min(abs_idx, len(attn_vec))

                position_rows.append({
                    "qid": qid,
                    "model": model_key,
                    "gen_idx": gen_idx,
                    "n_gen": n_gen_pos,
                    "phase": phase,
                    "image_attn": float(attn_vec[img_s_pos:img_e_pos].sum()) / attn_total * 100,
                    "prompt_attn": float(attn_vec[ps_pos:pe_c].sum()) / attn_total * 100,
                    "prior_gen_attn": float(attn_vec[gen_s_pos:gen_end_c].sum()) / attn_total * 100 if gen_s_pos < gen_end_c else 0.0,
                    "system_attn": float(attn_vec[:img_s_pos].sum()) / attn_total * 100 if img_s_pos > 0 else 0.0,
                })

            # Free memory
            del data
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Report skipped
    if skipped:
        print(f"\nSkipped {len(skipped)} cases:")
        for qid, model, reason in skipped:
            print(f"  {qid}/{model}: {reason}")

    print(f"Baseline (non-keyword) tokens collected: {len(baseline_rows)}")
    print(f"Position (all gen tokens) collected: {len(position_rows)}")
    return all_rows, baseline_rows, manifest, position_rows


def write_csv(rows, output_dir):
    """Write per-token CSV results."""
    csv_path = os.path.join(output_dir, "prompt_keyword_attention.csv")
    if not rows:
        print("No rows to write.")
        return csv_path

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written: {csv_path} ({len(rows)} rows)")
    return csv_path


def write_summary(rows, output_dir, manifest, baseline_rows=None, position_rows=None):
    """Write summary markdown and JSON with aggregate stats and statistical tests."""
    if not rows:
        return

    # Filter to layer='all' for aggregate stats
    all_layer = [r for r in rows if r["layer"] == "all"]

    md_path = os.path.join(output_dir, "prompt_keyword_summary.md")
    json_path = os.path.join(output_dir, "prompt_keyword_summary.json")
    summary = {}

    with open(md_path, "w") as md:
        md.write("# Prompt Keyword Attention Analysis\n\n")
        md.write(f"Analyzed {len(all_layer)} keyword-matching generated tokens "
                 f"across {len(QUESTION_KEYWORDS)} questions and {len(MODEL_KEYS)} models.\n\n")
        md.write("**Research question**: When a keyword from the question appears in "
                 "generated text, does the model attend more to the textual mention "
                 "(keyword in prompt) or to the visual object in the image?\n\n")

        # ── 1. Overall attention decomposition per model ──
        md.write("## 1. Attention Decomposition by Model (layer-averaged)\n\n")
        md.write("| Model | Keyword Prompt % | Other Prompt % | Full Image % | Prior Gen % | System % | N |\n")
        md.write("|-------|-----------------|----------------|-------------|------------|---------|---|\n")

        summary["per_model"] = {}
        for mk in MODEL_KEYS:
            model_rows = [r for r in all_layer if r["model"] == mk]
            if not model_rows:
                continue
            kw_p = np.array([r["keyword_prompt_attn"] for r in model_rows])
            op = np.array([r["other_prompt_attn"] for r in model_rows])
            fi = np.array([r["full_image_attn"] for r in model_rows])
            pg = np.array([r["prior_gen_attn"] for r in model_rows])
            sy = np.array([r["system_attn"] for r in model_rows])

            md.write(f"| {MODEL_LABELS[mk]} | {kw_p.mean():.2f} +/- {kw_p.std():.2f} | "
                     f"{op.mean():.2f} +/- {op.std():.2f} | "
                     f"{fi.mean():.2f} +/- {fi.std():.2f} | "
                     f"{pg.mean():.2f} +/- {pg.std():.2f} | "
                     f"{sy.mean():.2f} +/- {sy.std():.2f} | {len(model_rows)} |\n")

            summary["per_model"][mk] = {
                "keyword_prompt_attn": {"mean": round(kw_p.mean(), 4), "std": round(kw_p.std(), 4)},
                "full_image_attn": {"mean": round(fi.mean(), 4), "std": round(fi.std(), 4)},
                "prior_gen_attn": {"mean": round(pg.mean(), 4), "std": round(pg.std(), 4)},
                "n": len(model_rows),
            }
        md.write("\n")

        # ── 2. RAPT comparison ──
        md.write("## 2. RAPT-Normalized Attention (per-token-count fairness)\n\n")
        md.write("RAPT > 1 = disproportionately high attention; < 1 = below average.\n\n")
        md.write("| Model | Keyword Prompt RAPT | Full Image RAPT | Object Bbox RAPT | Prior Gen RAPT | Soft IoU | N |\n")
        md.write("|-------|--------------------|-----------------|--------------------|---------------|----------|---|\n")

        summary["per_model_rapt"] = {}
        for mk in MODEL_KEYS:
            model_rows = [r for r in all_layer if r["model"] == mk]
            if not model_rows:
                continue
            kpr = np.array([r["keyword_prompt_rapt"] for r in model_rows])
            fir = np.array([r["full_image_rapt"] for r in model_rows])
            obr = np.array([r["object_bbox_rapt"] for r in model_rows if not np.isnan(r["object_bbox_rapt"])])
            pgr = np.array([r["prior_gen_rapt"] for r in model_rows])
            siou = np.array([r["soft_iou"] for r in model_rows if not np.isnan(r["soft_iou"])])

            obr_str = f"{obr.mean():.2f} +/- {obr.std():.2f} (n={len(obr)})" if len(obr) > 0 else "N/A"
            siou_str = f"{siou.mean():.4f} +/- {siou.std():.4f} (n={len(siou)})" if len(siou) > 0 else "N/A"
            md.write(f"| {MODEL_LABELS[mk]} | {kpr.mean():.2f} +/- {kpr.std():.2f} | "
                     f"{fir.mean():.2f} +/- {fir.std():.2f} | "
                     f"{obr_str} | "
                     f"{pgr.mean():.2f} +/- {pgr.std():.2f} | "
                     f"{siou_str} | {len(model_rows)} |\n")

            summary["per_model_rapt"][mk] = {
                "keyword_prompt_rapt": {"mean": round(kpr.mean(), 4), "std": round(kpr.std(), 4)},
                "full_image_rapt": {"mean": round(fir.mean(), 4), "std": round(fir.std(), 4)},
                "prior_gen_rapt": {"mean": round(pgr.mean(), 4), "std": round(pgr.std(), 4)},
                "soft_iou": {"mean": round(float(siou.mean()), 4), "std": round(float(siou.std()), 4), "n": len(siou)} if len(siou) > 0 else None,
            }
        md.write("\n")

        # ── 3. Bbox vs no-bbox keywords (split by model) ──
        md.write("## 3. Keywords With vs Without Bounding Box (by Model)\n\n")
        md.write("| Model | Has BBox | Keyword Prompt % | Full Image % | Object BBox % | Prior Gen % | Soft IoU | N |\n")
        md.write("|-------|----------|-----------------|-------------|--------------|------------|----------|---|\n")

        for mk in MODEL_KEYS:
            for has_bb in [True, False]:
                bb_rows = [r for r in all_layer if r["has_bbox"] == has_bb and r["model"] == mk]
                if not bb_rows:
                    continue
                kw_p = np.array([r["keyword_prompt_attn"] for r in bb_rows])
                fi = np.array([r["full_image_attn"] for r in bb_rows])
                pg = np.array([r["prior_gen_attn"] for r in bb_rows])
                if has_bb:
                    ob = np.array([r["object_bbox_attn"] for r in bb_rows if not np.isnan(r["object_bbox_attn"])])
                    ob_str = f"{ob.mean():.2f} +/- {ob.std():.2f}"
                    siou = np.array([r["soft_iou"] for r in bb_rows if not np.isnan(r["soft_iou"])])
                    siou_str = f"{siou.mean():.4f} +/- {siou.std():.4f}" if len(siou) > 0 else "N/A"
                else:
                    ob_str = "N/A"
                    siou_str = "N/A"
                label = "Yes" if has_bb else "No"
                md.write(f"| {MODEL_LABELS[mk]} | {label} | {kw_p.mean():.2f} +/- {kw_p.std():.2f} | "
                         f"{fi.mean():.2f} +/- {fi.std():.2f} | {ob_str} | "
                         f"{pg.mean():.2f} +/- {pg.std():.2f} | {siou_str} | {len(bb_rows)} |\n")
        md.write("\n")

        # ── 4. Phase comparison (Thinking models) ──
        md.write("## 4. Thinking vs Answer Phase (Thinking models)\n\n")
        summary["phase_comparison"] = {}
        for think_mk in [mk for mk in MODEL_KEYS if "thinking" in mk]:
            think_rows = [r for r in all_layer if r["model"] == think_mk]
            if not think_rows:
                continue
            md.write(f"### {MODEL_LABELS[think_mk]}\n\n")
            md.write("| Phase | Keyword Prompt % | Full Image % | Prior Gen % | KW Prompt RAPT | Image RAPT | Soft IoU | N |\n")
            md.write("|-------|-----------------|-------------|------------|---------------|-----------|----------|---|\n")

            for phase in ["thinking", "answer"]:
                phase_r = [r for r in think_rows if r["phase"] == phase]
                if not phase_r:
                    continue
                kw_p = np.array([r["keyword_prompt_attn"] for r in phase_r])
                fi = np.array([r["full_image_attn"] for r in phase_r])
                pg = np.array([r["prior_gen_attn"] for r in phase_r])
                kpr = np.array([r["keyword_prompt_rapt"] for r in phase_r])
                fir = np.array([r["full_image_rapt"] for r in phase_r])
                siou = np.array([r["soft_iou"] for r in phase_r if not np.isnan(r["soft_iou"])])

                siou_str = f"{siou.mean():.4f} +/- {siou.std():.4f}" if len(siou) > 0 else "N/A"
                md.write(f"| {phase.title()} | {kw_p.mean():.2f} +/- {kw_p.std():.2f} | "
                         f"{fi.mean():.2f} +/- {fi.std():.2f} | "
                         f"{pg.mean():.2f} +/- {pg.std():.2f} | "
                         f"{kpr.mean():.2f} +/- {kpr.std():.2f} | "
                         f"{fir.mean():.2f} +/- {fir.std():.2f} | "
                         f"{siou_str} | {len(phase_r)} |\n")

                summary["phase_comparison"][f"{think_mk}_{phase}"] = {
                    "keyword_prompt_attn": round(kw_p.mean(), 4),
                    "full_image_attn": round(fi.mean(), 4),
                    "soft_iou": round(float(siou.mean()), 4) if len(siou) > 0 else None,
                    "n": len(phase_r),
                }
            md.write("\n")

        # ── 5. Per-question breakdown ──
        md.write("## 5. Per-Question Keyword Attention (layer-averaged)\n\n")
        q_text_map = {q["id"]: q["question_text"] for q in manifest["questions"]}

        for qid in sorted(QUESTION_KEYWORDS.keys()):
            q_rows = [r for r in all_layer if r["qid"] == qid]
            if not q_rows:
                continue
            md.write(f"### {qid}: {q_text_map.get(qid, '')}\n")
            md.write(f"Keywords: {', '.join(QUESTION_KEYWORDS[qid])}\n\n")
            md.write("| Model | Keyword | Phase | KW Prompt % | Image % | Obj BBox % | Prior Gen % | RAPT(KW) | RAPT(Img) | Soft IoU | N |\n")
            md.write("|-------|---------|-------|------------|---------|-----------|------------|----------|----------|----------|---|\n")

            for mk in MODEL_KEYS:
                for kw in QUESTION_KEYWORDS[qid]:
                    for phase in ["thinking", "answer"]:
                        pr = [r for r in q_rows if r["model"] == mk and r["keyword"] == kw and r["phase"] == phase]
                        if not pr:
                            continue
                        kw_p = np.mean([r["keyword_prompt_attn"] for r in pr])
                        fi = np.mean([r["full_image_attn"] for r in pr])
                        ob_vals = [r["object_bbox_attn"] for r in pr if not np.isnan(r["object_bbox_attn"])]
                        ob = np.mean(ob_vals) if ob_vals else float("nan")
                        pg = np.mean([r["prior_gen_attn"] for r in pr])
                        kpr = np.mean([r["keyword_prompt_rapt"] for r in pr])
                        fir = np.mean([r["full_image_rapt"] for r in pr])
                        siou_vals = [r["soft_iou"] for r in pr if not np.isnan(r["soft_iou"])]
                        siou = np.mean(siou_vals) if siou_vals else float("nan")
                        ob_str = f"{ob:.2f}" if not np.isnan(ob) else "N/A"
                        siou_str = f"{siou:.4f}" if not np.isnan(siou) else "N/A"

                        md.write(f"| {MODEL_LABELS[mk]} | {kw} | {phase} | {kw_p:.2f} | "
                                 f"{fi:.2f} | {ob_str} | {pg:.2f} | "
                                 f"{kpr:.2f} | {fir:.2f} | {siou_str} | {len(pr)} |\n")
            md.write("\n")

        # ── 6. Statistical tests ──
        md.write("## 6. Statistical Tests\n\n")

        # 6a. Instruct vs Thinking (Mann-Whitney U) — per size
        summary["tests"] = {}
        MODEL_PAIRS = [("instruct", "thinking", "8B"), ("instruct_30b", "thinking_30b", "30B")]
        for inst_key, think_key, size_label in MODEL_PAIRS:
            md.write(f"### 6a. Instruct vs Thinking — {size_label} (Mann-Whitney U)\n\n")
            md.write(f"| Metric | U-stat | p-value | {MODEL_LABELS[inst_key]} Mean | {MODEL_LABELS[think_key]} Mean | Direction |\n")
            md.write("|--------|--------|---------|--------------|--------------|----------|\n")

            for metric in ["keyword_prompt_attn", "full_image_attn", "prior_gen_attn",
                            "keyword_prompt_rapt", "full_image_rapt", "soft_iou"]:
                vi = np.array([r[metric] for r in all_layer if r["model"] == inst_key and not np.isnan(r[metric])])
                vt = np.array([r[metric] for r in all_layer if r["model"] == think_key and not np.isnan(r[metric])])
                if len(vi) < 2 or len(vt) < 2:
                    continue
                u_stat, p_val = stats.mannwhitneyu(vi, vt, alternative="two-sided")
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                direction = "Think > Inst" if vt.mean() > vi.mean() else "Inst > Think"
                md.write(f"| {metric} | {u_stat:.0f} | {p_val:.4e} | {vi.mean():.4f} | {vt.mean():.4f} | {direction} {sig} |\n")
                summary["tests"][f"{inst_key}_vs_{think_key}_{metric}"] = {
                    "u_stat": round(float(u_stat), 2), "p_val": round(float(p_val), 6),
                    "instruct_mean": round(vi.mean(), 4), "thinking_mean": round(vt.mean(), 4),
                }
            md.write("\n")

        # 6a-cross. 8B vs 30B within same model type
        SIZE_PAIRS = [("instruct", "instruct_30b"), ("thinking", "thinking_30b")]
        md.write("### 6a-cross. 8B vs 30B (Mann-Whitney U)\n\n")
        md.write("| Comparison | Metric | U-stat | p-value | 8B Mean | 30B Mean | Direction |\n")
        md.write("|------------|--------|--------|---------|---------|---------|----------|\n")

        for key_8b, key_30b in SIZE_PAIRS:
            for metric in ["keyword_prompt_attn", "full_image_attn", "prior_gen_attn",
                            "keyword_prompt_rapt", "full_image_rapt", "soft_iou"]:
                v8 = np.array([r[metric] for r in all_layer if r["model"] == key_8b and not np.isnan(r[metric])])
                v30 = np.array([r[metric] for r in all_layer if r["model"] == key_30b and not np.isnan(r[metric])])
                if len(v8) < 2 or len(v30) < 2:
                    continue
                u_stat, p_val = stats.mannwhitneyu(v8, v30, alternative="two-sided")
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                direction = "30B > 8B" if v30.mean() > v8.mean() else "8B > 30B"
                label = f"{MODEL_LABELS[key_8b]} vs {MODEL_LABELS[key_30b]}"
                md.write(f"| {label} | {metric} | {u_stat:.0f} | {p_val:.4e} | {v8.mean():.4f} | {v30.mean():.4f} | {direction} {sig} |\n")
                summary["tests"][f"{key_8b}_vs_{key_30b}_{metric}"] = {
                    "u_stat": round(float(u_stat), 2), "p_val": round(float(p_val), 6),
                    "8b_mean": round(v8.mean(), 4), "30b_mean": round(v30.mean(), 4),
                }
        md.write("\n")

        # 6b. Thinking phase vs Answer phase (within each Thinking model)
        for think_mk in [mk for mk in MODEL_KEYS if "thinking" in mk]:
            md.write(f"### 6b. Thinking Phase vs Answer Phase — {MODEL_LABELS[think_mk]} (Mann-Whitney U)\n\n")
            md.write("| Metric | U-stat | p-value | Thinking Mean | Answer Mean | Direction |\n")
            md.write("|--------|--------|---------|--------------|------------|----------|\n")

            for metric in ["keyword_prompt_attn", "full_image_attn", "prior_gen_attn",
                            "keyword_prompt_rapt", "full_image_rapt", "soft_iou"]:
                vt = np.array([r[metric] for r in all_layer if r["model"] == think_mk and r["phase"] == "thinking" and not np.isnan(r[metric])])
                va = np.array([r[metric] for r in all_layer if r["model"] == think_mk and r["phase"] == "answer" and not np.isnan(r[metric])])
                if len(vt) < 2 or len(va) < 2:
                    continue
                u_stat, p_val = stats.mannwhitneyu(vt, va, alternative="two-sided")
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                direction = "Answer > Think" if va.mean() > vt.mean() else "Think > Answer"
                md.write(f"| {metric} | {u_stat:.0f} | {p_val:.4e} | {vt.mean():.4f} | {va.mean():.4f} | {direction} {sig} |\n")
            md.write("\n")

        # 6c. Layer-wise trends (Spearman)
        md.write("### 6c. Layer-wise Trends (Spearman correlation)\n\n")
        md.write("Positive rho = metric increases with depth.\n\n")
        md.write("| Model | Metric | Spearman rho | p-value | Trend |\n")
        md.write("|-------|--------|-------------|---------|-------|\n")

        layer_rows = [r for r in rows if r["layer"] != "all"]
        for mk in MODEL_KEYS:
            for metric in ["keyword_prompt_attn", "full_image_attn", "keyword_prompt_rapt", "full_image_rapt", "soft_iou"]:
                layers = []
                vals = []
                for r in layer_rows:
                    if r["model"] == mk and not np.isnan(r[metric]):
                        layers.append(int(r["layer"]))
                        vals.append(r[metric])
                if len(layers) < 10:
                    continue
                rho, p_val = stats.spearmanr(layers, vals)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                trend = "deeper layers higher" if rho > 0 else "deeper layers lower"
                md.write(f"| {MODEL_LABELS[mk]} | {metric} | {rho:.4f} | {p_val:.4e} | {trend} {sig} |\n")
        md.write("\n")

        # ── 7. TAM comparison ──
        md.write("## 7. TAM Decomposition (layer-averaged)\n\n")
        md.write("| Model | Keyword Prompt TAM % | Full Image TAM % | Prior Gen TAM % | N |\n")
        md.write("|-------|---------------------|-----------------|----------------|---|\n")

        for mk in MODEL_KEYS:
            model_rows = [r for r in all_layer if r["model"] == mk]
            if not model_rows:
                continue
            kw_t = np.array([r["keyword_prompt_tam"] for r in model_rows if not np.isnan(r["keyword_prompt_tam"])])
            fi_t = np.array([r["full_image_tam"] for r in model_rows if not np.isnan(r["full_image_tam"])])
            pg_t = np.array([r["prior_gen_tam"] for r in model_rows if not np.isnan(r["prior_gen_tam"])])
            n = len(kw_t)
            if n > 0:
                md.write(f"| {MODEL_LABELS[mk]} | {kw_t.mean():.2f} +/- {kw_t.std():.2f} | "
                         f"{fi_t.mean():.2f} +/- {fi_t.std():.2f} | "
                         f"{pg_t.mean():.2f} +/- {pg_t.std():.2f} | {n} |\n")
        md.write("\n")

        # ── 8. Three-Way Token Comparison ──
        # Keyword tokens, prompt-matching non-keyword tokens, non-matching tokens
        if baseline_rows:
            # Split baseline into prompt-matching vs non-matching
            prompt_match_rows = [r for r in baseline_rows if r.get("has_prompt_match", False)]
            no_match_rows = [r for r in baseline_rows if not r.get("has_prompt_match", False)]

            md.write("## 8. Textual Echo Analysis: Three-Way Token Comparison\n\n")
            md.write("Does the 'textual echo' (attending back to matching prompt token) happen for ALL "
                     "tokens that appear in both the prompt and generated text, or only for semantically "
                     "important keywords?\n\n")
            md.write("Three token categories:\n")
            md.write("- **Keyword**: generated tokens matching a manually-identified question keyword "
                     "(e.g., 'vehicles', 'pedestrians')\n")
            md.write("- **Prompt-match**: non-keyword generated tokens that also appear in the prompt "
                     "(e.g., 'the', 'how', 'many', 'are'). 'Matched prompt attn' = attention to the "
                     "matching token position in the prompt.\n")
            md.write("- **No-match**: generated tokens with no corresponding prompt token\n\n")

            # 8.0 — Summary table with matched-prompt attention
            md.write("### 8.0 Self-Referencing Attention (matched prompt token)\n\n")
            md.write("For keyword tokens, 'self-ref attn' = attention to the keyword in the prompt. "
                     "For prompt-match tokens, 'self-ref attn' = attention to the matching prompt "
                     "token (e.g., generated 'the' → prompt 'the'). For no-match tokens, there is "
                     "no matching prompt token so self-ref = 0.\n\n")
            md.write("| Model | Token Type | Self-Ref Prompt Attn % | Full Image % | Prior Gen % | System % | N |\n")
            md.write("|-------|-----------|----------------------|-------------|------------|---------|---|\n")

            summary["three_way"] = {}
            for mk in MODEL_KEYS:
                kw_rows_mk = [r for r in all_layer if r["model"] == mk]
                pm_rows_mk = [r for r in prompt_match_rows if r["model"] == mk]
                nm_rows_mk = [r for r in no_match_rows if r["model"] == mk]

                if not kw_rows_mk:
                    continue

                # Keyword tokens: self-ref = keyword_prompt_attn
                kw_self = np.array([r["keyword_prompt_attn"] for r in kw_rows_mk])
                kw_img = np.array([r["full_image_attn"] for r in kw_rows_mk])
                kw_pg = np.array([r["prior_gen_attn"] for r in kw_rows_mk])
                kw_sys = np.array([r["system_attn"] for r in kw_rows_mk])
                md.write(f"| {MODEL_LABELS[mk]} | Keyword | "
                         f"{kw_self.mean():.3f} +/- {kw_self.std():.3f} | "
                         f"{kw_img.mean():.2f} +/- {kw_img.std():.2f} | "
                         f"{kw_pg.mean():.2f} +/- {kw_pg.std():.2f} | "
                         f"{kw_sys.mean():.2f} +/- {kw_sys.std():.2f} | {len(kw_rows_mk)} |\n")

                summary["three_way"][f"{mk}_keyword"] = {
                    "self_ref_attn": round(kw_self.mean(), 4),
                    "full_image_attn": round(kw_img.mean(), 4),
                    "n": len(kw_rows_mk),
                }

                # Prompt-match tokens: self-ref = matched_prompt_attn
                if pm_rows_mk:
                    pm_self = np.array([r["matched_prompt_attn"] for r in pm_rows_mk])
                    pm_img = np.array([r["full_image_attn"] for r in pm_rows_mk])
                    pm_pg = np.array([r["prior_gen_attn"] for r in pm_rows_mk])
                    pm_sys = np.array([r["system_attn"] for r in pm_rows_mk])
                    md.write(f"| {MODEL_LABELS[mk]} | Prompt-match | "
                             f"{pm_self.mean():.3f} +/- {pm_self.std():.3f} | "
                             f"{pm_img.mean():.2f} +/- {pm_img.std():.2f} | "
                             f"{pm_pg.mean():.2f} +/- {pm_pg.std():.2f} | "
                             f"{pm_sys.mean():.2f} +/- {pm_sys.std():.2f} | {len(pm_rows_mk)} |\n")

                    summary["three_way"][f"{mk}_prompt_match"] = {
                        "self_ref_attn": round(pm_self.mean(), 4),
                        "full_image_attn": round(pm_img.mean(), 4),
                        "n": len(pm_rows_mk),
                    }

                # No-match tokens: self-ref = 0 by definition
                if nm_rows_mk:
                    nm_img = np.array([r["full_image_attn"] for r in nm_rows_mk])
                    nm_pg = np.array([r["prior_gen_attn"] for r in nm_rows_mk])
                    nm_sys = np.array([r["system_attn"] for r in nm_rows_mk])
                    md.write(f"| {MODEL_LABELS[mk]} | No-match | "
                             f"0.000 +/- 0.000 | "
                             f"{nm_img.mean():.2f} +/- {nm_img.std():.2f} | "
                             f"{nm_pg.mean():.2f} +/- {nm_pg.std():.2f} | "
                             f"{nm_sys.mean():.2f} +/- {nm_sys.std():.2f} | {len(nm_rows_mk)} |\n")

                    summary["three_way"][f"{mk}_no_match"] = {
                        "self_ref_attn": 0.0,
                        "full_image_attn": round(nm_img.mean(), 4),
                        "n": len(nm_rows_mk),
                    }

            md.write("\n")

            # 8.1 — Full attention profile comparison (all 5 regions)
            md.write("### 8.1 Full Attention Profile (all regions)\n\n")
            md.write("| Model | Token Type | KW Prompt % | Other Prompt % | Full Image % | Prior Gen % | System % | N |\n")
            md.write("|-------|-----------|------------|----------------|-------------|------------|---------|---|\n")

            for mk in MODEL_KEYS:
                kw_rows_mk = [r for r in all_layer if r["model"] == mk]
                pm_rows_mk = [r for r in prompt_match_rows if r["model"] == mk]
                nm_rows_mk = [r for r in no_match_rows if r["model"] == mk]

                for label, src in [("Keyword", kw_rows_mk), ("Prompt-match", pm_rows_mk), ("No-match", nm_rows_mk)]:
                    if not src:
                        continue
                    kw_p = np.array([r["keyword_prompt_attn"] for r in src])
                    op = np.array([r["other_prompt_attn"] for r in src])
                    fi = np.array([r["full_image_attn"] for r in src])
                    pg = np.array([r["prior_gen_attn"] for r in src])
                    sy = np.array([r["system_attn"] for r in src])
                    md.write(f"| {MODEL_LABELS[mk]} | {label} | "
                             f"{kw_p.mean():.2f} +/- {kw_p.std():.2f} | "
                             f"{op.mean():.2f} +/- {op.std():.2f} | "
                             f"{fi.mean():.2f} +/- {fi.std():.2f} | "
                             f"{pg.mean():.2f} +/- {pg.std():.2f} | "
                             f"{sy.mean():.2f} +/- {sy.std():.2f} | {len(src)} |\n")
            md.write("\n")

            # 8.2 — Statistical tests: self-referencing attention
            md.write("### 8.2 Statistical Tests: Self-Referencing Attention (Mann-Whitney U)\n\n")
            md.write("Tests whether keyword tokens show more self-referencing than function "
                     "words that also appear in the prompt.\n\n")
            md.write("| Model | Comparison | U-stat | p-value | Mean A | Mean B | Direction |\n")
            md.write("|-------|-----------|--------|---------|--------|--------|----------|\n")

            for mk in MODEL_KEYS:
                kw_rows_mk = [r for r in all_layer if r["model"] == mk]
                pm_rows_mk = [r for r in prompt_match_rows if r["model"] == mk]
                nm_rows_mk = [r for r in no_match_rows if r["model"] == mk]

                # Test 1: Keyword self-ref vs Prompt-match self-ref
                if kw_rows_mk and pm_rows_mk:
                    vk = np.array([r["keyword_prompt_attn"] for r in kw_rows_mk])
                    vp = np.array([r["matched_prompt_attn"] for r in pm_rows_mk])
                    if len(vk) >= 2 and len(vp) >= 2:
                        u_stat, p_val = stats.mannwhitneyu(vk, vp, alternative="two-sided")
                        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        direction = "KW > PM" if vk.mean() > vp.mean() else "PM > KW"
                        md.write(f"| {MODEL_LABELS[mk]} | Keyword vs Prompt-match (self-ref) | "
                                 f"{u_stat:.0f} | {p_val:.4e} | {vk.mean():.4f} | {vp.mean():.4f} | {direction} {sig} |\n")

                # Test 2: Prompt-match self-ref vs 0 (proxy: prompt-match vs no-match keyword_prompt_attn)
                # Actually test: does prompt-match have higher attention to its matched prompt token
                # than no-match has to ANY prompt token? (matched_prompt_attn vs 0)
                if pm_rows_mk:
                    vp = np.array([r["matched_prompt_attn"] for r in pm_rows_mk])
                    # One-sample: is matched_prompt_attn > 0?
                    if len(vp) >= 2 and vp.mean() > 0:
                        # Use Wilcoxon signed-rank (test median > 0)
                        # Filter out exact zeros for the test
                        vp_nonzero = vp[vp > 0]
                        if len(vp_nonzero) >= 2:
                            w_stat, p_val = stats.wilcoxon(vp_nonzero, alternative="greater")
                            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                            md.write(f"| {MODEL_LABELS[mk]} | Prompt-match self-ref > 0 (Wilcoxon) | "
                                     f"{w_stat:.0f} | {p_val:.4e} | {vp.mean():.4f} | 0.0000 | PM > 0 {sig} |\n")

                # Test 3: Keyword vs Prompt-match on full_image_attn
                if kw_rows_mk and pm_rows_mk:
                    vk_img = np.array([r["full_image_attn"] for r in kw_rows_mk])
                    vp_img = np.array([r["full_image_attn"] for r in pm_rows_mk])
                    if len(vk_img) >= 2 and len(vp_img) >= 2:
                        u_stat, p_val = stats.mannwhitneyu(vk_img, vp_img, alternative="two-sided")
                        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        direction = "KW > PM" if vk_img.mean() > vp_img.mean() else "PM > KW"
                        md.write(f"| {MODEL_LABELS[mk]} | Keyword vs Prompt-match (image attn) | "
                                 f"{u_stat:.0f} | {p_val:.4e} | {vk_img.mean():.4f} | {vp_img.mean():.4f} | {direction} {sig} |\n")

            md.write("\n")

            # 8.3 — Top prompt-matching function words
            md.write("### 8.3 Most Common Prompt-Matching Non-Keyword Tokens\n\n")
            md.write("Shows the most frequent non-keyword generated tokens that match a prompt token, "
                     "and their average self-referencing attention.\n\n")

            for mk in MODEL_KEYS:
                pm_rows_mk = [r for r in prompt_match_rows if r["model"] == mk]
                if not pm_rows_mk:
                    continue

                md.write(f"**{MODEL_LABELS[mk]}**\n\n")
                md.write("| Token | Count | Self-Ref Attn % | Full Image % | Prior Gen % |\n")
                md.write("|-------|-------|----------------|-------------|------------|\n")

                # Group by token text
                token_groups = defaultdict(list)
                for r in pm_rows_mk:
                    token_groups[r["gen_token_text"]].append(r)

                # Sort by count, show top 15
                sorted_tokens = sorted(token_groups.items(), key=lambda x: -len(x[1]))[:15]
                for tok_text, tok_rows in sorted_tokens:
                    sr = np.array([r["matched_prompt_attn"] for r in tok_rows])
                    fi = np.array([r["full_image_attn"] for r in tok_rows])
                    pg = np.array([r["prior_gen_attn"] for r in tok_rows])
                    md.write(f"| '{tok_text}' | {len(tok_rows)} | "
                             f"{sr.mean():.3f} +/- {sr.std():.3f} | "
                             f"{fi.mean():.2f} +/- {fi.std():.2f} | "
                             f"{pg.mean():.2f} +/- {pg.std():.2f} |\n")
                md.write("\n")

        # ── 9. Sliding Window: Attention Regime Trajectory ──
        N_BINS = 10
        if position_rows:
            md.write("## 9. Attention Regime Trajectory (Sliding Window)\n\n")
            md.write("How does the attention profile change across generation? "
                     "Each generated token is assigned to a normalized position bin "
                     f"(0-100% of generation length, {N_BINS} bins). "
                     "This reveals whether the model front-loads image attention early "
                     "and shifts to prior-gen attention later, or uses distinct "
                     "'attention regimes' during generation.\n\n")

            # 9.0 — Per-model trajectory table
            md.write("### 9.0 Attention Profile by Normalized Position\n\n")

            summary["sliding_window"] = {}
            for mk in MODEL_KEYS:
                mk_rows = [r for r in position_rows if r["model"] == mk]
                if not mk_rows:
                    continue

                md.write(f"**{MODEL_LABELS[mk]}** (N={len(mk_rows)} tokens)\n\n")
                md.write("| Bin (%) | Image % | Prompt % | Prior Gen % | System % | N | Dom. Phase |\n")
                md.write("|---------|---------|----------|-------------|----------|---|------------|\n")

                bin_data = defaultdict(list)
                for r in mk_rows:
                    b = min(int(r["gen_idx"] / r["n_gen"] * N_BINS), N_BINS - 1)
                    bin_data[b].append(r)

                model_bins = {}
                for b in range(N_BINS):
                    rows_b = bin_data.get(b, [])
                    if not rows_b:
                        md.write(f"| {b*10}-{(b+1)*10} | — | — | — | — | 0 | — |\n")
                        continue
                    img = np.array([r["image_attn"] for r in rows_b])
                    prm = np.array([r["prompt_attn"] for r in rows_b])
                    pg = np.array([r["prior_gen_attn"] for r in rows_b])
                    sy = np.array([r["system_attn"] for r in rows_b])
                    # Dominant phase
                    phases = [r["phase"] for r in rows_b]
                    think_frac = sum(1 for p in phases if p == "thinking") / len(phases)
                    if think_frac > 0.8:
                        dom_phase = "thinking"
                    elif think_frac < 0.2:
                        dom_phase = "answer"
                    else:
                        dom_phase = f"mixed ({think_frac:.0%} think)"

                    md.write(f"| {b*10}-{(b+1)*10} | "
                             f"{img.mean():.2f} +/- {img.std():.2f} | "
                             f"{prm.mean():.2f} +/- {prm.std():.2f} | "
                             f"{pg.mean():.2f} +/- {pg.std():.2f} | "
                             f"{sy.mean():.2f} +/- {sy.std():.2f} | "
                             f"{len(rows_b)} | {dom_phase} |\n")

                    model_bins[b] = {
                        "image_attn": round(img.mean(), 4),
                        "prompt_attn": round(prm.mean(), 4),
                        "prior_gen_attn": round(pg.mean(), 4),
                        "system_attn": round(sy.mean(), 4),
                        "n": len(rows_b),
                        "think_frac": round(think_frac, 4),
                    }

                summary["sliding_window"][mk] = model_bins
                md.write("\n")

            # 9.1 — Think→Answer boundary position
            md.write("### 9.1 Think→Answer Boundary Position\n\n")
            md.write("For Thinking models, where does the phase transition fall "
                     "in normalized generation position?\n\n")
            md.write("| Model | Question | Boundary (norm %) | Think Tokens | Answer Tokens | Total |\n")
            md.write("|-------|---------|-------------------|-------------|---------------|-------|\n")

            summary["think_boundary"] = {}
            for mk in [m for m in MODEL_KEYS if "thinking" in m]:
                mk_rows = [r for r in position_rows if r["model"] == mk]
                if not mk_rows:
                    continue

                # Group by question
                by_q = defaultdict(list)
                for r in mk_rows:
                    by_q[r["qid"]].append(r)

                boundaries = []
                for qid in sorted(by_q.keys()):
                    q_rows = sorted(by_q[qid], key=lambda r: r["gen_idx"])
                    n_gen = q_rows[0]["n_gen"]
                    n_think = sum(1 for r in q_rows if r["phase"] == "thinking")
                    n_answer = sum(1 for r in q_rows if r["phase"] == "answer")
                    boundary_pct = n_think / n_gen * 100 if n_gen > 0 else 0
                    boundaries.append(boundary_pct)

                    md.write(f"| {MODEL_LABELS[mk]} | {qid} | {boundary_pct:.1f}% | "
                             f"{n_think} | {n_answer} | {n_gen} |\n")

                if boundaries:
                    b_arr = np.array(boundaries)
                    md.write(f"| **{MODEL_LABELS[mk]}** | **Mean** | **{b_arr.mean():.1f}% +/- {b_arr.std():.1f}%** | "
                             f"— | — | — |\n")
                    summary["think_boundary"][mk] = {
                        "mean_pct": round(b_arr.mean(), 2),
                        "std_pct": round(b_arr.std(), 2),
                    }

            md.write("\n")

            # 9.2 — Monotonicity tests (Spearman: region vs position)
            md.write("### 9.2 Monotonicity Tests (Spearman: attention region vs position)\n\n")
            md.write("Positive rho = attention increases as generation progresses; "
                     "negative = decreases.\n\n")
            md.write("| Model | Region | Spearman rho | p-value | Trend |\n")
            md.write("|-------|--------|-------------|---------|-------|\n")

            summary["sliding_window_spearman"] = {}
            regions_test = [("image_attn", "Image"), ("prompt_attn", "Prompt"),
                            ("prior_gen_attn", "Prior Gen"), ("system_attn", "System")]

            for mk in MODEL_KEYS:
                mk_rows = [r for r in position_rows if r["model"] == mk]
                if len(mk_rows) < 10:
                    continue
                norm_pos = np.array([r["gen_idx"] / r["n_gen"] for r in mk_rows])

                for reg_key, reg_label in regions_test:
                    vals = np.array([r[reg_key] for r in mk_rows])
                    rho, p_val = stats.spearmanr(norm_pos, vals)
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    trend = "increases" if rho > 0 else "decreases"
                    md.write(f"| {MODEL_LABELS[mk]} | {reg_label} | "
                             f"{rho:.4f} | {p_val:.4e} | {trend} {sig} |\n")
                    summary["sliding_window_spearman"][f"{mk}_{reg_key}"] = {
                        "rho": round(float(rho), 4),
                        "p_val": round(float(p_val), 6),
                    }

            md.write("\n")

            # 9.3 — Thinking-only vs Answer-only trajectory (for Thinking models)
            for mk in [m for m in MODEL_KEYS if "thinking" in m]:
                mk_rows = [r for r in position_rows if r["model"] == mk]
                if not mk_rows:
                    continue

                md.write(f"### 9.3 Phase-Specific Trajectory — {MODEL_LABELS[mk]}\n\n")
                md.write("Sliding window within each phase separately "
                         "(normalized to phase length, not full generation).\n\n")

                for phase_name in ["thinking", "answer"]:
                    phase_rows = [r for r in mk_rows if r["phase"] == phase_name]
                    if len(phase_rows) < 5:
                        continue

                    md.write(f"**{phase_name.title()} phase** (N={len(phase_rows)})\n\n")
                    md.write("| Bin (%) | Image % | Prompt % | Prior Gen % | System % | N |\n")
                    md.write("|---------|---------|----------|-------------|----------|---|\n")

                    # Normalize within-phase position per question
                    by_q = defaultdict(list)
                    for r in phase_rows:
                        by_q[r["qid"]].append(r)

                    phase_binned = defaultdict(list)
                    for qid, q_rows in by_q.items():
                        q_sorted = sorted(q_rows, key=lambda r: r["gen_idx"])
                        n_phase = len(q_sorted)
                        for pi, r in enumerate(q_sorted):
                            b = min(int(pi / n_phase * N_BINS), N_BINS - 1)
                            phase_binned[b].append(r)

                    for b in range(N_BINS):
                        rows_b = phase_binned.get(b, [])
                        if not rows_b:
                            continue
                        img = np.mean([r["image_attn"] for r in rows_b])
                        prm = np.mean([r["prompt_attn"] for r in rows_b])
                        pg = np.mean([r["prior_gen_attn"] for r in rows_b])
                        sy = np.mean([r["system_attn"] for r in rows_b])
                        md.write(f"| {b*10}-{(b+1)*10} | {img:.2f} | {prm:.2f} | "
                                 f"{pg:.2f} | {sy:.2f} | {len(rows_b)} |\n")

                    md.write("\n")

    print(f"Summary written: {md_path}")

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON written: {json_path}")


# ── Visualization ─────────────────────────────────────────────────────

def plot_attention_decomposition(rows, output_dir):
    """Stacked bar chart: 5-region decomposition per model."""
    all_layer = [r for r in rows if r["layer"] == "all"]
    if not all_layer:
        return

    n_models = len(MODEL_KEYS)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]
    fig.suptitle("Attention Decomposition: Where Do Keyword Tokens Attend?", fontsize=14)

    regions = ["keyword_prompt_attn", "other_prompt_attn", "full_image_attn", "prior_gen_attn", "system_attn"]
    labels = ["Keyword in Prompt", "Other Prompt", "Full Image", "Prior Generated", "System"]
    colors = ["#E74C3C", "#F39C12", "#3498DB", "#9B59B6", "#95A5A6"]

    for mi, mk in enumerate(MODEL_KEYS):
        ax = axes[mi]
        model_rows = [r for r in all_layer if r["model"] == mk]
        if not model_rows:
            continue

        means = [np.mean([r[reg] for r in model_rows]) for reg in regions]
        stds = [np.std([r[reg] for r in model_rows]) for reg in regions]

        # Stacked bar
        bottom = 0
        bars = []
        for i, (m, s, label, color) in enumerate(zip(means, stds, labels, colors)):
            bar = ax.bar(0, m, bottom=bottom, color=color, label=label, width=0.5)
            # Add text label if > 2%
            if m > 2:
                ax.text(0, bottom + m / 2, f"{m:.1f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white" if m > 5 else "black")
            bottom += m
            bars.append(bar)

        ax.set_title(MODEL_LABELS[mk], fontsize=12)
        ax.set_ylabel("Attention %")
        ax.set_ylim(0, 105)
        ax.set_xticks([])
        if mi == 0:
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "plots", "attention_decomposition_stacked.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot: {path}")


def plot_rapt_comparison(rows, output_dir):
    """Grouped bar chart comparing RAPT values across models."""
    all_layer = [r for r in rows if r["layer"] == "all"]
    if not all_layer:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = ["keyword_prompt_rapt", "full_image_rapt", "prior_gen_rapt"]
    labels = ["Keyword Prompt", "Full Image", "Prior Generated"]
    x = np.arange(len(labels))
    active_models = [mk for mk in MODEL_KEYS if any(r["model"] == mk for r in all_layer)]
    n_active = len(active_models)
    width = 0.8 / max(n_active, 1)
    colors = {"instruct": "#3498DB", "thinking": "#E74C3C",
              "instruct_30b": "#2ECC71", "thinking_30b": "#9B59B6"}

    for mi, mk in enumerate(active_models):
        model_rows = [r for r in all_layer if r["model"] == mk]
        if not model_rows:
            continue
        means = [np.mean([r[m] for r in model_rows if not np.isnan(r[m])]) for m in metrics]
        stds = [np.std([r[m] for r in model_rows if not np.isnan(r[m])]) for m in metrics]
        offset = -0.4 + width * (mi + 0.5)
        ax.bar(x + offset, means, width, yerr=stds, label=MODEL_LABELS[mk],
               color=colors.get(mk, "#777"), alpha=0.8, capsize=3)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="RAPT = 1 (average)")
    ax.set_xlabel("Attention Region")
    ax.set_ylabel("RAPT (Relative Attention Per Token)")
    ax.set_title("RAPT Comparison: All Models")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "plots", "rapt_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot: {path}")


def plot_layer_trends(rows, output_dir):
    """Per-layer RAPT trends for keyword prompt vs image attention."""
    layer_rows = [r for r in rows if r["layer"] != "all"]
    if not layer_rows:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Layer-wise Attention Trends for Keyword Tokens", fontsize=14)

    metrics = [
        ("keyword_prompt_attn", "full_image_attn"),
        ("keyword_prompt_rapt", "full_image_rapt"),
    ]
    titles = ["Raw Attention %", "RAPT-Normalized"]
    colors_kw = {"instruct": "#E74C3C", "thinking": "#FF6B6B",
                  "instruct_30b": "#27AE60", "thinking_30b": "#8E44AD"}
    colors_img = {"instruct": "#2980B9", "thinking": "#5DADE2",
                  "instruct_30b": "#1ABC9C", "thinking_30b": "#D35400"}
    linestyles = {"instruct": "--", "thinking": "-",
                  "instruct_30b": ":", "thinking_30b": "-."}

    for ai, ((m_kw, m_img), title) in enumerate(zip(metrics, titles)):
        ax = axes[ai]
        for mk in MODEL_KEYS:
            mk_rows = [r for r in layer_rows if r["model"] == mk]
            if not mk_rows:
                continue

            layer_vals_kw = defaultdict(list)
            layer_vals_img = defaultdict(list)
            for r in mk_rows:
                l = int(r["layer"])
                if not np.isnan(r[m_kw]):
                    layer_vals_kw[l].append(r[m_kw])
                if not np.isnan(r[m_img]):
                    layer_vals_img[l].append(r[m_img])

            layers = sorted(layer_vals_kw.keys())
            kw_means = [np.mean(layer_vals_kw[l]) for l in layers]
            img_means = [np.mean(layer_vals_img[l]) for l in layers]

            ls = linestyles.get(mk, "-")
            ax.plot(layers, kw_means, ls, color=colors_kw.get(mk, "#777"),
                    label=f"{MODEL_LABELS[mk]} - Keyword Prompt", linewidth=1.5, alpha=0.8)
            ax.plot(layers, img_means, ls, color=colors_img.get(mk, "#777"),
                    label=f"{MODEL_LABELS[mk]} - Full Image", linewidth=1.5, alpha=0.8)

        ax.set_xlabel("Layer")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "plots", "layer_trend_keyword_vs_image.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot: {path}")


def plot_phase_comparison(rows, output_dir):
    """Thinking vs Answer phase comparison for Thinking models."""
    think_models = [mk for mk in MODEL_KEYS if "thinking" in mk]
    for think_mk in think_models:
        all_layer = [r for r in rows if r["layer"] == "all" and r["model"] == think_mk]
        if not all_layer:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Thinking vs Answer Phase: Keyword Token Attention ({MODEL_LABELS[think_mk]})", fontsize=14)

        # Left: raw attention decomposition
        ax = axes[0]
        regions = ["keyword_prompt_attn", "other_prompt_attn", "full_image_attn", "prior_gen_attn", "system_attn"]
        labels = ["KW Prompt", "Other Prompt", "Image", "Prior Gen", "System"]

        x = np.arange(len(labels))
        width = 0.35
        for pi, phase in enumerate(["thinking", "answer"]):
            phase_rows = [r for r in all_layer if r["phase"] == phase]
            if not phase_rows:
                continue
            means = [np.mean([r[reg] for r in phase_rows]) for reg in regions]
            offset = -width / 2 + pi * width
            ax.bar(x + offset, means, width, label=phase.title(),
                   color=("#FF9800" if phase == "thinking" else "#4CAF50"), alpha=0.8)

        ax.set_ylabel("Attention %")
        ax.set_title("Raw Attention Decomposition")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Right: RAPT
        ax = axes[1]
        rapt_metrics = ["keyword_prompt_rapt", "full_image_rapt", "prior_gen_rapt"]
        rapt_labels = ["KW Prompt RAPT", "Image RAPT", "Prior Gen RAPT"]
        x = np.arange(len(rapt_labels))

        for pi, phase in enumerate(["thinking", "answer"]):
            phase_rows = [r for r in all_layer if r["phase"] == phase]
            if not phase_rows:
                continue
            means = [np.mean([r[m] for r in phase_rows if not np.isnan(r[m])]) for m in rapt_metrics]
            offset = -width / 2 + pi * width
            ax.bar(x + offset, means, width, label=phase.title(),
                   color=("#FF9800" if phase == "thinking" else "#4CAF50"), alpha=0.8)

        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("RAPT")
        ax.set_title("RAPT-Normalized")
        ax.set_xticks(x)
        ax.set_xticklabels(rapt_labels, rotation=30, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        suffix = "_30b" if "30b" in think_mk else ""
        path = os.path.join(output_dir, "plots", f"phase_comparison{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot: {path}")


def plot_per_question_heatmap(rows, output_dir):
    """Heatmap: questions x metrics for each model."""
    all_layer = [r for r in rows if r["layer"] == "all"]
    if not all_layer:
        return

    metrics = ["keyword_prompt_attn", "full_image_attn", "prior_gen_attn", "keyword_prompt_rapt", "full_image_rapt", "soft_iou"]
    metric_labels = ["KW Prompt %", "Image %", "Prior Gen %", "KW RAPT", "Image RAPT", "Soft IoU"]

    for mk in MODEL_KEYS:
        model_rows = [r for r in all_layer if r["model"] == mk]
        if not model_rows:
            continue

        # Group by qid+keyword
        groups = defaultdict(list)
        for r in model_rows:
            groups[(r["qid"], r["keyword"])].append(r)

        if not groups:
            continue

        sorted_keys = sorted(groups.keys())
        ylabels = [f"{qid} ({kw})" for qid, kw in sorted_keys]
        data = np.zeros((len(sorted_keys), len(metrics)))

        for gi, key in enumerate(sorted_keys):
            for mi, metric in enumerate(metrics):
                vals = [r[metric] for r in groups[key] if not np.isnan(r[metric])]
                data[gi, mi] = np.mean(vals) if vals else 0

        fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_keys) * 0.4)))
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd")

        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metric_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=8)

        # Annotate cells
        for gi in range(len(sorted_keys)):
            for mi in range(len(metrics)):
                val = data[gi, mi]
                ax.text(mi, gi, f"{val:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if val > data.max() * 0.6 else "black")

        ax.set_title(f"Per-Question Keyword Attention — {MODEL_LABELS[mk]}")
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        path = os.path.join(output_dir, "plots", f"per_question_heatmap_{mk}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot: {path}")


def plot_per_keyword_breakdown(rows, output_dir):
    """Individual charts per keyword showing attention over generation order."""
    all_layer = [r for r in rows if r["layer"] == "all"]
    if not all_layer:
        return

    # Group by keyword
    by_keyword = defaultdict(list)
    for r in all_layer:
        by_keyword[r["keyword"]].append(r)

    for keyword, kw_rows in sorted(by_keyword.items()):
        active_models = [mk for mk in MODEL_KEYS if any(r["model"] == mk for r in kw_rows)]
        n_active = max(len(active_models), 1)
        fig, axes = plt.subplots(1, n_active, figsize=(7 * n_active, 5))
        if n_active == 1:
            axes = [axes]
        fig.suptitle(f'Keyword: "{keyword}" — Attention Over Generation Order', fontsize=13)

        for mi, mk in enumerate(active_models):
            ax = axes[mi]
            mk_rows = sorted([r for r in kw_rows if r["model"] == mk],
                             key=lambda r: r["gen_token_idx"])
            if not mk_rows:
                ax.set_title(f"{MODEL_LABELS[mk]} — No matches")
                continue

            x = range(len(mk_rows))
            kw_attn = [r["keyword_prompt_attn"] for r in mk_rows]
            img_attn = [r["full_image_attn"] for r in mk_rows]
            gen_attn = [r["prior_gen_attn"] for r in mk_rows]

            ax.plot(x, kw_attn, "o-", color="#E74C3C", label="KW Prompt", markersize=3, linewidth=1.5)
            ax.plot(x, img_attn, "s-", color="#3498DB", label="Full Image", markersize=3, linewidth=1.5)
            ax.plot(x, gen_attn, "^-", color="#9B59B6", label="Prior Gen", markersize=3, linewidth=1.5)

            # Mark phase boundaries for thinking models
            if "thinking" in mk:
                for i, r in enumerate(mk_rows):
                    if i > 0 and mk_rows[i - 1]["phase"] != r["phase"]:
                        ax.axvline(x=i, color="gray", linestyle="--", alpha=0.5)
                        ax.text(i, ax.get_ylim()[1] * 0.95, f" {r['phase']}",
                                fontsize=7, color="gray")

            ax.set_title(MODEL_LABELS[mk])
            ax.set_xlabel("Token Order (within keyword matches)")
            ax.set_ylabel("Attention %")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        safe_kw = keyword.replace(" ", "_")
        path = os.path.join(output_dir, "plots", "per_keyword_breakdown", f"{safe_kw}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot: {path}")


def plot_sliding_window(position_rows, output_dir):
    """Stacked area chart: attention regions across normalized generation position."""
    if not position_rows:
        return

    N_BINS = 10
    regions = ["image_attn", "prompt_attn", "prior_gen_attn", "system_attn"]
    labels = ["Image", "Prompt", "Prior Generated", "System"]
    colors = ["#3498DB", "#F39C12", "#9B59B6", "#95A5A6"]

    active_models = [mk for mk in MODEL_KEYS if any(r["model"] == mk for r in position_rows)]
    n_models = len(active_models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]
    fig.suptitle("Attention Regime Trajectory Across Generation", fontsize=14, y=1.02)

    for mi, mk in enumerate(active_models):
        ax = axes[mi]
        mk_rows = [r for r in position_rows if r["model"] == mk]
        if not mk_rows:
            continue

        bin_data = defaultdict(list)
        for r in mk_rows:
            b = min(int(r["gen_idx"] / r["n_gen"] * N_BINS), N_BINS - 1)
            bin_data[b].append(r)

        x = []
        means = {reg: [] for reg in regions}
        for b in range(N_BINS):
            rows_b = bin_data.get(b, [])
            if not rows_b:
                continue
            x.append(b * 10 + 5)  # bin center
            for reg in regions:
                means[reg].append(np.mean([r[reg] for r in rows_b]))

        if not x:
            continue

        # Stacked area
        y = np.array([means[reg] for reg in regions])
        ax.stackplot(x, y, labels=labels, colors=colors, alpha=0.8)

        # Mark think→answer boundary for thinking models
        if "thinking" in mk:
            by_q = defaultdict(list)
            for r in mk_rows:
                by_q[r["qid"]].append(r)
            boundaries = []
            for qid, q_rows in by_q.items():
                n_think = sum(1 for r in q_rows if r["phase"] == "thinking")
                n_gen = q_rows[0]["n_gen"]
                boundaries.append(n_think / n_gen * 100)
            if boundaries:
                mean_boundary = np.mean(boundaries)
                ax.axvline(x=mean_boundary, color="red", linestyle="--", linewidth=2, alpha=0.8)
                ax.text(mean_boundary + 1, 5, f"think→ans\n({mean_boundary:.0f}%)",
                        fontsize=7, color="red", va="bottom")

        ax.set_title(MODEL_LABELS[mk], fontsize=11)
        ax.set_xlabel("Generation Position (%)")
        if mi == 0:
            ax.set_ylabel("Attention %")
            ax.legend(loc="upper left", fontsize=7)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    path = os.path.join(output_dir, "plots", "sliding_window_trajectory.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot: {path}")

    # Also plot line chart (not stacked) for clearer trend reading
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]
    fig.suptitle("Attention Region Trends Across Generation", fontsize=14, y=1.02)

    for mi, mk in enumerate(active_models):
        ax = axes[mi]
        mk_rows = [r for r in position_rows if r["model"] == mk]
        if not mk_rows:
            continue

        bin_data = defaultdict(list)
        for r in mk_rows:
            b = min(int(r["gen_idx"] / r["n_gen"] * N_BINS), N_BINS - 1)
            bin_data[b].append(r)

        x = []
        means = {reg: [] for reg in regions}
        stds = {reg: [] for reg in regions}
        for b in range(N_BINS):
            rows_b = bin_data.get(b, [])
            if not rows_b:
                continue
            x.append(b * 10 + 5)
            for reg in regions:
                vals = [r[reg] for r in rows_b]
                means[reg].append(np.mean(vals))
                stds[reg].append(np.std(vals) / np.sqrt(len(vals)))  # SEM

        if not x:
            continue

        for reg, label, color in zip(regions, labels, colors):
            ax.plot(x, means[reg], "o-", color=color, label=label,
                    markersize=4, linewidth=2)
            ax.fill_between(x,
                            np.array(means[reg]) - np.array(stds[reg]),
                            np.array(means[reg]) + np.array(stds[reg]),
                            color=color, alpha=0.15)

        if "thinking" in mk:
            by_q = defaultdict(list)
            for r in mk_rows:
                by_q[r["qid"]].append(r)
            boundaries = []
            for qid, q_rows in by_q.items():
                n_think = sum(1 for r in q_rows if r["phase"] == "thinking")
                n_gen = q_rows[0]["n_gen"]
                boundaries.append(n_think / n_gen * 100)
            if boundaries:
                mean_boundary = np.mean(boundaries)
                ax.axvline(x=mean_boundary, color="red", linestyle="--",
                           linewidth=2, alpha=0.8, label=f"think→ans ({mean_boundary:.0f}%)")

        ax.set_title(MODEL_LABELS[mk], fontsize=11)
        ax.set_xlabel("Generation Position (%)")
        if mi == 0:
            ax.set_ylabel("Attention %")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "plots", "sliding_window_lines.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot: {path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prompt keyword attention experiment")
    parser.add_argument("--output_dir", default="results/prompt_keyword_attention",
                        help="Output directory")
    parser.add_argument("--manifest", default=None,
                        help="Path to manifest.json (default: data/manifest.json)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    manifest_path = args.manifest or os.path.join(base_dir, "data", "manifest.json")

    print(f"Manifest: {manifest_path}")
    print(f"Output:   {args.output_dir}")
    print(f"PT base:  {PT_BASE}")
    print()

    # Run analysis
    all_rows, baseline_rows, manifest, position_rows = analyze_all(manifest_path, args.output_dir)
    print(f"\nTotal rows: {len(all_rows)}")
    print(f"Baseline (non-keyword) rows: {len(baseline_rows)}")
    print(f"Position (all gen tokens) rows: {len(position_rows)}")

    # Write outputs
    write_csv(all_rows, args.output_dir)
    write_summary(all_rows, args.output_dir, manifest, baseline_rows, position_rows)

    # Generate plots
    print("\nGenerating plots...")
    plot_attention_decomposition(all_rows, args.output_dir)
    plot_rapt_comparison(all_rows, args.output_dir)
    plot_layer_trends(all_rows, args.output_dir)
    plot_phase_comparison(all_rows, args.output_dir)
    plot_per_question_heatmap(all_rows, args.output_dir)
    plot_per_keyword_breakdown(all_rows, args.output_dir)
    plot_sliding_window(position_rows, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
