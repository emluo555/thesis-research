"""
precompute.py - Convert raw contributions.pt data into JSON format for the Data Explorer.

Reads existing contributions.pt files (from analyze_cot_attribution.py) and DriVQA
question metadata, then produces:
  - data/manifest.json  (master index)
  - data/precomputed/q{NNNN}_thinking.json  (per-question token data)
  - data/images/q{NNNN}.jpg  (symlinked source images)

Reuses load_contributions() and query_token() from scripts/analyze_cot_attribution.py.
"""

import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# Cross-module imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_cot_attribution import load_contributions, query_token, load_questions

# ── Config ────────────────────────────────────────────────────────

CONTRIBUTIONS_DIR = ROOT / "DriVQA" / "outputs" / "cot_attribution_new"
HUMAN_MAPS_DIR = ROOT / "DriVQA" / "outputs" / "human_maps"
QUESTIONS_JSON = ROOT / "DriVQA" / "data" / "drivqa_questions.json"
IMAGES_DIR = ROOT / "DriVQA" / "data" / "Images"
OUT_DIR = Path(__file__).resolve().parent / "data"

FLOAT_PRECISION = 4


def infer_patch_grid(num_image_tokens: int, img_width: int, img_height: int) -> tuple:
    """Infer the (rows, cols) patch grid from the number of image tokens and aspect ratio.

    Qwen3-VL uses dynamic resolution, so the grid varies per image. We find the
    factorization of num_image_tokens whose aspect ratio best matches the image.
    """
    aspect = img_width / img_height
    best = None
    best_err = float("inf")
    for r in range(1, int(math.sqrt(num_image_tokens)) + 1):
        if num_image_tokens % r == 0:
            c = num_image_tokens // r
            err = abs(c / r - aspect)
            if err < best_err:
                best_err = err
                best = (r, c)
    return best


def downsample_heatmap(heatmap: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Downsample a full-resolution heatmap to the patch grid by block-averaging.

    heatmap: shape (img_h, img_w), e.g. (900, 1600)
    Returns: flat array of length grid_h * grid_w, row-major.
    """
    img_h, img_w = heatmap.shape
    # Resize using block averaging
    block_h = img_h / grid_h
    block_w = img_w / grid_w
    result = np.zeros((grid_h, grid_w), dtype=np.float64)
    for r in range(grid_h):
        r_start = int(round(r * block_h))
        r_end = int(round((r + 1) * block_h))
        for c in range(grid_w):
            c_start = int(round(c * block_w))
            c_end = int(round((c + 1) * block_w))
            result[r, c] = heatmap[r_start:r_end, c_start:c_end].mean()
    # Normalize to sum to 1
    total = result.sum()
    if total > 0:
        result /= total
    return result.flatten()


def process_question(qid: str, question_info: dict, out_idx: int) -> dict:
    """Process a single question's contributions.pt into explorer JSON format.

    Returns the manifest entry for this question.
    """
    contrib_path = CONTRIBUTIONS_DIR / qid / "contributions.pt"
    if not contrib_path.exists():
        print(f"  [SKIP] {qid}: no contributions.pt")
        return None

    data = load_contributions(str(contrib_path))
    regions = data["token_regions"]
    input_len = data["input_len"]
    num_generated = data["num_generated"]
    token_strings = data["token_strings"]

    # Image info
    camera, num = qid.rsplit("_", 1)
    img_filename = f"{camera}/{num}.jpg"
    img_path = IMAGES_DIR / camera / f"{num}.jpg"
    if not img_path.exists():
        print(f"  [SKIP] {qid}: image not found at {img_path}")
        return None

    from PIL import Image
    img = Image.open(img_path)
    img_width, img_height = img.size

    # Patch grid
    num_image_tokens = regions["image_end"] - regions["image_start"]
    grid_h, grid_w = infer_patch_grid(num_image_tokens, img_width, img_height)
    print(f"  {qid}: {num_image_tokens} image tokens → {grid_h}×{grid_w} grid, "
          f"{num_generated} generated tokens")

    # Token regions for the explorer
    gen_start = regions["generation_start"]
    think_start = regions.get("think_start")
    think_end = regions.get("think_end")
    answer_start = regions.get("answer_start")

    # Build generated token list with phase labels
    generated_tokens = []
    for i in range(num_generated):
        abs_idx = input_len + i
        tok_str = token_strings[abs_idx] if abs_idx < len(token_strings) else ""
        phase = "answer"
        if think_start is not None and think_end is not None:
            if abs_idx >= think_start and abs_idx <= think_end:
                phase = "thinking"
        generated_tokens.append({"text": tok_str, "phase": phase})

    # Build per-token precomputed data
    token_data = []
    for gen_idx in range(num_generated):
        abs_idx = input_len + gen_idx
        result = query_token(data, abs_idx)

        # Extract image patch contributions
        img_start = regions["image_start"]
        img_end = regions["image_end"]

        patch_attention = []
        patch_tam = []
        text_attention = []
        text_tam = []

        if result["attn"] is not None:
            attn_vec = result["attn"]
            patch_attention = [round(float(v), FLOAT_PRECISION) for v in attn_vec[img_start:img_end]]
            # Text = prior generated tokens only
            text_attention = [round(float(v), FLOAT_PRECISION) for v in attn_vec[gen_start:gen_start + gen_idx]]

        if result["tam"] is not None:
            tam_vec = result["tam"]
            # TAM vector length grows with each step; image patch indices are the same
            if len(tam_vec) > img_start:
                end = min(img_end, len(tam_vec))
                patch_tam = [round(float(v), FLOAT_PRECISION) for v in tam_vec[img_start:end]]
                # Pad if TAM vector is shorter than image region (shouldn't happen for gen tokens)
                if len(patch_tam) < num_image_tokens:
                    patch_tam.extend([0.0] * (num_image_tokens - len(patch_tam)))
            # Text TAM = prior generated tokens
            gen_tam_start = gen_start
            gen_tam_end = gen_start + gen_idx
            if gen_tam_end <= len(tam_vec):
                text_tam = [round(float(v), FLOAT_PRECISION) for v in tam_vec[gen_tam_start:gen_tam_end]]

        token_data.append({
            "token_idx": gen_idx,
            "patch_attention": patch_attention,
            "patch_tam": patch_tam,
            "text_attention": text_attention,
            "text_tam": text_tam,
        })

    # Write per-question precomputed JSON
    q_label = f"q{out_idx:04d}"
    precomputed = {"token_data": token_data}
    precomputed_path = OUT_DIR / "precomputed" / f"{q_label}_thinking.json"
    with open(precomputed_path, "w") as f:
        json.dump(precomputed, f)
    precomputed_size_mb = precomputed_path.stat().st_size / 1e6
    print(f"    → {precomputed_path.name} ({precomputed_size_mb:.1f} MB)")

    # Copy image (symlink for efficiency)
    img_dest = OUT_DIR / "images" / f"{q_label}.jpg"
    if img_dest.exists():
        img_dest.unlink()
    shutil.copy2(img_path, img_dest)

    # Human heatmap
    human_heatmap = None
    density_path = HUMAN_MAPS_DIR / f"{qid}_density.npy"
    if density_path.exists():
        density = np.load(density_path)
        human_heatmap = [round(float(v), FLOAT_PRECISION) for v in
                         downsample_heatmap(density, grid_h, grid_w)]

    # Build manifest entry
    return {
        "id": q_label,
        "image_id": qid,
        "question_text": question_info["question"],
        "expected_answer": question_info["answer"],
        "image_path": f"data/images/{q_label}.jpg",
        "patch_grid": [grid_h, grid_w],
        "models": {
            "thinking": {
                "data_path": f"data/precomputed/{q_label}_thinking.json",
                "generated_tokens": generated_tokens,
            }
        },
        "human_heatmap": human_heatmap,
        "token_regions": {
            "think_start_idx": (think_start - input_len) if think_start is not None else None,
            "think_end_idx": (think_end - input_len) if think_end is not None else None,
            "answer_start_idx": (answer_start - input_len) if answer_start is not None else None,
        }
    }


def main():
    print("=== Data Explorer Precompute ===\n")

    # Load DriVQA questions
    questions = load_questions(str(QUESTIONS_JSON))
    print(f"Loaded {len(questions)} DriVQA questions")

    # Find available contributions
    available = sorted([
        d for d in os.listdir(CONTRIBUTIONS_DIR)
        if (CONTRIBUTIONS_DIR / d / "contributions.pt").exists()
    ])
    print(f"Found {len(available)} contributions.pt files: {available}\n")

    # Ensure output dirs exist
    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "precomputed").mkdir(parents=True, exist_ok=True)

    # Process each question
    manifest_entries = []
    for idx, qid in enumerate(available):
        if qid not in questions:
            print(f"  [SKIP] {qid}: not in DriVQA questions")
            continue
        entry = process_question(qid, questions[qid], idx + 1)
        if entry is not None:
            manifest_entries.append(entry)

    # Write manifest
    manifest = {
        "questions": manifest_entries,
        "notes": (
            "patch_grid is per-question (variable grid from Qwen3-VL dynamic resolution). "
            "Human heatmap is downsampled from full-resolution gaze density. "
            "Only Thinking model data available currently."
        ),
    }
    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n✓ Wrote {manifest_path} with {len(manifest_entries)} questions")
    print(f"  Total precomputed files: {len(manifest_entries)}")


if __name__ == "__main__":
    main()
