"""
analyze_bbox_tokens.py - Analyze bounding-box-matched tokens across all models and layers.

For each question × model × matched token, computes per-layer:
  - Soft IoU (attention vector vs box mask)
  - Attn% (fraction of image attention inside bounding box)
  - img↔text ratio (image attention sum / (image + text attention sum))

Outputs:
  - results/bbox_token_analysis.md   — markdown tables
  - results/bbox_token_graphs/       — per-object-type line charts over generation order

Usage:
    conda activate tam3
    python analyze_bbox_tokens.py [--output_dir results]
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- Synonym map (mirrors index.html SYNONYM_MAP) ---
SYNONYM_MAP = {
    'vehicle': ['vehicle', 'car', 'van', 'suv', 'truck', 'bus', 'motorcycle'],
    'vehicles': ['vehicle', 'car', 'van', 'suv', 'truck', 'bus', 'motorcycle'],
    'car': ['vehicle', 'car', 'van', 'suv', 'truck'],
    'cars': ['vehicle', 'car', 'van', 'suv', 'truck'],
    'van': ['vehicle', 'van', 'car', 'suv'],
    'suv': ['vehicle', 'suv', 'car', 'van'],
    'truck': ['vehicle', 'truck'],
    'trucks': ['vehicle', 'truck'],
    'bus': ['vehicle', 'bus'],
    'motorcycle': ['vehicle', 'motorcycle'],
    'bicycle': ['bicycle'],
    'pedestrian': ['pedestrian', 'person'],
    'pedestrians': ['pedestrian', 'person'],
    'person': ['pedestrian', 'person'],
    'people': ['pedestrian', 'person'],
    'tree': ['tree'],
    'trees': ['tree'],
    'building': ['building', 'store', 'restaurant'],
    'store': ['store', 'building', 'restaurant'],
    'restaurant': ['restaurant', 'store', 'building'],
    'road': ['road', 'lane'],
    'lane': ['lane', 'road'],
    'sidewalk': ['sidewalk', 'crosswalk'],
    'crosswalk': ['crosswalk', 'sidewalk'],
    'traffic': ['traffic light', 'traffic sign'],
    'sign': ['traffic sign', 'stop sign'],
    'snow': ['snow'],
    'parking': ['parking space', 'parking lot'],
}

MODEL_KEYS = ['instruct', 'thinking', 'instruct_30b', 'thinking_30b']
MODEL_LABELS = {
    'instruct': 'Instruct 8B',
    'thinking': 'Thinking 8B',
    'instruct_30b': 'Instruct 30B',
    'thinking_30b': 'Thinking 30B',
}


def match_token_to_detections(token_text, detections):
    """Return list of matched detections for a token (same logic as JS)."""
    if not detections:
        return []
    normalized = token_text.strip().lower()
    if not normalized:
        return []
    synonyms = SYNONYM_MAP.get(normalized, [normalized])
    matched = []
    for d in detections:
        dl = d['label'].lower()
        if any(s in dl or dl in s for s in synonyms):
            matched.append(d)
    return matched


def build_box_mask(matched_detections, grid_h, grid_w):
    """Build binary mask over patch grid from matched detection bboxes."""
    mask = np.zeros(grid_h * grid_w, dtype=np.float64)
    for det in matched_detections:
        x1, y1, x2, y2 = det['bbox_normalized']
        r_start = max(0, int(math.floor(y1 * grid_h)))
        r_end = min(grid_h, int(math.ceil(y2 * grid_h)))
        c_start = max(0, int(math.floor(x1 * grid_w)))
        c_end = min(grid_w, int(math.ceil(x2 * grid_w)))
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                mask[r * grid_w + c] = 1.0
    return mask


def soft_iou(attn, mask):
    """Distribution-normalized Soft IoU: both vectors normalized to sum to 1."""
    attn_sum = np.sum(attn)
    mask_sum = np.sum(mask)
    if attn_sum == 0 or mask_sum == 0:
        return 0.0
    norm_attn = attn / attn_sum
    norm_mask = mask / mask_sum
    intersection = np.sum(np.minimum(norm_attn, norm_mask))
    union = np.sum(np.maximum(norm_attn, norm_mask))
    if union == 0:
        return 0.0
    return intersection / union


def grounding_ratio(patch_attention, mask):
    """Fraction of total image attention that falls inside the box mask."""
    total = np.sum(patch_attention)
    if total == 0:
        return 0.0
    inside = np.sum(patch_attention * mask)
    return inside / total


def img_text_ratio(patch_attention, text_attention):
    """Ratio of image attention to total (image + text)."""
    img_sum = np.sum(patch_attention)
    text_sum = np.sum(text_attention)
    total = img_sum + text_sum
    if total == 0:
        return 0.0
    return img_sum / total


def load_layer_data(base_dir, layer_path):
    """Load a per-layer precomputed JSON file."""
    path = os.path.join(base_dir, layer_path)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Analyze bbox-matched tokens across models and layers")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    manifest_path = os.path.join(base_dir, "data", "manifest.json")

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Load question type classification
    qtypes_path = os.path.join(base_dir, "data", "question_types.json")
    qtype_map = {}
    if os.path.exists(qtypes_path):
        with open(qtypes_path) as f:
            qtypes_data = json.load(f)
        for qt in qtypes_data['questions']:
            qtype_map[qt['question']] = qt['type']

    bbox_dir = os.path.join(args.output_dir, "bbox_token")
    os.makedirs(bbox_dir, exist_ok=True)
    graph_dir = os.path.join(bbox_dir, "bbox_token_graphs")
    os.makedirs(graph_dir, exist_ok=True)

    # Collect all results: list of dicts
    all_results = []
    # For graphs: group by (question_id, matched_label, model) -> list of (token_idx, generation_order, layer, soft_iou, attn_pct, ratio)
    graph_data = defaultdict(list)  # key: (qid, label_group, model) -> [{token_idx, gen_order, layers: {L: (cos, attn, ratio)}}]

    num_questions = len(manifest['questions'])
    for qi, question in enumerate(manifest['questions']):
        qid = question['id']
        detections = question.get('detections', [])
        if not detections:
            continue
        grid_h, grid_w = question['patch_grid']

        for model_key in MODEL_KEYS:
            if model_key not in question['models']:
                continue
            model_info = question['models'][model_key]
            tokens = model_info.get('generated_tokens', [])
            layer_paths = model_info.get('layer_data_paths', {})
            if not tokens or not layer_paths:
                continue

            num_layers = len(layer_paths)
            layer_indices = sorted(layer_paths.keys(), key=int)

            # Find matched tokens
            matched_tokens = []
            for ti, tok in enumerate(tokens):
                matched_dets = match_token_to_detections(tok['text'], detections)
                if matched_dets:
                    matched_tokens.append((ti, tok['text'], tok['phase'], matched_dets))

            if not matched_tokens:
                continue

            print(f"[{qi+1}/{num_questions}] {qid} / {model_key}: {len(matched_tokens)} matched tokens across {num_layers} layers")

            # Load all layer data
            layer_data_cache = {}
            for li in layer_indices:
                ld = load_layer_data(base_dir, layer_paths[li])
                if ld is not None:
                    layer_data_cache[li] = ld

            # For each matched token, compute metrics per layer
            for ti, text, phase, matched_dets in matched_tokens:
                box_mask = build_box_mask(matched_dets, grid_h, grid_w)
                label_group = matched_dets[0]['label'].lower()

                token_result = {
                    'qid': qid,
                    'model': model_key,
                    'token_idx': ti,
                    'token_text': text,
                    'phase': phase,
                    'matched_labels': [d['label'] for d in matched_dets],
                    'is_correct': model_info.get('is_correct'),
                    'detection_count': len(detections),
                    'question_type': qtype_map.get(question['question_text'], 'Unknown'),
                    'num_matched_dets': len(matched_dets),
                    'layers': {},
                }

                for li in layer_indices:
                    if li not in layer_data_cache:
                        continue
                    ld = layer_data_cache[li]
                    # Find this token in the layer data
                    if ti >= len(ld['token_data']):
                        continue
                    td = ld['token_data'][ti]
                    patch_attn = np.array(td['patch_attention'], dtype=np.float64)
                    text_attn = np.array(td.get('text_attention', []), dtype=np.float64)

                    siou = soft_iou(patch_attn, box_mask)
                    attn_pct = grounding_ratio(patch_attn, box_mask)
                    ratio = img_text_ratio(patch_attn, text_attn)

                    token_result['layers'][li] = {
                        'soft_iou': round(siou, 4),
                        'attn_pct': round(attn_pct, 4),
                        'img_text_ratio': round(ratio, 4),
                    }

                all_results.append(token_result)
                graph_data[(qid, label_group, model_key)].append(token_result)

    # Build qid -> question_text lookup
    q_text_map = {q['id']: q['question_text'] for q in manifest['questions']}

    # --- Write markdown ---
    md_path = os.path.join(bbox_dir, "bbox_token_analysis.md")
    with open(md_path, 'w') as md:
        md.write("# Bounding Box Token Analysis\n\n")
        md.write(f"Analyzed {len(all_results)} matched tokens across {len(manifest['questions'])} questions and 4 models.\n\n")

        # Group by question
        by_question = defaultdict(list)
        for r in all_results:
            by_question[r['qid']].append(r)

        for qid in sorted(by_question.keys()):
            q_results = by_question[qid]
            # Find question text
            q_info = next(q for q in manifest['questions'] if q['id'] == qid)
            md.write(f"## {qid}: {q_info['question_text']}\n\n")

            # Group by model
            by_model = defaultdict(list)
            for r in q_results:
                by_model[r['model']].append(r)

            for model_key in MODEL_KEYS:
                if model_key not in by_model:
                    continue
                model_results = by_model[model_key]
                md.write(f"### {MODEL_LABELS[model_key]}\n\n")

                for tr in model_results:
                    md.write(f"**Token #{tr['token_idx']}**: `{tr['token_text']}` ({tr['phase']}) — matched: {', '.join(tr['matched_labels'])}\n\n")
                    md.write("| Layer | Soft IoU | Attn% | Img↔Text Ratio |\n")
                    md.write("|-------|---------|-------|----------------|\n")

                    for li in sorted(tr['layers'].keys(), key=int):
                        m = tr['layers'][li]
                        md.write(f"| L{li} | {m['soft_iou']:.4f} | {m['attn_pct']*100:.1f}% | {m['img_text_ratio']:.4f} |\n")

                    # ALL row: average across layers
                    if tr['layers']:
                        avg_siou = np.mean([m['soft_iou'] for m in tr['layers'].values()])
                        avg_attn = np.mean([m['attn_pct'] for m in tr['layers'].values()])
                        avg_ratio = np.mean([m['img_text_ratio'] for m in tr['layers'].values()])
                        md.write(f"| **ALL** | **{avg_siou:.4f}** | **{avg_attn*100:.1f}%** | **{avg_ratio:.4f}** |\n")
                    md.write("\n")

        # Summary statistics
        md.write("---\n\n## Summary Statistics\n\n")
        md.write("Average metrics across all matched tokens (per model, averaged over layers):\n\n")
        md.write("| Model | Avg Soft IoU | Avg Attn% | Avg Img↔Text Ratio | Token Count |\n")
        md.write("|-------|-------------|-----------|--------------------|-----------|\n")

        for model_key in MODEL_KEYS:
            model_results = [r for r in all_results if r['model'] == model_key]
            if not model_results:
                continue
            all_siou, all_attn, all_ratio = [], [], []
            for tr in model_results:
                for li, m in tr['layers'].items():
                    all_siou.append(m['soft_iou'])
                    all_attn.append(m['attn_pct'])
                    all_ratio.append(m['img_text_ratio'])
            n = len(model_results)
            md.write(f"| {MODEL_LABELS[model_key]} | {np.mean(all_siou):.4f} | {np.mean(all_attn)*100:.1f}% | {np.mean(all_ratio):.4f} | {n} |\n")

        md.write("\n")

    print(f"\nMarkdown written to {md_path}")

    # --- Write CSV ---
    csv_path = os.path.join(bbox_dir, "bbox_token_analysis.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['qid', 'question_text', 'model', 'token_idx', 'token_text',
                          'phase', 'matched_labels', 'is_correct', 'question_type',
                          'detection_count', 'num_matched_dets',
                          'layer', 'soft_iou', 'attn_pct', 'img_text_ratio'])
        for tr in all_results:
            q_text = q_text_map.get(tr['qid'], '')
            labels_str = ';'.join(tr['matched_labels'])
            common = [tr['qid'], q_text, tr['model'], tr['token_idx'],
                      tr['token_text'], tr['phase'], labels_str,
                      tr['is_correct'], tr['question_type'],
                      tr['detection_count'], tr['num_matched_dets']]
            for li in sorted(tr['layers'].keys(), key=int):
                m = tr['layers'][li]
                writer.writerow(common + [li, m['soft_iou'], m['attn_pct'], m['img_text_ratio']])
            # ALL row
            if tr['layers']:
                avg_siou = np.mean([m['soft_iou'] for m in tr['layers'].values()])
                avg_attn = np.mean([m['attn_pct'] for m in tr['layers'].values()])
                avg_ratio = np.mean([m['img_text_ratio'] for m in tr['layers'].values()])
                writer.writerow(common + ['ALL', round(avg_siou, 4), round(avg_attn, 4), round(avg_ratio, 4)])
    print(f"CSV written to {csv_path}")

    # --- Statistical Analysis ---
    stats_path = os.path.join(bbox_dir, "bbox_token_stats.md")
    with open(stats_path, 'w') as sf:
        sf.write("# Statistical Analysis — Bounding Box Token Metrics\n\n")

        # Build per-token ALL averages for each model
        model_avgs = defaultdict(lambda: {'soft_iou': [], 'attn_pct': [], 'img_text_ratio': []})
        phase_model_avgs = defaultdict(lambda: {'soft_iou': [], 'attn_pct': [], 'img_text_ratio': []})

        for tr in all_results:
            if not tr['layers']:
                continue
            avg_siou = np.mean([m['soft_iou'] for m in tr['layers'].values()])
            avg_attn = np.mean([m['attn_pct'] for m in tr['layers'].values()])
            avg_ratio = np.mean([m['img_text_ratio'] for m in tr['layers'].values()])
            model_avgs[tr['model']]['soft_iou'].append(avg_siou)
            model_avgs[tr['model']]['attn_pct'].append(avg_attn)
            model_avgs[tr['model']]['img_text_ratio'].append(avg_ratio)
            phase_model_avgs[(tr['phase'], tr['model'])]['soft_iou'].append(avg_siou)
            phase_model_avgs[(tr['phase'], tr['model'])]['attn_pct'].append(avg_attn)
            phase_model_avgs[(tr['phase'], tr['model'])]['img_text_ratio'].append(avg_ratio)

        # 1. Descriptive stats per model (models as columns, metrics as rows)
        sf.write("## 1. Descriptive Statistics (layer-averaged per token)\n\n")
        sf.write("Each token's metric is first averaged across all layers, then aggregated per model.\n\n")
        active_models = [mk for mk in MODEL_KEYS if model_avgs[mk]['soft_iou']]
        header = "| Metric | Stat | " + " | ".join(MODEL_LABELS[mk] for mk in active_models) + " |\n"
        sep = "|--------|------| " + " | ".join("---" for _ in active_models) + " |\n"
        sf.write(header)
        sf.write(sep)
        for metric in ['soft_iou', 'attn_pct', 'img_text_ratio']:
            label = {'soft_iou': 'Soft IoU', 'attn_pct': 'Attn%', 'img_text_ratio': 'Img↔Text'}[metric]
            for stat_name, stat_fn in [('Mean', np.mean), ('Std', np.std)]:
                vals_per_model = {mk: stat_fn(np.array(model_avgs[mk][metric])) for mk in active_models}
                best = max(vals_per_model.values()) if stat_name == 'Mean' else None
                cells = []
                for mk in active_models:
                    v = vals_per_model[mk]
                    cell = f"{v:.4f}"
                    if best is not None and v == best:
                        cell = f"**{cell}**"
                    cells.append(cell)
                sf.write(f"| {label} | {stat_name} | " + " | ".join(cells) + " |\n")
        # N row
        n_cells = [str(len(model_avgs[mk]['soft_iou'])) for mk in active_models]
        sf.write(f"| | N | " + " | ".join(n_cells) + " |\n")
        sf.write("\n")

        # 2. 8B vs 30B comparisons (Mann-Whitney U)
        sf.write("## 2. 8B vs 30B Comparison (Mann-Whitney U test)\n\n")
        sf.write("Mann-Whitney U is a non-parametric rank-sum test comparing two independent groups.\n")
        sf.write("Chosen because metric distributions are non-normal (bounded, skewed) and sample sizes differ across models.\n")
        sf.write("Two-sided alternative: tests whether distributions differ in location, without assuming direction.\n\n")
        sf.write("| Comparison | Metric | U-stat | p-value | 8B Mean | 30B Mean | Effect |\n")
        sf.write("|-----------|--------|--------|---------|---------|----------|--------|\n")
        for pair_name, mk_8b, mk_30b in [('Instruct', 'instruct', 'instruct_30b'),
                                            ('Thinking', 'thinking', 'thinking_30b')]:
            for metric in ['soft_iou', 'attn_pct', 'img_text_ratio']:
                v8 = np.array(model_avgs[mk_8b][metric])
                v30 = np.array(model_avgs[mk_30b][metric])
                if len(v8) < 2 or len(v30) < 2:
                    continue
                u_stat, p_val = stats.mannwhitneyu(v8, v30, alternative='two-sided')
                label = {'soft_iou': 'Soft IoU', 'attn_pct': 'Attn%', 'img_text_ratio': 'Img↔Text'}[metric]
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                direction = '30B > 8B' if np.mean(v30) > np.mean(v8) else '8B > 30B'
                sf.write(f"| {pair_name} | {label} | {u_stat:.0f} | {p_val:.4e} | {np.mean(v8):.4f} | {np.mean(v30):.4f} | {direction} {sig} |\n")
        sf.write("\n")

        # 3. Instruct vs Thinking comparisons
        sf.write("## 3. Instruct vs Thinking Comparison (Mann-Whitney U test)\n\n")
        sf.write("Same non-parametric test as §2, comparing Instruct (direct answer) vs Thinking (chain-of-thought) within each model size.\n\n")
        sf.write("| Comparison | Metric | U-stat | p-value | Instruct Mean | Thinking Mean | Effect |\n")
        sf.write("|-----------|--------|--------|---------|--------------|--------------|--------|\n")
        for pair_name, mk_i, mk_t in [('8B', 'instruct', 'thinking'),
                                        ('30B', 'instruct_30b', 'thinking_30b')]:
            for metric in ['soft_iou', 'attn_pct', 'img_text_ratio']:
                vi = np.array(model_avgs[mk_i][metric])
                vt = np.array(model_avgs[mk_t][metric])
                if len(vi) < 2 or len(vt) < 2:
                    continue
                u_stat, p_val = stats.mannwhitneyu(vi, vt, alternative='two-sided')
                label = {'soft_iou': 'Soft IoU', 'attn_pct': 'Attn%', 'img_text_ratio': 'Img↔Text'}[metric]
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                direction = 'Think > Inst' if np.mean(vt) > np.mean(vi) else 'Inst > Think'
                sf.write(f"| {pair_name} | {label} | {u_stat:.0f} | {p_val:.4e} | {np.mean(vi):.4f} | {np.mean(vt):.4f} | {direction} {sig} |\n")
        sf.write("\n")

        # 4. Thinking vs Answer phase comparison (within thinking models)
        sf.write("## 4. Thinking vs Answer Phase (within Thinking models)\n\n")
        sf.write("Mann-Whitney U comparing tokens generated during `<think>` reasoning vs tokens in the final answer.\n")
        sf.write("Tests whether spatial grounding differs between reasoning and answering phases.\n\n")
        sf.write("| Model | Metric | U-stat | p-value | Thinking Mean | Answer Mean | Effect |\n")
        sf.write("|-------|--------|--------|---------|--------------|------------|--------|\n")
        for mk in ['thinking', 'thinking_30b']:
            for metric in ['soft_iou', 'attn_pct', 'img_text_ratio']:
                vt = np.array(phase_model_avgs[('thinking', mk)][metric])
                va = np.array(phase_model_avgs[('answer', mk)][metric])
                if len(vt) < 2 or len(va) < 2:
                    continue
                u_stat, p_val = stats.mannwhitneyu(vt, va, alternative='two-sided')
                label = {'soft_iou': 'Soft IoU', 'attn_pct': 'Attn%', 'img_text_ratio': 'Img↔Text'}[metric]
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                direction = 'Answer > Think' if np.mean(va) > np.mean(vt) else 'Think > Answer'
                sf.write(f"| {MODEL_LABELS[mk]} | {label} | {u_stat:.0f} | {p_val:.4e} | {np.mean(vt):.4f} | {np.mean(va):.4f} | {direction} {sig} |\n")
        sf.write("\n")

        # 5. Layer-wise trends: Spearman correlation of layer index vs metric
        sf.write("## 5. Layer-wise Trends (Spearman correlation: layer index vs metric)\n\n")
        sf.write("Spearman rank correlation measures monotonic association between layer depth and metric value.\n")
        sf.write("Non-parametric: does not assume linear relationship, only consistent direction. Each (token, layer) pair is one observation.\n")
        sf.write("Positive rho = metric increases with depth; negative = decreases.\n\n")
        sf.write("| Model | Metric | Spearman rho | p-value | Trend |\n")
        sf.write("|-------|--------|-------------|---------|-------|\n")
        for mk in MODEL_KEYS:
            mk_results = [r for r in all_results if r['model'] == mk]
            if not mk_results:
                continue
            # Collect (layer, metric_value) across all tokens
            for metric in ['soft_iou', 'attn_pct', 'img_text_ratio']:
                layers_flat, vals_flat = [], []
                for tr in mk_results:
                    for li, m in tr['layers'].items():
                        layers_flat.append(int(li))
                        vals_flat.append(m[metric])
                if len(layers_flat) < 10:
                    continue
                rho, p_val = stats.spearmanr(layers_flat, vals_flat)
                label = {'soft_iou': 'Soft IoU', 'attn_pct': 'Attn%', 'img_text_ratio': 'Img↔Text'}[metric]
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                trend = '↑ deeper' if rho > 0 else '↓ deeper'
                sf.write(f"| {MODEL_LABELS[mk]} | {label} | {rho:.4f} | {p_val:.4e} | {trend} {sig} |\n")
        sf.write("\n")

        # 6. Peak layer analysis
        sf.write("## 6. Peak Layer Analysis\n\n")
        sf.write("Layer with highest average metric across all matched tokens:\n\n")
        sf.write("| Model | Metric | Peak Layer | Peak Value |\n")
        sf.write("|-------|--------|-----------|------------|\n")
        for mk in MODEL_KEYS:
            mk_results = [r for r in all_results if r['model'] == mk]
            if not mk_results:
                continue
            for metric in ['soft_iou', 'attn_pct', 'img_text_ratio']:
                layer_vals = defaultdict(list)
                for tr in mk_results:
                    for li, m in tr['layers'].items():
                        layer_vals[int(li)].append(m[metric])
                if not layer_vals:
                    continue
                layer_means = {l: np.mean(v) for l, v in layer_vals.items()}
                peak_layer = max(layer_means, key=layer_means.get)
                label = {'soft_iou': 'Soft IoU', 'attn_pct': 'Attn%', 'img_text_ratio': 'Img↔Text'}[metric]
                sf.write(f"| {MODEL_LABELS[mk]} | {label} | L{peak_layer} | {layer_means[peak_layer]:.4f} |\n")
        sf.write("\n")

        # --- Helper to compute layer-averaged metrics for a list of token results ---
        def _avg_metrics(results_list):
            """Return (soft_iou_list, attn_pct_list, img_text_ratio_list) of layer-averaged values."""
            siou, attn, ratio = [], [], []
            for tr in results_list:
                if not tr['layers']:
                    continue
                siou.append(np.mean([m['soft_iou'] for m in tr['layers'].values()]))
                attn.append(np.mean([m['attn_pct'] for m in tr['layers'].values()]))
                ratio.append(np.mean([m['img_text_ratio'] for m in tr['layers'].values()]))
            return siou, attn, ratio

        METRIC_LABEL = {'soft_iou': 'Soft IoU', 'attn_pct': 'Attn%', 'img_text_ratio': 'Img↔Text'}

        def _sig(p):
            return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

        # 7. Correct vs Incorrect per model
        sf.write("## 7. Correct vs Incorrect (per model)\n\n")
        sf.write("Mann-Whitney U comparing tokens from questions the model answered correctly vs incorrectly.\n")
        sf.write("Tests whether spatial grounding quality correlates with answer correctness.\n\n")
        sf.write("| Model | Metric | Correct Mean | Incorrect Mean | U-stat | p-value | Effect |\n")
        sf.write("|-------|--------|-------------|---------------|--------|---------|--------|\n")
        for mk in MODEL_KEYS:
            correct_trs = [tr for tr in all_results if tr['model'] == mk and tr['is_correct'] == True]
            incorrect_trs = [tr for tr in all_results if tr['model'] == mk and tr['is_correct'] == False]
            c_siou, c_attn, c_ratio = _avg_metrics(correct_trs)
            i_siou, i_attn, i_ratio = _avg_metrics(incorrect_trs)
            for metric, vc, vi in [('soft_iou', c_siou, i_siou), ('attn_pct', c_attn, i_attn), ('img_text_ratio', c_ratio, i_ratio)]:
                vc_a, vi_a = np.array(vc), np.array(vi)
                if len(vc_a) < 2 or len(vi_a) < 2:
                    continue
                u_stat, p_val = stats.mannwhitneyu(vc_a, vi_a, alternative='two-sided')
                direction = 'Correct > Incorrect' if np.mean(vc_a) > np.mean(vi_a) else 'Incorrect > Correct'
                sf.write(f"| {MODEL_LABELS[mk]} | {METRIC_LABEL[metric]} | {np.mean(vc_a):.4f} | {np.mean(vi_a):.4f} | {u_stat:.0f} | {p_val:.4e} | {direction} {_sig(p_val)} |\n")
        sf.write("\n")

        # 8. By Question Type
        sf.write("## 8. By Question Type\n\n")
        sf.write("Questions classified as Perception (identifying objects/environment) or Planning (ego vehicle action decisions).\n")
        sf.write("Mann-Whitney U tests whether question type affects spatial grounding metrics.\n\n")
        type_model_vals = defaultdict(lambda: {'soft_iou': [], 'attn_pct': [], 'img_text_ratio': []})
        for tr in all_results:
            if not tr['layers']:
                continue
            key = (tr['question_type'], tr['model'])
            s, a, r = _avg_metrics([tr])
            if s:
                type_model_vals[key]['soft_iou'].extend(s)
                type_model_vals[key]['attn_pct'].extend(a)
                type_model_vals[key]['img_text_ratio'].extend(r)

        active_types = sorted(set(k[0] for k in type_model_vals.keys() if type_model_vals[k]['soft_iou']))

        sf.write("### Descriptive Statistics\n\n")
        sf.write("| Type | Model | N | Soft IoU | Attn% | Img↔Text |\n")
        sf.write("|------|-------|---|---------|-------|----------|\n")
        for qtype in active_types:
            for mk in MODEL_KEYS:
                vals = type_model_vals.get((qtype, mk))
                if not vals or not vals['soft_iou']:
                    continue
                n = len(vals['soft_iou'])
                sf.write(f"| {qtype} | {MODEL_LABELS[mk]} | {n} | {np.mean(vals['soft_iou']):.4f} | {np.mean(vals['attn_pct']):.4f} | {np.mean(vals['img_text_ratio']):.4f} |\n")
        sf.write("\n")

        # Perception vs Planning comparison if both exist
        if 'Perception' in active_types and 'Planning' in active_types:
            sf.write("### Perception vs Planning (Mann-Whitney U)\n\n")
            sf.write("| Model | Metric | Perc Mean | Plan Mean | U-stat | p-value | Effect |\n")
            sf.write("|-------|--------|-----------|-----------|--------|---------|--------|\n")
            for mk in MODEL_KEYS:
                perc = type_model_vals.get(('Perception', mk))
                plan = type_model_vals.get(('Planning', mk))
                if not perc or not plan:
                    continue
                for metric in ['soft_iou', 'attn_pct', 'img_text_ratio']:
                    vp = np.array(perc[metric])
                    vpl = np.array(plan[metric])
                    if len(vp) < 2 or len(vpl) < 2:
                        continue
                    u_stat, p_val = stats.mannwhitneyu(vp, vpl, alternative='two-sided')
                    direction = 'Perc > Plan' if np.mean(vp) > np.mean(vpl) else 'Plan > Perc'
                    sf.write(f"| {MODEL_LABELS[mk]} | {METRIC_LABEL[metric]} | {np.mean(vp):.4f} | {np.mean(vpl):.4f} | {u_stat:.0f} | {p_val:.4e} | {direction} {_sig(p_val)} |\n")
            sf.write("\n")

        # 9. Number of Objects — Binned
        sf.write("## 9. By Number of Detected Objects (Binned)\n\n")
        sf.write("Questions grouped by total YOLO detection count. Shows whether scene complexity affects grounding.\n\n")
        det_bins = [(1, 5, '1\u20135'), (6, 10, '6\u201310'), (11, 999, '11+')]
        bin_model_vals = defaultdict(lambda: {'soft_iou': [], 'attn_pct': [], 'img_text_ratio': []})
        for tr in all_results:
            if not tr['layers']:
                continue
            dc = tr['detection_count']
            for lo, hi, bin_label in det_bins:
                if lo <= dc <= hi:
                    s, a, r = _avg_metrics([tr])
                    if s:
                        bin_model_vals[(bin_label, tr['model'])]['soft_iou'].extend(s)
                        bin_model_vals[(bin_label, tr['model'])]['attn_pct'].extend(a)
                        bin_model_vals[(bin_label, tr['model'])]['img_text_ratio'].extend(r)
                    break

        sf.write("| Bin | Model | N | Soft IoU | Attn% | Img↔Text |\n")
        sf.write("|-----|-------|---|---------|-------|----------|\n")
        for _, _, bin_label in det_bins:
            for mk in MODEL_KEYS:
                vals = bin_model_vals.get((bin_label, mk))
                if not vals or not vals['soft_iou']:
                    continue
                n = len(vals['soft_iou'])
                sf.write(f"| {bin_label} | {MODEL_LABELS[mk]} | {n} | {np.mean(vals['soft_iou']):.4f} | {np.mean(vals['attn_pct']):.4f} | {np.mean(vals['img_text_ratio']):.4f} |\n")
        sf.write("\n")

        # 10. Detection Count vs Metrics (Spearman correlation)
        sf.write("## 10. Detection Count vs Metrics (Spearman)\n\n")
        sf.write("Spearman rank correlation between number of detected objects per question and token-level metrics.\n")
        sf.write("Treats detection count as a continuous variable rather than binning. Each matched token is one observation.\n\n")
        sf.write("| Model | Metric | Spearman rho | p-value | Trend |\n")
        sf.write("|-------|--------|-------------|---------|-------|\n")
        for mk in MODEL_KEYS:
            mk_results = [r for r in all_results if r['model'] == mk and r['layers']]
            if not mk_results:
                continue
            for metric in ['soft_iou', 'attn_pct', 'img_text_ratio']:
                det_counts_list, metric_vals = [], []
                for tr in mk_results:
                    avg_val = np.mean([m[metric] for m in tr['layers'].values()])
                    det_counts_list.append(tr['detection_count'])
                    metric_vals.append(avg_val)
                if len(det_counts_list) < 10:
                    continue
                rho, p_val = stats.spearmanr(det_counts_list, metric_vals)
                trend = '\u2191 more objects' if rho > 0 else '\u2193 more objects'
                sf.write(f"| {MODEL_LABELS[mk]} | {METRIC_LABEL[metric]} | {rho:.4f} | {p_val:.4e} | {trend} {_sig(p_val)} |\n")
        sf.write("\n")

        # 11. Single vs Multi-Detection Token Matches
        sf.write("## 11. Single vs Multi-Detection Token Matches\n\n")
        sf.write("Compares tokens that matched exactly 1 detected object vs tokens matching 2+ objects.\n")
        sf.write("Multi-detection tokens (e.g., 'vehicles' matching 3 cars) have larger combined bounding box masks.\n\n")
        sf.write("### Descriptive Statistics\n\n")
        sf.write("| Model | Match | N | Soft IoU | Attn% | Img↔Text |\n")
        sf.write("|-------|-------|---|---------|-------|----------|\n")
        for mk in MODEL_KEYS:
            for match_label, lo, hi in [('Single (1)', 1, 1), ('Multi (2+)', 2, 999)]:
                matched = [tr for tr in all_results if tr['model'] == mk and lo <= tr['num_matched_dets'] <= hi and tr['layers']]
                if not matched:
                    continue
                s, a, r = _avg_metrics(matched)
                sf.write(f"| {MODEL_LABELS[mk]} | {match_label} | {len(matched)} | {np.mean(s):.4f} | {np.mean(a):.4f} | {np.mean(r):.4f} |\n")
        sf.write("\n")

        sf.write("### Single vs Multi-Detection (Mann-Whitney U)\n\n")
        sf.write("| Model | Metric | Single Mean | Multi Mean | U-stat | p-value | Effect |\n")
        sf.write("|-------|--------|------------|-----------|--------|---------|--------|\n")
        for mk in MODEL_KEYS:
            single = [tr for tr in all_results if tr['model'] == mk and tr['num_matched_dets'] == 1 and tr['layers']]
            multi = [tr for tr in all_results if tr['model'] == mk and tr['num_matched_dets'] >= 2 and tr['layers']]
            s_siou, s_attn, s_ratio = _avg_metrics(single)
            m_siou, m_attn, m_ratio = _avg_metrics(multi)
            for metric, vs, vm in [('soft_iou', s_siou, m_siou), ('attn_pct', s_attn, m_attn), ('img_text_ratio', s_ratio, m_ratio)]:
                vs_a, vm_a = np.array(vs), np.array(vm)
                if len(vs_a) < 2 or len(vm_a) < 2:
                    continue
                u_stat, p_val = stats.mannwhitneyu(vs_a, vm_a, alternative='two-sided')
                direction = 'Multi > Single' if np.mean(vm_a) > np.mean(vs_a) else 'Single > Multi'
                sf.write(f"| {MODEL_LABELS[mk]} | {METRIC_LABEL[metric]} | {np.mean(vs_a):.4f} | {np.mean(vm_a):.4f} | {u_stat:.0f} | {p_val:.4e} | {direction} {_sig(p_val)} |\n")
        sf.write("\n")

        # 11b. Cross-model comparison within single- and multi-detection subsets
        sf.write("### Cross-Model Comparison within Detection Subsets (Mann-Whitney U)\n\n")
        sf.write("Tests whether models differ when controlling for detection count (single or multi).\n\n")
        for match_label, lo, hi in [('Single (1)', 1, 1), ('Multi (2+)', 2, 999)]:
            sf.write(f"#### {match_label} Detection Tokens\n\n")
            # Instruct vs Thinking within each size
            sf.write("**Instruct vs Thinking**\n\n")
            sf.write("| Size | Metric | Instruct Mean | Thinking Mean | U-stat | p-value | Effect |\n")
            sf.write("|------|--------|--------------|--------------|--------|---------|--------|\n")
            for pair_name, mk_i, mk_t in [('8B', 'instruct', 'thinking'), ('30B', 'instruct_30b', 'thinking_30b')]:
                trs_i = [tr for tr in all_results if tr['model'] == mk_i and lo <= tr['num_matched_dets'] <= hi and tr['layers']]
                trs_t = [tr for tr in all_results if tr['model'] == mk_t and lo <= tr['num_matched_dets'] <= hi and tr['layers']]
                si_i, ai_i, ri_i = _avg_metrics(trs_i)
                si_t, ai_t, ri_t = _avg_metrics(trs_t)
                for metric, vi, vt in [('soft_iou', si_i, si_t), ('attn_pct', ai_i, ai_t), ('img_text_ratio', ri_i, ri_t)]:
                    vi_a, vt_a = np.array(vi), np.array(vt)
                    if len(vi_a) < 2 or len(vt_a) < 2:
                        continue
                    u_stat, p_val = stats.mannwhitneyu(vi_a, vt_a, alternative='two-sided')
                    direction = 'Think > Inst' if np.mean(vt_a) > np.mean(vi_a) else 'Inst > Think'
                    sf.write(f"| {pair_name} | {METRIC_LABEL[metric]} | {np.mean(vi_a):.4f} | {np.mean(vt_a):.4f} | {u_stat:.0f} | {p_val:.4e} | {direction} {_sig(p_val)} |\n")
            sf.write("\n")

            # 8B vs 30B within each mode
            sf.write("**8B vs 30B**\n\n")
            sf.write("| Mode | Metric | 8B Mean | 30B Mean | U-stat | p-value | Effect |\n")
            sf.write("|------|--------|---------|----------|--------|---------|--------|\n")
            for pair_name, mk_8b, mk_30b in [('Instruct', 'instruct', 'instruct_30b'), ('Thinking', 'thinking', 'thinking_30b')]:
                trs_8 = [tr for tr in all_results if tr['model'] == mk_8b and lo <= tr['num_matched_dets'] <= hi and tr['layers']]
                trs_30 = [tr for tr in all_results if tr['model'] == mk_30b and lo <= tr['num_matched_dets'] <= hi and tr['layers']]
                si_8, ai_8, ri_8 = _avg_metrics(trs_8)
                si_30, ai_30, ri_30 = _avg_metrics(trs_30)
                for metric, v8, v30 in [('soft_iou', si_8, si_30), ('attn_pct', ai_8, ai_30), ('img_text_ratio', ri_8, ri_30)]:
                    v8_a, v30_a = np.array(v8), np.array(v30)
                    if len(v8_a) < 2 or len(v30_a) < 2:
                        continue
                    u_stat, p_val = stats.mannwhitneyu(v8_a, v30_a, alternative='two-sided')
                    direction = '30B > 8B' if np.mean(v30_a) > np.mean(v8_a) else '8B > 30B'
                    sf.write(f"| {pair_name} | {METRIC_LABEL[metric]} | {np.mean(v8_a):.4f} | {np.mean(v30_a):.4f} | {u_stat:.0f} | {p_val:.4e} | {direction} {_sig(p_val)} |\n")
            sf.write("\n")

        # 12. Per-Question Summary Table
        sf.write("---\n\n## 12. Per-Question Summary\n\n")
        sf.write("Layer-averaged mean metrics per question per model. Correctness column: \u2713 = correct, \u2717 = incorrect.\n\n")

        # Build per-(qid, model) averages
        qid_model_metrics = defaultdict(lambda: {'soft_iou': [], 'attn_pct': [], 'img_text_ratio': []})
        for tr in all_results:
            if not tr['layers']:
                continue
            key = (tr['qid'], tr['model'])
            s, a, r = _avg_metrics([tr])
            if s:
                qid_model_metrics[key]['soft_iou'].extend(s)
                qid_model_metrics[key]['attn_pct'].extend(a)
                qid_model_metrics[key]['img_text_ratio'].extend(r)

        # Header
        sf.write("| Q | Question | Type | # Det | ")
        for metric_short in ['Soft IoU', 'Attn%', 'Img\u2194Text']:
            for mk in MODEL_KEYS:
                sf.write(f"{metric_short} {MODEL_LABELS[mk]} | ")
        sf.write("\n")
        sf.write("|---|----------|------|-------|")
        for _ in range(3 * len(MODEL_KEYS)):
            sf.write("---|")
        sf.write("\n")

        for question in manifest['questions']:
            qid = question['id']
            q_text = question['question_text']
            short_q = q_text if len(q_text) <= 35 else q_text[:32] + '...'
            q_type = qtype_map.get(q_text, '?')
            det_count = len(question.get('detections', []))

            # Correctness string per model
            correct_strs = {}
            for mk in MODEL_KEYS:
                if mk in question['models']:
                    ic = question['models'][mk].get('is_correct')
                    correct_strs[mk] = '\u2713' if ic else '\u2717'
                else:
                    correct_strs[mk] = '-'

            # Build correctness label: show per model
            correct_label = '/'.join(correct_strs[mk] for mk in MODEL_KEYS)

            row = f"| {qid} | {short_q} | {q_type} | {det_count} | "
            for metric in ['soft_iou', 'attn_pct', 'img_text_ratio']:
                for mk in MODEL_KEYS:
                    vals = qid_model_metrics.get((qid, mk))
                    if vals and vals[metric]:
                        v = np.mean(vals[metric])
                        if metric == 'attn_pct':
                            row += f"{v*100:.1f}% {correct_strs[mk]} | "
                        else:
                            row += f"{v:.4f} {correct_strs[mk]} | "
                    else:
                        row += "- | "
            sf.write(row + "\n")

        sf.write(f"\nCorrectness key: order is {' / '.join(MODEL_LABELS[mk] for mk in MODEL_KEYS)}\n")
        sf.write("\n")

    print(f"Stats written to {stats_path}")

    # --- Generate per-question generation trend graphs ---
    # Group by label_group -> model -> qid for per-object-type graphs
    by_label = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (qid, label, model_key), tokens in graph_data.items():
        by_label[label][model_key][qid].extend(tokens)

    metric_names = ['soft_iou', 'attn_pct', 'img_text_ratio']
    metric_labels = ['Soft IoU', 'Attn% in Box', 'Image↔Text Ratio']
    model_colors = {'instruct': '#2196F3', 'thinking': '#FF9800', 'instruct_30b': '#4CAF50', 'thinking_30b': '#E91E63'}
    question_markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '<', '>', 'p']

    for label_group, model_data in sorted(by_label.items()):
        # Collect all question ids across all models for this label
        all_qids = sorted(set(qid for mdata in model_data.values() for qid in mdata.keys()))
        qid_marker = {qid: question_markers[i % len(question_markers)] for i, qid in enumerate(all_qids)}

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Matched: "{label_group}" — Generation Order Trends', fontsize=13)

        for mi, (metric_key, metric_label) in enumerate(zip(metric_names, metric_labels)):
            ax = axes[mi]
            for model_key in MODEL_KEYS:
                if model_key not in model_data:
                    continue
                for qid in all_qids:
                    tokens_list = model_data[model_key].get(qid, [])
                    if not tokens_list:
                        continue

                    points = []
                    for tr in tokens_list:
                        if not tr['layers']:
                            continue
                        vals = [tr['layers'][li][metric_key] for li in tr['layers']]
                        avg_val = np.mean(vals)
                        points.append((tr['token_idx'], avg_val))

                    points.sort(key=lambda x: x[0])
                    if not points:
                        continue

                    x = list(range(len(points)))
                    y = [p[1] for p in points]
                    if metric_key == 'attn_pct':
                        y = [v * 100 for v in y]

                    q_short = qid.upper()
                    ax.plot(x, y, marker=qid_marker[qid], color=model_colors[model_key],
                            label=f'{MODEL_LABELS[model_key]} {q_short}',
                            markersize=5, alpha=0.7, linewidth=1.5)

            ax.set_title(metric_label, fontsize=12)
            ax.set_xlabel('Token Order (within matched tokens)')
            ax.set_ylabel(metric_label + (' (%)' if metric_key == 'attn_pct' else ''))
            if mi == 0:
                ax.legend(fontsize=6, loc='best', ncol=2)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        safe_label = label_group.replace(' ', '_')
        fig_path = os.path.join(graph_dir, f"gen_trend_{safe_label}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Graph saved: {fig_path}")

        # Also generate per-question graphs for this object type
        for qid in all_qids:
            q_num = qid.replace('q', '').lstrip('0') or '0'
            q_text = q_text_map.get(qid, '')
            short_q = q_text if len(q_text) <= 50 else q_text[:47] + '...'

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f'{qid} — "{short_q}" — Matched: "{label_group}"', fontsize=13)

            for mi, (metric_key, metric_label) in enumerate(zip(metric_names, metric_labels)):
                ax = axes[mi]
                for model_key in MODEL_KEYS:
                    if model_key not in model_data:
                        continue
                    tokens_list = model_data[model_key].get(qid, [])
                    if not tokens_list:
                        continue

                    points = []
                    for tr in tokens_list:
                        if not tr['layers']:
                            continue
                        vals = [tr['layers'][li][metric_key] for li in tr['layers']]
                        avg_val = np.mean(vals)
                        points.append((tr['token_idx'], avg_val))

                    points.sort(key=lambda x: x[0])
                    if not points:
                        continue

                    x = list(range(len(points)))
                    y = [p[1] for p in points]
                    if metric_key == 'attn_pct':
                        y = [v * 100 for v in y]

                    ax.plot(x, y, 'o-', color=model_colors[model_key],
                            label=MODEL_LABELS[model_key], markersize=4, alpha=0.8, linewidth=1.5)

                ax.set_title(metric_label, fontsize=12)
                ax.set_xlabel('Token Order (within matched tokens)')
                ax.set_ylabel(metric_label + (' (%)' if metric_key == 'attn_pct' else ''))
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            fig_path = os.path.join(graph_dir, f"gen_trend_{safe_label}_q{q_num}.png")
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Graph saved: {fig_path}")

    # --- Per-layer breakdown graphs (averaged across all tokens per model) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Per-Layer Metrics (Averaged Across All Matched Tokens)', fontsize=14)

    for mi, (metric_key, metric_label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[mi]
        for model_key in MODEL_KEYS:
            model_results = [r for r in all_results if r['model'] == model_key]
            if not model_results:
                continue

            # Collect per-layer averages
            layer_vals = defaultdict(list)
            for tr in model_results:
                for li, m in tr['layers'].items():
                    layer_vals[int(li)].append(m[metric_key])

            if not layer_vals:
                continue

            layers_sorted = sorted(layer_vals.keys())
            means = [np.mean(layer_vals[l]) for l in layers_sorted]

            if metric_key == 'attn_pct':
                means = [v * 100 for v in means]

            ax.plot(layers_sorted, means, '-', color=model_colors[model_key],
                    label=MODEL_LABELS[model_key], linewidth=2, alpha=0.8)

        ax.set_title(metric_label, fontsize=12)
        ax.set_xlabel('Layer')
        ax.set_ylabel(metric_label + (' (%)' if metric_key == 'attn_pct' else ''))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(graph_dir, "per_layer_averages.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Graph saved: {fig_path}")

    # --- Per-layer breakdown by phase (thinking vs answer) ---
    for phase in ['thinking', 'answer']:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Per-Layer Metrics — {phase.title()} Tokens Only', fontsize=14)

        for mi, (metric_key, metric_label) in enumerate(zip(metric_names, metric_labels)):
            ax = axes[mi]
            for model_key in MODEL_KEYS:
                model_results = [r for r in all_results if r['model'] == model_key and r['phase'] == phase]
                if not model_results:
                    continue

                layer_vals = defaultdict(list)
                for tr in model_results:
                    for li, m in tr['layers'].items():
                        layer_vals[int(li)].append(m[metric_key])

                if not layer_vals:
                    continue

                layers_sorted = sorted(layer_vals.keys())
                means = [np.mean(layer_vals[l]) for l in layers_sorted]
                if metric_key == 'attn_pct':
                    means = [v * 100 for v in means]

                ax.plot(layers_sorted, means, '-', color=model_colors[model_key],
                        label=MODEL_LABELS[model_key], linewidth=2, alpha=0.8)

            ax.set_title(metric_label, fontsize=12)
            ax.set_xlabel('Layer')
            ax.set_ylabel(metric_label + (' (%)' if metric_key == 'attn_pct' else ''))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(graph_dir, f"per_layer_{phase}_only.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Graph saved: {fig_path}")

    print(f"\nDone. {len(all_results)} matched tokens analyzed.")


if __name__ == "__main__":
    main()
