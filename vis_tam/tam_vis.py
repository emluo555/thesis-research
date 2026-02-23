"""
tam_vis.py - Visualization functions for TAM (Token Activation Map) analysis.

Mirrors attention_vis.py in structure and plot types, adapted for TAM data
(hidden-state activations instead of attention weights).

Provides:
    Mode A (Deep Single-Token):  Detailed activation for a chosen generated token.
    Mode B (Generation Summary): Overview across all generated tokens.
    Structural plots:            Per-layer metrics, activation matrix.
    Comparative plots:           Side-by-side comparison of two models.
    HTML report:                 Index page embedding all plots.
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import PowerNorm
from typing import Optional, List, Dict, Tuple, Any

from tam_analyzer import TokenRegions

# ------------------------------------------------------------------
# Color and style constants (matching attention_vis.py)
# ------------------------------------------------------------------

REGION_COLORS = {
    "system": "#888888",
    "image": "#4A90D9",
    "prompt": "#50B848",
    "thinking": "#E74C3C",
    "answer": "#9B59B6",
    "generated": "#F39C12",
}

REGION_LABELS = {
    "system": "System",
    "image": "Image",
    "prompt": "Prompt",
    "thinking": "Thinking",
    "answer": "Answer",
    "generated": "Generated",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
})


def _ensure_dir(path: str):
    if path:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)


def _save_or_show(fig, save_path: Optional[str]):
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ======================================================================
# MODE A: Deep Single-Token Inspection
# ======================================================================

def plot_activation_bar(
    activation_row: np.ndarray,
    token_strings: List[str],
    token_regions: TokenRegions,
    token_idx: int,
    title: str = "",
    save_path: Optional[str] = None,
    max_text_tokens: int = 80,
):
    """Bar chart of TAM activation scores for token i over all prior positions.

    Image tokens are aggregated into a single bar for readability.
    Bars are colored by region (system, image, prompt, thinking, answer, generated).

    Args:
        activation_row: (score_len,) normalized TAM scores (layer-averaged).
        token_strings: Full list of decoded token strings.
        token_regions: TokenRegions.
        token_idx: Absolute index of the token being inspected.
        title: Additional title text.
        save_path: Path to save the figure.
        max_text_tokens: Maximum individual text token bars to show.
    """
    r = token_regions
    score_len = len(activation_row)

    # Aggregate image tokens into one bar
    img_s = r.image_start
    img_e = min(r.image_end, score_len)
    img_agg = activation_row[img_s:img_e].sum() if img_e > img_s else 0.0

    bar_labels = []
    bar_values = []
    bar_colors = []

    # System tokens before image
    for j in range(0, min(r.image_start, score_len)):
        bar_labels.append(f"[{j}] {_short(token_strings[j])}")
        bar_values.append(float(activation_row[j]))
        bar_colors.append(REGION_COLORS["system"])

    # Aggregated image
    if img_e > img_s:
        bar_labels.append(f"[Image] ({img_e - img_s} tokens)")
        bar_values.append(float(img_agg))
        bar_colors.append(REGION_COLORS["image"])

    # Text tokens (prompt + generated up to score_len)
    text_start = min(r.prompt_start, score_len)
    text_end = score_len - 1
    text_indices = list(range(text_start, text_end + 1))

    # Skip image region indices
    text_indices = [j for j in text_indices if j < r.image_start or j >= r.image_end]

    if len(text_indices) > max_text_tokens:
        step = len(text_indices) // max_text_tokens
        text_indices = text_indices[::step]

    for j in text_indices:
        if j >= score_len:
            break
        region = r.get_region_label(j)
        lbl = token_strings[j] if j < len(token_strings) else "?"
        bar_labels.append(f"[{j}] {_short(lbl)}")
        bar_values.append(float(activation_row[j]))
        bar_colors.append(REGION_COLORS.get(region, "#888888"))

    fig, ax = plt.subplots(figsize=(max(10, len(bar_labels) * 0.15), 5))
    x = np.arange(len(bar_labels))
    ax.bar(x, bar_values, color=bar_colors, width=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, rotation=90, fontsize=6)
    ax.set_ylabel("TAM Activation Score")

    target_str = (
        f"Token {token_idx}: '{_short(token_strings[token_idx])}'"
        if token_idx < len(token_strings)
        else f"Token {token_idx}"
    )
    ax.set_title(f"TAM Activation for {target_str} | All Layers (avg)\n{title}")

    regions_present = _collect_regions(r, score_len)
    _add_region_legend(ax, regions_present)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_activation_image_heatmap(
    image: np.ndarray,
    activation_img: np.ndarray,
    vision_shape: Tuple[int, ...],
    token_idx: int,
    token_str: str = "",
    title: str = "",
    save_path: Optional[str] = None,
):
    """3-panel image heatmap: Original | TAM Heatmap | Overlay.

    Args:
        image: Original image as (H, W, 3) RGB numpy array.
        activation_img: (num_image_tokens,) TAM scores for image region.
        vision_shape: (H_tokens, W_tokens) of the vision token grid.
        token_idx: Token being inspected.
        token_str: Decoded string for the token.
        title: Additional title text.
        save_path: Path to save.
    """
    if len(vision_shape) == 2:
        h_tok, w_tok = vision_shape
    elif len(vision_shape) == 3:
        _, h_tok, w_tok = vision_shape
    else:
        print(f"  Warning: unexpected vision_shape {vision_shape}, skipping heatmap")
        return

    expected_len = h_tok * w_tok
    if len(activation_img) != expected_len:
        if len(activation_img) > expected_len:
            activation_img = activation_img[:expected_len]
        else:
            activation_img = np.pad(
                activation_img, (0, expected_len - len(activation_img))
            )

    heatmap = activation_img.reshape(h_tok, w_tok)

    import cv2
    img_h, img_w = image.shape[:2]
    heatmap_resized = cv2.resize(
        heatmap.astype(np.float32), (img_w, img_h),
        interpolation=cv2.INTER_LINEAR,
    )

    hm_max = heatmap_resized.max()
    if hm_max > 0:
        heatmap_resized = heatmap_resized / hm_max

    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    overlay = (0.5 * image + 0.5 * heatmap_colored).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("TAM Activation Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    target_label = (
        f"Token {token_idx}: '{token_str}'" if token_str
        else f"Token {token_idx}"
    )
    fig.suptitle(
        f"Image Activation for {target_label}\n{title}", fontsize=12
    )
    fig.tight_layout()
    _save_or_show(fig, save_path)


def generate_image_heatmap_gif(
    image: np.ndarray,
    result_raw_scores: List[List[np.ndarray]],
    num_layers: int,
    token_regions: "TokenRegions",
    token_strings: List[str],
    vision_shape: Tuple[int, ...],
    input_len: int,
    num_generated: int,
    title: str = "",
    save_path: Optional[str] = None,
    fps: int = 4,
):
    """Create a GIF showing how the TAM image heatmap evolves over
    generated tokens.

    Each frame shows the overlay of the current token's layer-averaged
    image activation on the original image, with the token string annotated.

    Args:
        image: Original image as (H, W, 3) RGB numpy array.
        result_raw_scores: raw_scores from TAMResult.
        num_layers: Number of model layers.
        token_regions: TokenRegions for boundary info.
        token_strings: Decoded token strings for the full sequence.
        vision_shape: (H_tokens, W_tokens) of the vision grid.
        input_len: Number of input tokens.
        num_generated: Number of generated tokens.
        title: Model label for the frame title.
        save_path: Where to write the .gif file.
        fps: Frames per second for the GIF.
    """
    import io
    from PIL import Image as PILImage
    import cv2

    if save_path is None or vision_shape is None:
        return

    r = token_regions
    if len(vision_shape) == 2:
        h_tok, w_tok = vision_shape
    elif len(vision_shape) == 3:
        _, h_tok, w_tok = vision_shape
    else:
        return

    expected_len = h_tok * w_tok
    img_h, img_w = image.shape[:2]

    # Determine answer-only range
    answer_offset = (
        (r.answer_start - input_len)
        if r.answer_start is not None else 0
    )

    frames = []
    for gen_idx in range(answer_offset, num_generated):
        # Layer-averaged activation for this generated token
        layer_scores = [
            result_raw_scores[gen_idx][l]
            for l in range(num_layers)
        ]
        avg = np.mean(layer_scores, axis=0)

        # Normalize
        s_min, s_max = avg.min(), avg.max()
        if s_max - s_min > 1e-8:
            avg = (avg - s_min) / (s_max - s_min)
        else:
            avg = np.zeros_like(avg)

        # Extract image region
        img_act = avg[r.image_start:r.image_end]
        if len(img_act) != expected_len:
            if len(img_act) > expected_len:
                img_act = img_act[:expected_len]
            else:
                img_act = np.pad(img_act, (0, expected_len - len(img_act)))

        heatmap = img_act.reshape(h_tok, w_tok)
        heatmap_resized = cv2.resize(
            heatmap.astype(np.float32), (img_w, img_h),
            interpolation=cv2.INTER_LINEAR,
        )
        hm_max = heatmap_resized.max()
        if hm_max > 0:
            heatmap_resized = heatmap_resized / hm_max

        heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        overlay = (0.5 * image + 0.5 * heatmap_colored).astype(np.uint8)

        abs_idx = input_len + gen_idx
        tok_str = token_strings[abs_idx] if abs_idx < len(token_strings) else "?"
        label = f"Token {abs_idx}: '{tok_str.strip()}'"

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(overlay)
        ax.set_title(f"{title}\n{label}", fontsize=11)
        ax.axis("off")
        fig.tight_layout(pad=0.5)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(PILImage.open(buf).copy())
        buf.close()

    if frames:
        duration = int(1000 / fps)
        frames[0].save(
            save_path, save_all=True, append_images=frames[1:],
            duration=duration, loop=0,
        )
        print(f"  Saved GIF: {save_path} ({len(frames)} frames)")

        frames_dir = os.path.splitext(save_path)[0] + "_frames"
        os.makedirs(frames_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            frame.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))
        print(f"  Saved {len(frames)} frames to: {frames_dir}")


def plot_activation_layer_grid(
    per_layer: np.ndarray,
    token_idx: int,
    token_regions: TokenRegions,
    token_str: str = "",
    max_layers: int = 36,
    save_path: Optional[str] = None,
):
    """Grid of bar charts showing TAM activation at each layer for token i.

    Unlike the attention layer-x-head grid, TAM has no head dimension,
    so this is a single-dimensional grid of layers arranged in rows/columns.

    Args:
        per_layer: (L, score_len) normalized activation at each layer.
        token_idx: Token being inspected.
        token_regions: TokenRegions.
        token_str: Decoded string for the token.
        max_layers: Maximum layers to show (subsamples if more).
        save_path: Path to save.
    """
    L, seq = per_layer.shape
    r = token_regions

    # Subsample layers if too many
    if L > max_layers:
        layer_step = L // max_layers
        layer_indices = list(range(0, L, layer_step))[:max_layers]
        per_layer = per_layer[layer_indices]
        L_display = len(layer_indices)
    else:
        layer_indices = list(range(L))
        L_display = L

    # Arrange in a grid: determine rows and columns
    ncols = min(6, L_display)
    nrows = math.ceil(L_display / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 2.5, nrows * 1.5),
        squeeze=False,
    )

    for idx in range(nrows * ncols):
        row_i, col_i = divmod(idx, ncols)
        ax = axes[row_i][col_i]

        if idx >= L_display:
            ax.set_visible(False)
            continue

        layer_row = per_layer[idx]
        colors = [
            REGION_COLORS.get(r.get_region_label(j), "#888888")
            for j in range(len(layer_row))
        ]
        ax.bar(range(len(layer_row)), layer_row, color=colors, width=1.0)
        ax.set_xlim(0, len(layer_row))
        ax.set_ylim(0, max(layer_row.max() * 1.1, 0.01))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"L{layer_indices[idx]}", fontsize=7, pad=2)

    target_label = (
        f"Token {token_idx}: '{token_str}'" if token_str
        else f"Token {token_idx}"
    )
    fig.suptitle(
        f"TAM Activation Grid (per Layer) for {target_label}",
        fontsize=11,
    )
    fig.tight_layout()
    _save_or_show(fig, save_path)


# ======================================================================
# MODE B: Generation Summary
# ======================================================================

def plot_activation_ratio_over_generation(
    summary: List[Dict],
    title: str = "",
    save_path: Optional[str] = None,
):
    """Stacked area chart of image/prompt/generated activation ratios.

    Uses the same layout as the attention version: stacked area for >= 4
    data points, stacked bar for short generations.
    """
    if not summary:
        return

    steps = [s["token_idx"] for s in summary]
    img_ratios = [s["image_activation_ratio"] for s in summary]
    prompt_ratios = [s["prompt_activation_ratio"] for s in summary]
    gen_ratios = [s["generated_activation_ratio"] for s in summary]

    n = len(steps)
    colors = [
        REGION_COLORS["image"], REGION_COLORS["prompt"],
        REGION_COLORS["generated"],
    ]
    labels = ["Image", "Prompt", "Prior Generated"]

    fig, ax = plt.subplots(figsize=(max(6, min(12, n * 1.5)), 4))

    if n >= 4:
        ax.stackplot(
            range(n), img_ratios, prompt_ratios, gen_ratios,
            labels=labels, colors=colors, alpha=0.8,
        )
        ax.set_xlim(0, n - 1)
        _add_token_xticks(ax, summary, max_ticks=40)
    else:
        x = np.arange(n)
        bottom_prompt = np.array(img_ratios)
        bottom_gen = bottom_prompt + np.array(prompt_ratios)
        bar_w = 0.6
        ax.bar(x, img_ratios, bar_w, label=labels[0], color=colors[0], alpha=0.8)
        ax.bar(x, prompt_ratios, bar_w, bottom=bottom_prompt,
               label=labels[1], color=colors[1], alpha=0.8)
        ax.bar(x, gen_ratios, bar_w, bottom=bottom_gen,
               label=labels[2], color=colors[2], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_short(s["token_str"], 10) for s in summary], fontsize=8
        )

    ax.set_ylim(0, 1)
    ax.set_xlabel("Generation Step")
    ax.set_ylabel("Activation Ratio")
    ax.set_title(f"Activation Source Distribution Over Generation\n{title}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_activation_entropy_over_generation(
    summary_instruct: Optional[List[Dict]] = None,
    summary_thinking: Optional[List[Dict]] = None,
    title: str = "",
    save_path: Optional[str] = None,
):
    """Line chart of activation entropy per generated token."""
    fig, ax = plt.subplots(figsize=(12, 4))

    if summary_instruct:
        steps = range(len(summary_instruct))
        ent = [s["entropy"] for s in summary_instruct]
        ax.plot(steps, ent, label="Instruct", color="#2196F3", linewidth=1.2)

    if summary_thinking:
        steps = range(len(summary_thinking))
        ent = [s["entropy"] for s in summary_thinking]
        ax.plot(steps, ent, label="Thinking", color="#E74C3C", linewidth=1.2)
        _add_phase_markers(ax, summary_thinking)

    ax.set_xlabel("Generation Step")
    ax.set_ylabel("Activation Entropy (nats)")
    ax.set_title(f"TAM Activation Entropy Over Generation\n{title}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_top_activated_tokens_heatmap(
    summary: List[Dict],
    token_regions: Optional[TokenRegions] = None,
    num_tokens: int = 50,
    title: str = "",
    save_path: Optional[str] = None,
):
    """Sparse heatmap of top-5 most activated source tokens per generated token.

    Each row is normalized by its max; zeros are set to NaN (white background).
    """
    if not summary:
        return

    display = summary[:num_tokens]
    max_src = max(s["token_idx"] for s in display) + 1

    heatmap = np.zeros((len(display), max_src))
    for i, s in enumerate(display):
        for idx, val, _ in s["top_k_tokens"]:
            if idx < max_src:
                heatmap[i, idx] = val

    row_maxes = heatmap.max(axis=1, keepdims=True)
    row_maxes[row_maxes == 0] = 1.0
    heatmap = heatmap / row_maxes
    heatmap[heatmap == 0] = np.nan

    fig, ax = plt.subplots(figsize=(14, max(4, len(display) * 0.12)))
    ax.set_facecolor("white")
    im = ax.imshow(
        heatmap, aspect="auto", cmap="YlOrRd", interpolation="none",
        vmin=0, vmax=1,
    )
    ax.set_xlabel("Source Token Index")
    ax.set_ylabel("Generated Token")

    ytick_step = max(1, len(display) // 30)
    yticks = list(range(0, len(display), ytick_step))
    ax.set_yticks(yticks)
    ax.set_yticklabels(
        [f"{display[i]['token_str'][:8]}" for i in yticks], fontsize=6
    )

    if token_regions is not None:
        boundaries = [
            (token_regions.image_start, "image", "Img"),
            (token_regions.image_end, "image", None),
            (token_regions.prompt_start, "prompt", "Prompt"),
            (token_regions.generation_start, "generated", "Gen"),
        ]
        for pos, region, label in boundaries:
            if pos < max_src:
                color = REGION_COLORS.get(region, "#888888")
                ax.axvline(
                    x=pos, color=color, linestyle="--", alpha=0.6, linewidth=0.8
                )
                if label:
                    ax.text(
                        pos + 1, 0.5, label, fontsize=6, color=color,
                        va="bottom", ha="left",
                    )

    ax.set_title(f"Top-5 Activated Source Tokens (row-normalized)\n{title}")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Relative Activation (per row)")
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_reasoning_phase_transition(
    summary_thinking: List[Dict],
    title: str = "",
    save_path: Optional[str] = None,
):
    """Activation ratios with phase boundaries for the Thinking model.

    4-component stacked area: image / prompt / thinking / answer.
    """
    if not summary_thinking:
        return

    steps = range(len(summary_thinking))
    img_r = [s["image_activation_ratio"] for s in summary_thinking]
    prompt_r = [s["prompt_activation_ratio"] for s in summary_thinking]
    think_r = [s.get("thinking_activation_ratio", 0) for s in summary_thinking]
    ans_r = [s.get("answer_activation_ratio", 0) for s in summary_thinking]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.stackplot(
        steps, img_r, prompt_r, think_r, ans_r,
        labels=["Image", "Prompt", "Thinking (self)", "Answer (self)"],
        colors=[
            REGION_COLORS["image"], REGION_COLORS["prompt"],
            REGION_COLORS["thinking"], REGION_COLORS["answer"],
        ],
        alpha=0.8,
    )

    _add_phase_markers(ax, summary_thinking)

    ax.set_xlim(0, len(summary_thinking) - 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Generation Step")
    ax.set_ylabel("Activation Ratio")
    ax.set_title(f"Reasoning Phase Activation Transition\n{title}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    _save_or_show(fig, save_path)


# ======================================================================
# STRUCTURAL AND COMPARATIVE PLOTS
# ======================================================================

def plot_activation_matrix(
    result_raw_scores: List[List[np.ndarray]],
    num_layers: int,
    input_len: int,
    num_generated: int,
    token_labels: Optional[List[str]] = None,
    token_regions: Optional[TokenRegions] = None,
    title: str = "",
    save_path: Optional[str] = None,
    max_cols: int = 300,
    max_rows: int = 100,
):
    """Compact activation matrix: rows = generated tokens, columns = all source positions.

    Unlike the attention matrix (S x S for every token), TAM only produces
    scores for generated tokens, so we show a compact (num_generated x S)
    heatmap.  Each row is one generated token; each column is a source
    position in the full sequence.  Columns beyond the causal horizon of
    a given row are masked to NaN (white).

    Region boundaries (image, prompt, generation start) are drawn as
    vertical dashed lines on the x-axis.  For the y-axis, if the model
    is a thinking model, a horizontal line marks the think/answer
    transition.

    Args:
        result_raw_scores: raw_scores from TAMResult.
        num_layers: Number of layers.
        input_len: Length of the input sequence.
        num_generated: Number of generated tokens.
        token_labels: Decoded token strings for axis labels.
        token_regions: For drawing boundary lines.
        title: Plot title.
        save_path: Path to save.
        max_cols: Downsample columns if wider than this.
        max_rows: Downsample rows if taller than this.
    """
    if num_generated == 0:
        return

    # Total source positions = input_len + num_generated - 1
    # (the last generated token can see all prior positions)
    S = input_len + num_generated
    matrix = np.full((num_generated, S), np.nan)

    for gen_idx in range(num_generated):
        layer_scores = [
            result_raw_scores[gen_idx][l]
            for l in range(num_layers)
        ]
        avg_scores = np.mean(layer_scores, axis=0)
        score_len = min(len(avg_scores), S)
        matrix[gen_idx, :score_len] = avg_scores[:score_len]

    # Build label arrays before downsampling
    gen_labels = None
    src_labels = None
    if token_labels:
        gen_labels = token_labels[input_len:input_len + num_generated]
        src_labels = token_labels[:S]

    # Downsample columns if too wide
    col_step = 1
    if S > max_cols:
        col_step = S // max_cols
        matrix = matrix[:, ::col_step]
        if src_labels:
            src_labels = src_labels[::col_step]

    # Downsample rows if too tall
    row_step = 1
    if num_generated > max_rows:
        row_step = num_generated // max_rows
        matrix = matrix[::row_step, :]
        if gen_labels:
            gen_labels = gen_labels[::row_step]

    nrows_disp, ncols_disp = matrix.shape

    # Compute color scale from 95th percentile of valid values
    valid_vals = matrix[~np.isnan(matrix)]
    if len(valid_vals) > 0:
        vmax = float(np.percentile(valid_vals, 95))
        vmax = max(vmax, 1e-6)
    else:
        vmax = 1.0

    # Figure size: wider than tall since there are many more source
    # positions than generated tokens in typical runs
    fig_w = max(8, min(16, ncols_disp * 0.06))
    fig_h = max(4, min(12, nrows_disp * 0.12))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("white")

    im = ax.imshow(
        matrix, cmap="viridis", aspect="auto",
        norm=PowerNorm(gamma=0.3, vmin=0, vmax=vmax),
        interpolation="none",
    )

    ax.set_title(f"TAM Activation Matrix (Generated \u2192 Source)\n{title}")
    ax.set_xlabel("Source Position")
    ax.set_ylabel("Generated Token")

    # Y-axis: generated token labels
    if gen_labels and nrows_disp <= 60:
        yticks = list(range(nrows_disp))
        ax.set_yticks(yticks)
        ax.set_yticklabels(
            [_short(g, 8) for g in gen_labels], fontsize=5,
        )
    else:
        ytick_step = max(1, nrows_disp // 30)
        yticks = list(range(0, nrows_disp, ytick_step))
        ax.set_yticks(yticks)
        if gen_labels:
            ax.set_yticklabels(
                [_short(gen_labels[i], 8) for i in yticks], fontsize=5,
            )

    # X-axis: source position labels (subsampled)
    xtick_step = max(1, ncols_disp // 40)
    xticks = list(range(0, ncols_disp, xtick_step))
    ax.set_xticks(xticks)
    if src_labels:
        ax.set_xticklabels(
            [_short(src_labels[i], 6) for i in xticks],
            rotation=90, fontsize=4,
        )

    # Region boundary lines (vertical) on x-axis
    if token_regions is not None:
        x_boundaries = [
            (token_regions.image_start, "image", "Img"),
            (token_regions.image_end, "image", None),
            (token_regions.prompt_start, "prompt", "Prompt"),
            (token_regions.generation_start, "generated", "Gen"),
        ]
        for pos, region, label in x_boundaries:
            x_scaled = pos / col_step
            if 0 <= x_scaled < ncols_disp:
                color = REGION_COLORS.get(region, "#888888")
                ax.axvline(
                    x=x_scaled, color=color, linestyle="--",
                    alpha=0.7, linewidth=0.8,
                )
                if label:
                    ax.text(
                        x_scaled + 0.5, -0.5, label,
                        fontsize=6, color=color, va="bottom", ha="left",
                    )

        # Horizontal line at think/answer boundary (if thinking model)
        if (token_regions.think_end is not None
                and token_regions.answer_start is not None):
            y_think = (token_regions.answer_start - input_len) / row_step
            if 0 < y_think < nrows_disp:
                ax.axhline(
                    y=y_think, color=REGION_COLORS["thinking"],
                    linestyle="--", alpha=0.7, linewidth=1,
                )
                ax.text(
                    ncols_disp * 0.01, y_think - 0.5, "answer",
                    fontsize=6, color=REGION_COLORS["answer"],
                    va="top", ha="left",
                )

    fig.colorbar(im, ax=ax, shrink=0.6, label="TAM Score (layer-avg)")
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_activation_entropy_by_layer(
    metrics_instruct: Optional[Dict] = None,
    metrics_thinking: Optional[Dict] = None,
    title: str = "",
    save_path: Optional[str] = None,
):
    """Line plot of activation entropy across layers for one or both models."""
    fig, ax = plt.subplots(figsize=(10, 4))

    if metrics_instruct and "entropy_per_layer" in metrics_instruct:
        ent = metrics_instruct["entropy_per_layer"]
        ax.plot(
            range(len(ent)), ent, "o-", label="Instruct",
            color="#2196F3", markersize=3, linewidth=1.2,
        )

    if metrics_thinking and "entropy_per_layer" in metrics_thinking:
        ent = metrics_thinking["entropy_per_layer"]
        ax.plot(
            range(len(ent)), ent, "s-", label="Thinking",
            color="#E74C3C", markersize=3, linewidth=1.2,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Activation Entropy (nats)")
    ax.set_title(f"TAM Activation Entropy by Layer\n{title}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_image_activation_ratio_by_layer(
    metrics_instruct: Optional[Dict] = None,
    metrics_thinking: Optional[Dict] = None,
    title: str = "",
    save_path: Optional[str] = None,
    token_idx: Optional[int] = None,
    token_string: Optional[str] = None,
):
    """Image vs text activation across layers (two-panel plot).

    Left panel:  Line plot of mean image and text activation per layer.
    Right panel: Stacked bar chart of relative contribution (image vs text)
                 at each layer.

    Supports single-model (one metrics dict) or comparative (both dicts)
    mode.  In comparative mode the left panel overlays both models.

    Args:
        token_idx: Optional token index being inspected (shown in suptitle).
        token_string: Optional decoded token string (shown in suptitle).
    """
    # Different shades for instruct vs thinking in comparative mode
    # Image tokens: shades of red; Text tokens: shades of blue
    COLOR_IMG_INSTRUCT = "#e74c3c"   # bright red
    COLOR_IMG_THINKING = "#f5a6a6"   # light red / pink
    COLOR_TXT_INSTRUCT = "#2471a3"   # dark blue
    COLOR_TXT_THINKING = "#aed6f1"   # light blue
    key = "image_activation_ratio_per_layer"

    # Collect per-model data: list of (label, image_ratio_array)
    models = []
    if metrics_instruct and key in metrics_instruct:
        models.append(("Instruct", np.asarray(metrics_instruct[key])))
    if metrics_thinking and key in metrics_thinking:
        models.append(("Thinking", np.asarray(metrics_thinking[key])))

    if not models:
        return

    # Per-model color mapping
    img_colors = [COLOR_IMG_INSTRUCT, COLOR_IMG_THINKING]
    txt_colors = [COLOR_TXT_INSTRUCT, COLOR_TXT_THINKING]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Left panel: mean activation line plot ----
    line_styles = [("o-", 2), ("s--", 2)]
    for idx, (label, img_ratio) in enumerate(models):
        txt_ratio = 1.0 - img_ratio
        layers = np.arange(1, len(img_ratio) + 1)
        marker, lw = line_styles[idx % len(line_styles)]

        ax1.plot(layers, img_ratio, marker, color=img_colors[idx], linewidth=lw,
                 markersize=6, markerfacecolor="white", markeredgewidth=2,
                 label=f"Image Tokens ({label})")
        ax1.plot(layers, txt_ratio, marker, color=txt_colors[idx], linewidth=lw,
                 markersize=6, markerfacecolor="white", markeredgewidth=2,
                 label=f"Text Tokens ({label})")

    ax1.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Mean Activation", fontsize=12, fontweight="bold")
    ax1.set_title("Mean Activation Across Layers", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, frameon=True, edgecolor="black")
    ax1.grid(True, alpha=0.3)

    # ---- Right panel: stacked bar chart ----
    bar_label, bar_img = models[0]
    bar_txt = 1.0 - bar_img
    layers = np.arange(1, len(bar_img) + 1)
    width = 0.8

    ax2.bar(layers, bar_img, width, label="Image Tokens",
            color=img_colors[0], edgecolor="gray", linewidth=0.5)
    ax2.bar(layers, bar_txt, width, bottom=bar_img, label="Text Tokens",
            color=txt_colors[0], edgecolor="gray", linewidth=0.5)

    ax2.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Relative Avg Contribution", fontsize=12, fontweight="bold")
    ax2.set_title(f"Relative Avg Contribution ({bar_label})",
                  fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=10, frameon=True, edgecolor="black")
    ax2.grid(True, alpha=0.3, axis="y")
    num_layers = len(bar_img)
    ax2.set_xticks(layers[::max(1, num_layers // 10)])

    # Build suptitle with optional token info
    suptitle = f"{title}"
    if token_idx is not None and token_string is not None:
        suptitle += f" — Token #{token_idx}: {token_string}"
    elif token_idx is not None:
        suptitle += f" — Token #{token_idx}"

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_activation_image_comparison(
    activation_img_instruct: Optional[np.ndarray],
    activation_img_thinking: Optional[np.ndarray],
    image: np.ndarray,
    vision_shape: Tuple[int, ...],
    title: str = "",
    save_path: Optional[str] = None,
):
    """Side-by-side TAM image heatmaps for Instruct vs Thinking.

    Replaces rollout_comparison from the attention pipeline since TAM
    has no rollout concept.
    """
    import cv2

    panels = []
    labels_list = []

    for act_img, label in [
        (activation_img_instruct, "Instruct"),
        (activation_img_thinking, "Thinking"),
    ]:
        if act_img is None:
            continue

        if len(vision_shape) == 2:
            h_tok, w_tok = vision_shape
        else:
            _, h_tok, w_tok = vision_shape

        expected = h_tok * w_tok
        if len(act_img) != expected:
            act_img = np.pad(
                act_img, (0, max(0, expected - len(act_img)))
            )[:expected]

        heatmap = act_img.reshape(h_tok, w_tok)
        img_h, img_w = image.shape[:2]
        hm = cv2.resize(heatmap.astype(np.float32), (img_w, img_h))
        hm_max = hm.max()
        if hm_max > 0:
            hm = hm / hm_max
        hm_color = (plt.cm.jet(hm)[:, :, :3] * 255).astype(np.uint8)
        overlay = (0.5 * image + 0.5 * hm_color).astype(np.uint8)

        panels.append(overlay)
        labels_list.append(label)

    if not panels:
        return

    num_panels = len(panels) + 1
    fig, axes = plt.subplots(1, num_panels, figsize=(6 * num_panels, 5))
    if num_panels == 1:
        axes = [axes]

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, (panel, label) in enumerate(zip(panels, labels_list)):
        axes[i + 1].imshow(panel)
        axes[i + 1].set_title(f"{label} TAM")
        axes[i + 1].axis("off")

    fig.suptitle(f"TAM Activation Comparison\n{title}", fontsize=12)
    fig.tight_layout()
    _save_or_show(fig, save_path)


# ======================================================================
# HTML Report Generator
# ======================================================================

def save_comparative_report(
    output_dir: str,
    image_path: str = "",
    prompt: str = "",
    model_instruct_name: str = "",
    model_thinking_name: str = "",
    metrics_instruct: Optional[Dict] = None,
    metrics_thinking: Optional[Dict] = None,
    summary_instruct: Optional[List[Dict]] = None,
    summary_thinking: Optional[List[Dict]] = None,
):
    """Generate an HTML report embedding all TAM visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<title>TAM Analysis Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; max-width: 1200px; "
        "margin: 0 auto; padding: 20px; }",
        "h1 { color: #333; }",
        "h2 { color: #555; border-bottom: 1px solid #ddd; padding-bottom: 5px; }",
        "img { max-width: 100%; border: 1px solid #eee; margin: 10px 0; }",
        ".metrics { background: #f5f5f5; padding: 15px; border-radius: 5px; "
        "margin: 10px 0; }",
        ".note { background: #fff3cd; padding: 10px; border-radius: 5px; "
        "margin: 10px 0; font-size: 0.9em; }",
        "table { border-collapse: collapse; width: 100%; }",
        "td, th { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background: #f0f0f0; }",
        "</style>",
        "</head><body>",
        "<h1>Token Activation Map (TAM) Analysis Report</h1>",
    ]

    if prompt:
        html_parts.append(f"<p><strong>Prompt:</strong> {prompt}</p>")
    if model_instruct_name:
        html_parts.append(
            f"<p><strong>Instruct Model:</strong> {model_instruct_name}</p>"
        )
    if model_thinking_name:
        html_parts.append(
            f"<p><strong>Thinking Model:</strong> {model_thinking_name}</p>"
        )

    html_parts.append(
        "<div class='note'>"
        "<strong>Note:</strong> TAM (Token Activation Map) measures hidden-state "
        "activations projected through lm_head, not attention weights. "
        "Compared to the attention pipeline: (1) there is no head dimension, "
        "(2) there is no attention rollout, and (3) scores are logit values "
        "(clipped at 0) rather than probability distributions."
        "</div>"
    )

    # Metrics summary table
    if metrics_instruct or metrics_thinking:
        html_parts.append("<h2>Metrics Summary</h2>")
        html_parts.append("<div class='metrics'><table><tr><th>Metric</th>")
        if metrics_instruct:
            html_parts.append("<th>Instruct</th>")
        if metrics_thinking:
            html_parts.append("<th>Thinking</th>")
        html_parts.append("</tr>")

        all_keys = set()
        if metrics_instruct:
            all_keys.update(metrics_instruct.keys())
        if metrics_thinking:
            all_keys.update(metrics_thinking.keys())

        for key in sorted(all_keys):
            html_parts.append(f"<tr><td>{key}</td>")
            if metrics_instruct:
                val = metrics_instruct.get(key)
                html_parts.append(f"<td>{_fmt_metric(val)}</td>")
            if metrics_thinking:
                val = metrics_thinking.get(key)
                html_parts.append(f"<td>{_fmt_metric(val)}</td>")
            html_parts.append("</tr>")

        html_parts.append("</table></div>")

    # Embed all plots by scanning subdirectories
    for subdir_name in ["instruct", "thinking", "comparative"]:
        subdir = os.path.join(output_dir, subdir_name)
        if not os.path.isdir(subdir):
            continue
        png_files = sorted(
            f for f in os.listdir(subdir) if f.endswith(".png")
        )
        if png_files:
            html_parts.append(
                f"<h2>{subdir_name.title()} Visualizations</h2>"
            )
            for png in png_files:
                label = png.replace(".png", "").replace("_", " ").title()
                rel_path = f"{subdir_name}/{png}"
                html_parts.append(f"<h3>{label}</h3>")
                html_parts.append(f"<img src='{rel_path}' alt='{label}'>")

    # Also check for top-level PNGs
    top_pngs = sorted(
        f for f in os.listdir(output_dir)
        if f.endswith(".png") and os.path.isfile(os.path.join(output_dir, f))
    )
    if top_pngs:
        html_parts.append("<h2>Additional Plots</h2>")
        for png in top_pngs:
            label = png.replace(".png", "").replace("_", " ").title()
            html_parts.append(f"<h3>{label}</h3>")
            html_parts.append(f"<img src='{png}' alt='{label}'>")

    html_parts.extend(["</body></html>"])

    report_path = os.path.join(output_dir, "index.html")
    with open(report_path, "w") as f:
        f.write("\n".join(html_parts))

    print(f"  HTML report saved: {report_path}")


# ======================================================================
# Helpers
# ======================================================================

def _short(s: str, maxlen: int = 12) -> str:
    """Truncate a string for display."""
    s = s.replace("\n", "\\n").strip()
    return s[:maxlen] if len(s) > maxlen else s


def _fmt_metric(val) -> str:
    """Format a metric value for HTML display."""
    if val is None:
        return "-"
    if isinstance(val, np.ndarray):
        if val.size <= 4:
            return str(np.round(val, 4).tolist())
        return f"array(shape={val.shape}, mean={val.mean():.4f})"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _collect_regions(
    r: TokenRegions, max_idx: Optional[int] = None
) -> set:
    """Determine which regions are present up to max_idx."""
    regions = set()
    limit = max_idx if max_idx is not None else r.total_len

    if r.image_start < limit:
        regions.add("system")
    if r.image_start < limit and r.image_end > 0:
        regions.add("image")
    if r.prompt_start < limit and r.prompt_end > r.prompt_start:
        regions.add("prompt")
    if r.think_start is not None and r.think_start < limit:
        regions.add("thinking")
    if (r.think_end is not None and r.answer_start is not None
            and r.answer_start < limit):
        regions.add("answer")
    elif r.generation_start < limit and r.think_start is None:
        regions.add("generated")

    return regions


def _add_region_legend(ax, regions_present: Optional[set] = None):
    """Add a legend with region colors."""
    from matplotlib.patches import Patch
    if regions_present is not None:
        handles = [
            Patch(facecolor=REGION_COLORS[k], label=REGION_LABELS.get(k, k))
            for k in REGION_COLORS
            if k in regions_present
        ]
    else:
        handles = [
            Patch(facecolor=c, label=REGION_LABELS.get(k, k))
            for k, c in REGION_COLORS.items()
            if k in ("image", "prompt", "thinking", "answer", "generated")
        ]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=2)


def _add_token_xticks(ax, summary: List[Dict], max_ticks: int = 40):
    """Add subsampled token labels to x-axis."""
    n = len(summary)
    step = max(1, n // max_ticks)
    ticks = list(range(0, n, step))
    labels = [_short(summary[i]["token_str"], 8) for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=5)


def _add_phase_markers(ax, summary_thinking: List[Dict]):
    """Add vertical lines at thinking/answer phase boundaries."""
    prev_phase = None
    for i, s in enumerate(summary_thinking):
        phase = s.get("phase", "generated")
        if prev_phase != phase and prev_phase is not None:
            color = REGION_COLORS.get(phase, "#888888")
            ax.axvline(x=i, color=color, linestyle="--", alpha=0.7, linewidth=1)
            ax.text(
                i, ax.get_ylim()[1] * 0.95, f"  {phase}",
                fontsize=7, color=color, rotation=90, va="top",
            )
        prev_phase = phase
