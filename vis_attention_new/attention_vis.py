"""
attention_vis.py - Visualization functions for causal attention analysis.

Provides two analysis modes:
    Mode A (Deep Single-Token): Detailed causal attention for a chosen token i,
        showing which prior tokens 0..i-1 contributed to it.
    Mode B (Generation Summary): Compact overview across all generated tokens,
        showing how attention evolves during generation.

Also provides comparative plots for reasoning vs non-reasoning models and
an HTML report generator.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, PowerNorm
from typing import Optional, List, Dict, Tuple, Any

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from attention_map import TokenRegions

# ------------------------------------------------------------------
# Color and style constants
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
        os.makedirs(os.path.dirname(path) if "/" in path else ".", exist_ok=True)


def _save_or_show(fig, save_path: Optional[str]):
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ======================================================================
# MODE A: Deep Single-Token Causal Inspection
# ======================================================================

def plot_causal_attention_bar(
    attn_row: np.ndarray,
    token_strings: List[str],
    token_regions: TokenRegions,
    token_idx: int,
    layer_idx: Optional[int] = None,
    title: str = "",
    save_path: Optional[str] = None,
    max_text_tokens: int = 80,
):
    """Bar chart of token i's causal attention over tokens 0..i-1.

    Bars are colored by region (image, prompt, generated, thinking).
    Image tokens are aggregated into a single bar for readability.

    Args:
        attn_row: (token_idx+1,) attention weights (head-averaged for one layer,
                  or averaged across all layers).
        token_strings: Full list of decoded token strings.
        token_regions: TokenRegions.
        token_idx: The token being inspected.
        layer_idx: If set, label indicates a specific layer; else "all layers".
        title: Additional title text.
        save_path: Path to save the figure.
        max_text_tokens: Maximum number of individual text token bars to show.
    """
    r = token_regions
    row = attn_row[:token_idx + 1]

    # Aggregate image tokens into one bar
    img_s, img_e = r.image_start, min(r.image_end, token_idx + 1)
    img_agg = row[img_s:img_e].sum() if img_e > img_s else 0.0

    # Build bar data: image aggregate + individual text tokens
    bar_labels = []
    bar_values = []
    bar_colors = []

    # System tokens before image (usually few)
    for j in range(0, min(r.image_start, token_idx + 1)):
        bar_labels.append(f"[{j}] {_short(token_strings[j])}")
        bar_values.append(float(row[j]))
        bar_colors.append(REGION_COLORS["system"])

    # Aggregated image
    if img_e > img_s:
        bar_labels.append(f"[Image] ({img_e - img_s} tokens)")
        bar_values.append(float(img_agg))
        bar_colors.append(REGION_COLORS["image"])

    # Text tokens (prompt + generated up to token_idx)
    text_start = min(r.prompt_start, token_idx + 1)
    text_end = token_idx  # exclude token_idx itself
    text_indices = list(range(text_start, text_end + 1))

    # Subsample if too many
    if len(text_indices) > max_text_tokens:
        step = len(text_indices) // max_text_tokens
        text_indices = text_indices[::step]

    for j in text_indices:
        if j > token_idx:
            break
        region = r.get_region_label(j)
        bar_labels.append(f"[{j}] {_short(token_strings[j])}")
        bar_values.append(float(row[j]))
        bar_colors.append(REGION_COLORS.get(region, "#888888"))

    # Plot
    fig, ax = plt.subplots(figsize=(max(10, len(bar_labels) * 0.15), 5))
    x = np.arange(len(bar_labels))
    ax.bar(x, bar_values, color=bar_colors, width=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, rotation=90, fontsize=6)
    ax.set_ylabel("Attention Weight")
    layer_str = f"Layer {layer_idx}" if layer_idx is not None else "All Layers (avg)"
    target_str = f"Token {token_idx}: '{_short(token_strings[token_idx])}'"
    ax.set_title(f"Causal Attention for {target_str} | {layer_str}\n{title}")

    # Legend — only show regions actually present in the bar chart
    regions_present = _collect_regions_from_token_regions(r, token_idx)
    _add_region_legend(ax, regions_present=regions_present)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_causal_image_heatmap(
    image: np.ndarray,
    attn_row_image: np.ndarray,
    vision_shape: Tuple[int, ...],
    token_idx: int,
    token_str: str = "",
    title: str = "",
    save_path: Optional[str] = None,
):
    """Overlay token i's attention to image patches on the original image.

    Args:
        image: Original image as (H, W, 3) RGB numpy array.
        attn_row_image: (num_image_tokens,) attention weights for image region.
        vision_shape: (H_tokens, W_tokens) shape of the vision token grid.
        token_idx: Token being inspected.
        token_str: Decoded string for the token.
        title: Additional title text.
        save_path: Path to save.
    """
    if len(vision_shape) == 2:
        h_tok, w_tok = vision_shape
    elif len(vision_shape) == 3:
        # Video: use first frame shape
        _, h_tok, w_tok = vision_shape
    else:
        print(f"  Warning: unexpected vision_shape {vision_shape}, skipping heatmap")
        return

    expected_len = h_tok * w_tok
    if len(attn_row_image) != expected_len:
        # Try to truncate or pad
        if len(attn_row_image) > expected_len:
            attn_row_image = attn_row_image[:expected_len]
        else:
            attn_row_image = np.pad(
                attn_row_image, (0, expected_len - len(attn_row_image))
            )

    heatmap = attn_row_image.reshape(h_tok, w_tok)

    # Resize to image dimensions
    import cv2
    img_h, img_w = image.shape[:2]
    heatmap_resized = cv2.resize(
        heatmap.astype(np.float32), (img_w, img_h),
        interpolation=cv2.INTER_LINEAR
    )

    # Normalize
    hm_max = heatmap_resized.max()
    if hm_max > 0:
        heatmap_resized = heatmap_resized / hm_max

    # Apply colormap
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Blend
    overlay = (0.5 * image + 0.5 * heatmap_colored).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Attention Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    target_label = f"Token {token_idx}: '{token_str}'" if token_str else f"Token {token_idx}"
    fig.suptitle(f"Image Attention for {target_label}\n{title}", fontsize=12)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def generate_image_heatmap_gif(
    image: np.ndarray,
    attention_matrices,
    result,
    title: str = "",
    save_path: Optional[str] = None,
    fps: int = 4,
):
    """Create a GIF showing how the image attention heatmap evolves over
    generated tokens.

    Each frame shows the overlay of the current token's image attention on
    the original image, with the token string annotated.

    Args:
        image: Original image as (H, W, 3) RGB numpy array.
        attention_matrices: (L, H, S, S) attention tensor from AttentionResult.
        result: Full AttentionResult (for token_regions, token_strings, vision_shape).
        title: Model label for the frame title.
        save_path: Where to write the .gif file.
        fps: Frames per second for the GIF.
    """
    import io
    from PIL import Image as PILImage
    import cv2

    if save_path is None or result.vision_shape is None:
        return

    r = result.token_regions
    if len(result.vision_shape) == 2:
        h_tok, w_tok = result.vision_shape
    elif len(result.vision_shape) == 3:
        _, h_tok, w_tok = result.vision_shape
    else:
        return

    expected_len = h_tok * w_tok
    img_h, img_w = image.shape[:2]
    seq_len = attention_matrices.shape[2]

    # Determine answer-only range for generated tokens
    answer_start = r.answer_start if r.answer_start is not None else r.generation_start
    gen_end = min(r.total_len, seq_len)

    frames = []
    for tok_idx in range(answer_start, gen_end):
        # Average attention across layers and heads for this token
        row = attention_matrices[:, :, tok_idx, :tok_idx + 1].float().mean(dim=(0, 1)).numpy()

        # Extract image region
        img_attn = row[r.image_start:r.image_end]
        if len(img_attn) != expected_len:
            if len(img_attn) > expected_len:
                img_attn = img_attn[:expected_len]
            else:
                img_attn = np.pad(img_attn, (0, expected_len - len(img_attn)))

        heatmap = img_attn.reshape(h_tok, w_tok)
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

        tok_str = result.token_strings[tok_idx] if tok_idx < len(result.token_strings) else "?"
        label = f"Token {tok_idx}: '{tok_str.strip()}'"

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


def plot_causal_layer_head_grid(
    per_layer_head: np.ndarray,
    token_idx: int,
    token_regions: TokenRegions,
    token_str: str = "",
    selected_layers: Optional[List[int]] = None,
    max_layers: int = 12,
    save_path: Optional[str] = None,
):
    """Grid showing token i's causal attention at each layer and head.

    Args:
        per_layer_head: (L, H, token_idx+1) from get_causal_attention_for_token.
        token_idx: Token being inspected.
        token_regions: TokenRegions.
        token_str: Decoded string for the token.
        selected_layers: Layer indices (for labeling).
        max_layers: Maximum layers to show (subsamples if more).
        save_path: Path to save.
    """
    L, H, seq = per_layer_head.shape
    r = token_regions

    # Subsample layers if too many
    if L > max_layers:
        layer_step = L // max_layers
        layer_indices = list(range(0, L, layer_step))[:max_layers]
        per_layer_head = per_layer_head[layer_indices]
        L_display = len(layer_indices)
    else:
        layer_indices = list(range(L))
        L_display = L

    layer_labels = (
        [str(selected_layers[i]) for i in layer_indices]
        if selected_layers
        else [str(i) for i in layer_indices]
    )

    # For each layer/head, compute region-level summary (3 bars: image, prompt, gen)
    num_heads_display = min(H, 16)
    head_step = max(1, H // num_heads_display)
    head_indices = list(range(0, H, head_step))[:num_heads_display]

    fig, axes = plt.subplots(
        L_display, len(head_indices),
        figsize=(len(head_indices) * 1.2, L_display * 0.8),
        squeeze=False,
    )

    for li, l in enumerate(range(L_display)):
        for hi, h in enumerate(head_indices):
            ax = axes[li][hi]
            row = per_layer_head[l, h, :]

            # Color each position by region
            colors = []
            for j in range(len(row)):
                region = r.get_region_label(j)
                colors.append(REGION_COLORS.get(region, "#888888"))

            ax.bar(range(len(row)), row, color=colors, width=1.0)
            ax.set_xlim(0, len(row))
            ax.set_ylim(0, max(row.max() * 1.1, 0.01))
            ax.set_xticks([])
            ax.set_yticks([])

            if li == 0:
                ax.set_title(f"H{head_indices[hi]}", fontsize=7)
            if hi == 0:
                ax.set_ylabel(f"L{layer_labels[li]}", fontsize=7)

    target_label = f"Token {token_idx}: '{token_str}'" if token_str else f"Token {token_idx}"
    fig.suptitle(
        f"Causal Attention Grid (Layer x Head) for {target_label}",
        fontsize=11,
    )
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_causal_rollout_to_input(
    rollout_row: np.ndarray,
    image: Optional[np.ndarray],
    vision_shape: Optional[Tuple[int, ...]],
    token_strings: List[str],
    token_regions: TokenRegions,
    token_idx: int,
    title: str = "",
    save_path: Optional[str] = None,
):
    """Effective causal influence from all input tokens to token i after rollout.

    Shows image heatmap overlay + bar chart for text tokens.

    Args:
        rollout_row: (token_idx+1,) effective attention from rollout.
        image: Original image as (H, W, 3) RGB.
        vision_shape: (H_tok, W_tok) of vision tokens.
        token_strings: Decoded token strings.
        token_regions: TokenRegions.
        token_idx: Token being inspected.
        title: Additional title.
        save_path: Path to save.
    """
    r = token_regions
    has_image = (image is not None and vision_shape is not None
                 and r.image_end > r.image_start)

    if has_image:
        fig = plt.figure(figsize=(16, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])
        ax_img = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1])
    else:
        fig, ax_bar = plt.subplots(figsize=(12, 5))
        ax_img = None

    # Image heatmap
    if has_image and ax_img is not None:
        import cv2
        img_attn = rollout_row[r.image_start:r.image_end]
        if len(vision_shape) == 2:
            h_tok, w_tok = vision_shape
        else:
            _, h_tok, w_tok = vision_shape
        expected = h_tok * w_tok
        if len(img_attn) > expected:
            img_attn = img_attn[:expected]
        elif len(img_attn) < expected:
            img_attn = np.pad(img_attn, (0, expected - len(img_attn)))

        heatmap = img_attn.reshape(h_tok, w_tok)
        img_h, img_w = image.shape[:2]
        hm_resized = cv2.resize(heatmap.astype(np.float32), (img_w, img_h))
        hm_max = hm_resized.max()
        if hm_max > 0:
            hm_resized = hm_resized / hm_max
        hm_color = (plt.cm.jet(hm_resized)[:, :, :3] * 255).astype(np.uint8)
        overlay = (0.5 * image + 0.5 * hm_color).astype(np.uint8)
        ax_img.imshow(overlay)
        ax_img.set_title("Rollout: Image Attention")
        ax_img.axis("off")

    # Bar chart for text tokens
    text_start = r.prompt_start
    text_end = min(token_idx + 1, r.total_len)
    bar_labels = []
    bar_values = []
    bar_colors = []

    for j in range(text_start, text_end):
        if j >= r.image_start and j < r.image_end:
            continue
        region = r.get_region_label(j)
        bar_labels.append(f"[{j}] {_short(token_strings[j])}")
        bar_values.append(float(rollout_row[j]) if j < len(rollout_row) else 0.0)
        bar_colors.append(REGION_COLORS.get(region, "#888888"))

    if bar_labels:
        x = np.arange(len(bar_labels))
        ax_bar.bar(x, bar_values, color=bar_colors, width=0.8)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(bar_labels, rotation=90, fontsize=5)
        ax_bar.set_ylabel("Rollout Attention")
        ax_bar.set_title("Rollout: Text Token Attention")
        regions_present = _collect_regions_from_token_regions(r, token_idx)
        _add_region_legend(ax_bar, regions_present=regions_present)

    target_label = f"Token {token_idx}: '{_short(token_strings[token_idx])}'"
    fig.suptitle(f"Attention Rollout to {target_label}\n{title}", fontsize=12)
    fig.tight_layout()
    _save_or_show(fig, save_path)


# ======================================================================
# MODE B: Generation Summary
# ======================================================================

def plot_attention_ratio_over_generation(
    summary: List[Dict],
    title: str = "",
    save_path: Optional[str] = None,
):
    """Attention source distribution over generation.

    Uses a stacked area chart when there are enough data points (>= 4), or
    falls back to a stacked bar chart for short generations so that even a
    single generated token is clearly visible.
    """
    if not summary:
        return

    steps = [s["token_idx"] for s in summary]
    img_ratios = [s["image_attn_ratio"] for s in summary]
    prompt_ratios = [s["prompt_attn_ratio"] for s in summary]
    gen_ratios = [s["generated_attn_ratio"] for s in summary]

    n = len(steps)
    colors = [REGION_COLORS["image"], REGION_COLORS["prompt"], REGION_COLORS["generated"]]
    labels = ["Image", "Prompt", "Prior Generated"]

    fig, ax = plt.subplots(figsize=(max(6, min(12, n * 1.5)), 4))

    if n >= 4:
        # Stacked area chart for longer generations
        ax.stackplot(
            range(n),
            img_ratios, prompt_ratios, gen_ratios,
            labels=labels, colors=colors, alpha=0.8,
        )
        ax.set_xlim(0, n - 1)
        _add_token_xticks(ax, summary, max_ticks=40)
    else:
        # Stacked bar chart for short generations (1-3 tokens)
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
    ax.set_ylabel("Attention Ratio")
    ax.set_title(f"Attention Source Distribution Over Generation\n{title}")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_entropy_over_generation(
    summary_instruct: Optional[List[Dict]] = None,
    summary_thinking: Optional[List[Dict]] = None,
    title: str = "",
    save_path: Optional[str] = None,
):
    """Line chart of attention entropy per generated token for one or both models."""
    fig, ax = plt.subplots(figsize=(12, 4))

    if summary_instruct:
        steps = range(len(summary_instruct))
        ent = [s["entropy"] for s in summary_instruct]
        ax.plot(steps, ent, label="Instruct", color="#2196F3", linewidth=1.2)

    if summary_thinking:
        steps = range(len(summary_thinking))
        ent = [s["entropy"] for s in summary_thinking]
        ax.plot(steps, ent, label="Thinking", color="#E74C3C", linewidth=1.2)

        # Mark thinking phase
        _add_phase_markers(ax, summary_thinking)

    ax.set_xlabel("Generation Step")
    ax.set_ylabel("Attention Entropy (nats)")
    ax.set_title(f"Attention Entropy Over Generation\n{title}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_top_attended_tokens_heatmap(
    summary: List[Dict],
    token_regions: Optional[TokenRegions] = None,
    num_tokens: int = 50,
    title: str = "",
    save_path: Optional[str] = None,
):
    """Sparse heatmap: for each generated token, show top-5 attended source tokens.

    Rows = generated tokens, columns = source token indices.
    Zeros are set to NaN (white background) and each row is normalized by its
    max so the color gradient shows relative importance within each token's top-5.
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

    # Normalize each row by its max for relative importance
    row_maxes = heatmap.max(axis=1, keepdims=True)
    row_maxes[row_maxes == 0] = 1.0
    heatmap = heatmap / row_maxes

    # Set zeros to NaN so they render as white background
    heatmap[heatmap == 0] = np.nan

    fig, ax = plt.subplots(figsize=(14, max(4, len(display) * 0.12)))
    ax.set_facecolor("white")
    im = ax.imshow(
        heatmap, aspect="auto", cmap="YlOrRd", interpolation="none",
        vmin=0, vmax=1,
    )
    ax.set_xlabel("Source Token Index")
    ax.set_ylabel("Generated Token")

    # Y-axis labels
    ytick_step = max(1, len(display) // 30)
    yticks = list(range(0, len(display), ytick_step))
    ax.set_yticks(yticks)
    ax.set_yticklabels(
        [f"{display[i]['token_str'][:8]}" for i in yticks], fontsize=6
    )

    # Add region boundary lines if token_regions is provided
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
                ax.axvline(x=pos, color=color, linestyle="--", alpha=0.6, linewidth=0.8)
                if label:
                    ax.text(
                        pos + 1, 0.5, label, fontsize=6, color=color,
                        va="bottom", ha="left",
                    )

    ax.set_title(f"Top-5 Attended Source Tokens (row-normalized)\n{title}")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Relative Attention (per row)")
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_reasoning_phase_transition(
    summary_thinking: List[Dict],
    title: str = "",
    save_path: Optional[str] = None,
):
    """Attention ratios with phase boundaries for the Thinking model.

    Shows vertical lines at <think>/<\/think> transitions and plots
    image/prompt/thinking/answer attention ratios.
    """
    if not summary_thinking:
        return

    steps = range(len(summary_thinking))
    img_r = [s["image_attn_ratio"] for s in summary_thinking]
    prompt_r = [s["prompt_attn_ratio"] for s in summary_thinking]
    think_r = [s.get("thinking_attn_ratio", 0) for s in summary_thinking]
    ans_r = [s.get("answer_attn_ratio", 0) for s in summary_thinking]

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

    ax.set_xlim(0, len(steps) - 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Generation Step")
    ax.set_ylabel("Attention Ratio")
    ax.set_title(f"Reasoning Phase Attention Transition\n{title}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    _save_or_show(fig, save_path)


# ======================================================================
# COMPARATIVE AND STRUCTURAL PLOTS
# ======================================================================

def plot_attention_matrix(
    attn: np.ndarray,
    token_labels: Optional[List[str]] = None,
    token_regions: Optional[TokenRegions] = None,
    title: str = "",
    save_path: Optional[str] = None,
    max_size: int = 200,
):
    """Full causal (lower-triangular) attention heatmap.

    Args:
        attn: (seq_len, seq_len) attention matrix.
        token_labels: Labels for axes.
        token_regions: TokenRegions for boundary indicators.
        title: Plot title.
        save_path: Path to save.
        max_size: Downsample if sequence is longer.
    """
    S = attn.shape[0]
    downsample_step = 1
    if S > max_size:
        downsample_step = S // max_size
        attn = attn[::downsample_step, ::downsample_step]
        if token_labels:
            token_labels = token_labels[::downsample_step]

    attn = attn.copy().astype(np.float64)

    # Mask upper triangle (causal mask zeros) as NaN so they render as
    # white background and don't skew the color range.
    S_disp = attn.shape[0]
    mask = np.triu(np.ones((S_disp, S_disp), dtype=bool), k=1)
    attn[mask] = np.nan

    # Compute vmax from the 95th percentile of the lower-triangle values
    # so that a few extreme self-attention peaks don't wash everything out.
    lower_vals = attn[~mask]
    lower_vals = lower_vals[~np.isnan(lower_vals)]
    if len(lower_vals) > 0:
        vmax = float(np.percentile(lower_vals, 95))
        vmax = max(vmax, 1e-6)
    else:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(
        attn,
        cmap="viridis",
        aspect="auto",
        norm=PowerNorm(gamma=0.3, vmin=0, vmax=vmax),
    )
    ax.set_title(f"Causal Attention Matrix\n{title}")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")

    if token_labels and len(token_labels) <= 80:
        short_labels = [_short(t, 6) for t in token_labels]
        ax.set_xticks(range(len(short_labels)))
        ax.set_xticklabels(short_labels, rotation=90, fontsize=4)
        ax.set_yticks(range(len(short_labels)))
        ax.set_yticklabels(short_labels, fontsize=4)

    # Add region boundary lines if token_regions is provided
    if token_regions is not None:
        boundaries = [
            (token_regions.image_start, "image"),
            (token_regions.image_end, "image"),
            (token_regions.prompt_start, "prompt"),
            (token_regions.prompt_end, "prompt"),
            (token_regions.generation_start, "generated"),
        ]
        for pos, region in boundaries:
            scaled = pos / downsample_step
            color = REGION_COLORS.get(region, "#888888")
            ax.axhline(y=scaled, color=color, linestyle="--", alpha=0.5, linewidth=0.8)
            ax.axvline(x=scaled, color=color, linestyle="--", alpha=0.5, linewidth=0.8)

    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_rollout_comparison(
    rollout_instruct: Optional[np.ndarray],
    rollout_thinking: Optional[np.ndarray],
    image: np.ndarray,
    vision_shape: Tuple[int, ...],
    token_regions_instruct: Optional["TokenRegions"] = None,
    token_regions_thinking: Optional["TokenRegions"] = None,
    token_idx_instruct: Optional[int] = None,
    token_idx_thinking: Optional[int] = None,
    title: str = "",
    save_path: Optional[str] = None,
):
    """Side-by-side rollout heatmaps on the same image."""
    import cv2

    num_panels = sum([rollout_instruct is not None, rollout_thinking is not None])
    if num_panels == 0:
        return

    fig, axes = plt.subplots(1, num_panels + 1, figsize=(6 * (num_panels + 1), 5))
    if num_panels + 1 == 1:
        axes = [axes]

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    panel = 1
    for rollout, regions, tok_idx, label in [
        (rollout_instruct, token_regions_instruct, token_idx_instruct, "Instruct"),
        (rollout_thinking, token_regions_thinking, token_idx_thinking, "Thinking"),
    ]:
        if rollout is None or regions is None:
            continue

        tidx = tok_idx if tok_idx is not None else len(rollout) - 1
        row = rollout[tidx, :tidx + 1] if rollout.ndim == 2 else rollout

        img_attn = row[regions.image_start:regions.image_end]
        if len(vision_shape) == 2:
            h_tok, w_tok = vision_shape
        else:
            _, h_tok, w_tok = vision_shape

        expected = h_tok * w_tok
        if len(img_attn) != expected:
            img_attn = np.pad(img_attn, (0, max(0, expected - len(img_attn))))[:expected]

        heatmap = img_attn.reshape(h_tok, w_tok)
        img_h, img_w = image.shape[:2]
        hm = cv2.resize(heatmap.astype(np.float32), (img_w, img_h))
        hm_max = hm.max()
        if hm_max > 0:
            hm = hm / hm_max
        hm_color = (plt.cm.jet(hm)[:, :, :3] * 255).astype(np.uint8)
        overlay = (0.5 * image + 0.5 * hm_color).astype(np.uint8)

        axes[panel].imshow(overlay)
        axes[panel].set_title(f"{label} Rollout")
        axes[panel].axis("off")
        panel += 1

    fig.suptitle(f"Attention Rollout Comparison\n{title}", fontsize=12)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_entropy_by_layer(
    metrics_instruct: Optional[Dict] = None,
    metrics_thinking: Optional[Dict] = None,
    title: str = "",
    save_path: Optional[str] = None,
):
    """Line plot of attention entropy across layers for both models."""
    fig, ax = plt.subplots(figsize=(10, 4))

    if metrics_instruct and "entropy_per_layer" in metrics_instruct:
        ent = metrics_instruct["entropy_per_layer"]
        ax.plot(range(len(ent)), ent, "o-", label="Instruct",
                color="#2196F3", markersize=3, linewidth=1.2)

    if metrics_thinking and "entropy_per_layer" in metrics_thinking:
        ent = metrics_thinking["entropy_per_layer"]
        ax.plot(range(len(ent)), ent, "s-", label="Thinking",
                color="#E74C3C", markersize=3, linewidth=1.2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Attention Entropy (nats)")
    ax.set_title(f"Attention Entropy by Layer\n{title}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_head_specialization(
    metrics: Dict,
    model_label: str = "",
    save_path: Optional[str] = None,
):
    """Bar chart showing head specialization (std of image attention ratio)."""
    if "head_specialization" not in metrics:
        return

    spec = metrics["head_specialization"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(spec)), spec, color="#4A90D9", alpha=0.8)
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Specialization (std of image attn ratio)")
    ax.set_title(f"Head Specialization ({model_label})")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_image_attention_ratio_by_layer(
    metrics_instruct: Optional[Dict] = None,
    metrics_thinking: Optional[Dict] = None,
    title: str = "",
    save_path: Optional[str] = None,
    token_idx: Optional[int] = None,
    token_string: Optional[str] = None,
):
    """Image vs text attention across layers (two-panel plot).

    Left panel:  Line plot of mean image and text attention per layer.
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

    # Collect per-model data: list of (label, image_ratio_array)
    models = []
    if metrics_instruct and "image_attn_ratio_per_layer" in metrics_instruct:
        models.append(("Instruct", np.asarray(metrics_instruct["image_attn_ratio_per_layer"])))
    if metrics_thinking and "image_attn_ratio_per_layer" in metrics_thinking:
        models.append(("Thinking", np.asarray(metrics_thinking["image_attn_ratio_per_layer"])))

    if not models:
        return

    # Per-model color mapping
    img_colors = [COLOR_IMG_INSTRUCT, COLOR_IMG_THINKING]
    txt_colors = [COLOR_TXT_INSTRUCT, COLOR_TXT_THINKING]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Left panel: mean attention line plot ----
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
    ax1.set_ylabel("Mean Attention", fontsize=12, fontweight="bold")
    ax1.set_title("Mean Attention Across Layers", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, frameon=True, edgecolor="black")
    ax1.grid(True, alpha=0.3)

    # ---- Right panel: stacked bar chart ----
    # Use the first model's data for the stacked bar (single-model case)
    # or instruct if both are present.
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
    """Generate an HTML report with all visualizations.

    Assumes plots have already been saved as PNGs in output_dir.
    This function creates an index.html that references them.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all PNG files in output_dir
    png_files = sorted([
        f for f in os.listdir(output_dir) if f.endswith(".png")
    ])

    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<title>Attention Analysis Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }",
        "h1 { color: #333; }",
        "h2 { color: #555; border-bottom: 1px solid #ddd; padding-bottom: 5px; }",
        "img { max-width: 100%; border: 1px solid #eee; margin: 10px 0; }",
        ".metrics { background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0; }",
        "table { border-collapse: collapse; width: 100%; }",
        "td, th { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background: #f0f0f0; }",
        "</style>",
        "</head><body>",
        "<h1>Causal Attention Analysis Report</h1>",
    ]

    if prompt:
        html_parts.append(f"<p><strong>Prompt:</strong> {prompt}</p>")
    if model_instruct_name:
        html_parts.append(f"<p><strong>Instruct Model:</strong> {model_instruct_name}</p>")
    if model_thinking_name:
        html_parts.append(f"<p><strong>Thinking Model:</strong> {model_thinking_name}</p>")

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

    # Embed all plots
    html_parts.append("<h2>Visualizations</h2>")
    for png in png_files:
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


def _collect_regions_from_token_regions(
    r: TokenRegions, max_idx: Optional[int] = None
) -> set:
    """Determine which regions are present up to max_idx.

    Returns a set of region name strings suitable for _add_region_legend.
    """
    regions = set()
    limit = max_idx + 1 if max_idx is not None else r.total_len

    if r.image_start < limit:
        regions.add("system")
    if r.image_start < limit and r.image_end > 0:
        regions.add("image")
    if r.prompt_start < limit and r.prompt_end > r.prompt_start:
        regions.add("prompt")
    if r.think_start is not None and r.think_start < limit:
        regions.add("thinking")
    if r.think_end is not None and r.answer_start is not None and r.answer_start < limit:
        regions.add("answer")
    elif r.generation_start < limit and r.think_start is None:
        regions.add("generated")

    return regions


def _add_region_legend(ax, regions_present: Optional[set] = None):
    """Add a legend with region colors.

    Args:
        ax: Matplotlib axes.
        regions_present: Set of region names that actually appear in the data.
            If None, shows all non-system regions (legacy behavior).
    """
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
    ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=2)


def _add_token_xticks(ax, summary: List[Dict], max_ticks: int = 40):
    """Add subsampled token labels to x-axis."""
    n = len(summary)
    if n <= max_ticks:
        step = 1
    else:
        step = n // max_ticks

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
            ax.text(i, ax.get_ylim()[1] * 0.95, f"  {phase}",
                    fontsize=7, color=color, rotation=90, va="top")
        prev_phase = phase
