"""
attention_map.py - Core attention extraction and analysis for VLMs.

Extracts causal attention weights from Qwen3-VL models and provides:
- Per-token causal attention analysis (which prior tokens contributed to token i)
- Attention rollout across layers (effective multi-hop causal influence)
- Cross-modal attention (image <-> text)
- Quantitative metrics (entropy, sparsity, Gini coefficient, etc.)

Supports both Qwen3-VL-8B-Instruct (non-reasoning) and Qwen3-VL-8B-Thinking (reasoning),
with special handling for <think>...</think> reasoning phases.

Requirements:
    Model must be loaded with attn_implementation="eager" for attention extraction.
    Flash Attention / SDPA do not return attention weights.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any


@dataclass
class TokenRegions:
    """Index ranges for different token regions in the full sequence.

    All indices are positions in the full (input + generated) token sequence.
    Ranges follow Python convention: [start, end) is exclusive on end.
    """
    image_start: int
    image_end: int
    prompt_start: int
    prompt_end: int
    generation_start: int
    think_start: Optional[int] = None
    think_end: Optional[int] = None
    answer_start: Optional[int] = None
    total_len: int = 0

    @property
    def image_len(self):
        return self.image_end - self.image_start

    @property
    def prompt_len(self):
        return self.prompt_end - self.prompt_start

    @property
    def generation_len(self):
        return self.total_len - self.generation_start

    def get_region_label(self, idx: int) -> str:
        """Return which region a token index belongs to."""
        if self.image_start <= idx < self.image_end:
            return "image"
        if self.prompt_start <= idx < self.prompt_end:
            return "prompt"
        if self.think_start is not None and self.think_end is not None:
            if self.think_start <= idx <= self.think_end:
                return "thinking"
        if self.answer_start is not None and idx >= self.answer_start:
            return "answer"
        if idx >= self.generation_start:
            return "generated"
        return "system"

    def get_region_colors(self) -> Dict[str, str]:
        """Color map for each region, used in visualizations."""
        return {
            "system": "#888888",
            "image": "#4A90D9",
            "prompt": "#50B848",
            "thinking": "#E74C3C",
            "answer": "#F39C12",
            "generated": "#F39C12",
        }


@dataclass
class AttentionResult:
    """Structured result from attention extraction."""
    attention_matrices: torch.Tensor  # (num_layers, num_heads, seq_len, seq_len) fp16
    generated_ids: torch.Tensor
    token_regions: TokenRegions
    num_layers: int
    num_heads: int
    full_token_ids: List[int]
    token_strings: List[str]
    selected_layers: Optional[List[int]] = None
    vision_shape: Optional[Tuple[int, ...]] = None


class AttentionAnalyzer:
    """Extracts and analyzes causal attention from Qwen3-VL models.

    Usage:
        analyzer = AttentionAnalyzer(model, processor)
        result = analyzer.extract(inputs, max_new_tokens=256)
        causal = analyzer.get_causal_attention_for_token(result, token_idx=42)
        summary = analyzer.compute_generation_summary(result)
        metrics = analyzer.compute_metrics(result)
    """

    def __init__(self, model, processor, device="cuda"):
        self.model = model
        self.processor = processor
        self.device = device

        text_config = model.config.text_config
        self.num_layers = text_config.num_hidden_layers
        self.num_heads = text_config.num_attention_heads
        self.num_kv_heads = text_config.num_key_value_heads

        # Special token IDs from model config
        self.vision_start_id = model.config.vision_start_token_id
        self.vision_end_id = model.config.vision_end_token_id
        self.image_token_id = model.config.image_token_id

        # Thinking model tokens (looked up dynamically)
        self.think_start_id = self._get_token_id("<think>")
        self.think_end_id = self._get_token_id("</think>")

    # ------------------------------------------------------------------
    # Token ID helpers
    # ------------------------------------------------------------------

    def _get_token_id(self, token_str: str) -> Optional[int]:
        """Look up a special token's ID from the tokenizer."""
        tokenizer = self.processor.tokenizer
        try:
            tid = tokenizer.convert_tokens_to_ids(token_str)
            if tid != tokenizer.unk_token_id:
                return tid
        except Exception:
            pass
        try:
            ids = tokenizer.encode(token_str, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int = 256,
        selected_layers: Optional[List[int]] = None,
        do_sample: bool = False,
        vision_shape: Optional[Tuple[int, ...]] = None,
    ) -> AttentionResult:
        """Extract causal attention matrices via model.generate().

        Strategy:
            1. Run model.generate() with output_attentions=True to get per-step
               attention weights.
            2. Reconstruct the full causal attention matrix from the step-wise
               outputs (prefill + decode steps).

        Args:
            inputs: Tokenized inputs dict (input_ids, pixel_values, etc.)
            max_new_tokens: Maximum tokens to generate.
            selected_layers: Layer indices to extract (None = all layers).
                             Use e.g. [0, 8, 16, 24, 35] to reduce memory.
            do_sample: Whether to sample (False = greedy for reproducibility).
            vision_shape: Shape of the vision token grid (H, W) for image or
                          (T, H, W) for video. Extracted from image_grid_thw if
                          not provided.

        Returns:
            AttentionResult with full causal attention matrices and metadata.
        """
        input_len = inputs["input_ids"].shape[1]

        # Infer vision shape from processor output
        if vision_shape is None:
            vision_shape = self._infer_vision_shape(inputs)

        print(f"[AttentionAnalyzer] Generating with output_attentions=True "
              f"(input_len={input_len}, max_new_tokens={max_new_tokens})...")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_attentions=True,
                return_dict_in_generate=True,
                use_cache=True,
                do_sample=do_sample,
            )

        generated_ids = outputs.sequences  # (B, total_len)
        full_token_ids = generated_ids[0].cpu().tolist()
        num_generated = len(full_token_ids) - input_len

        print(f"[AttentionAnalyzer] Generated {num_generated} tokens "
              f"(total_len={len(full_token_ids)}). Reconstructing attention...")

        # Reconstruct full causal attention matrices
        attention_matrices = self._reconstruct_attention(
            outputs.attentions, input_len, selected_layers
        )

        # Free generation outputs
        del outputs
        torch.cuda.empty_cache()

        # Token regions
        token_regions = self.get_token_regions(full_token_ids, input_len)

        # Decode tokens to strings
        tokenizer = self.processor.tokenizer
        token_strings = [tokenizer.decode([tid]) for tid in full_token_ids]

        print(f"[AttentionAnalyzer] Attention shape: {attention_matrices.shape}")

        return AttentionResult(
            attention_matrices=attention_matrices,
            generated_ids=generated_ids,
            token_regions=token_regions,
            num_layers=attention_matrices.shape[0],
            num_heads=attention_matrices.shape[1],
            full_token_ids=full_token_ids,
            token_strings=token_strings,
            selected_layers=selected_layers,
            vision_shape=vision_shape,
        )

    def _infer_vision_shape(self, inputs: Dict) -> Optional[Tuple[int, ...]]:
        """Infer vision token grid shape from processor outputs."""
        if "image_grid_thw" in inputs:
            thw = inputs["image_grid_thw"][0].cpu().tolist()
            if thw[0] == 1:
                # Single image: (H, W) after spatial merge
                merge = getattr(self.model.config.vision_config, "spatial_merge_size", 2)
                return (int(thw[1]) // merge, int(thw[2]) // merge)
            else:
                merge = getattr(self.model.config.vision_config, "spatial_merge_size", 2)
                return (int(thw[0]), int(thw[1]) // merge, int(thw[2]) // merge)
        if "video_grid_thw" in inputs:
            thw = inputs["video_grid_thw"][0].cpu().tolist()
            merge = getattr(self.model.config.vision_config, "spatial_merge_size", 2)
            return (int(thw[0]), int(thw[1]) // merge, int(thw[2]) // merge)
        return None

    def _reconstruct_attention(
        self,
        raw_attentions: Tuple,
        input_len: int,
        selected_layers: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Reconstruct full causal attention matrices from step-wise outputs.

        raw_attentions structure (from model.generate with use_cache=True):
            raw_attentions[0]: prefill  -> tuple of L tensors, each (B, H, input_len, input_len)
            raw_attentions[k]: decode k -> tuple of L tensors, each (B, H, 1, input_len+k)

        Returns:
            (num_layers, num_heads, total_len, total_len) tensor in fp16.
        """
        num_steps = len(raw_attentions)
        num_generated = num_steps - 1
        total_len = input_len + num_generated
        num_all_layers = len(raw_attentions[0])
        H = raw_attentions[0][0].shape[1]

        layers_to_process = (
            selected_layers if selected_layers is not None
            else list(range(num_all_layers))
        )

        result_layers = []
        for l_idx in layers_to_process:
            layer_attn = torch.zeros(H, total_len, total_len, dtype=torch.float16)

            # Prefill: rows 0..input_len-1
            prefill = raw_attentions[0][l_idx][0]  # (H, input_len, input_len)
            layer_attn[:, :input_len, :input_len] = prefill.cpu().half()

            # Decode steps: each adds one row
            for k in range(1, num_steps):
                row_idx = input_len + k - 1
                kv_len = raw_attentions[k][l_idx].shape[-1]
                decode_row = raw_attentions[k][l_idx][0, :, 0, :]  # (H, kv_len)
                layer_attn[:, row_idx, :kv_len] = decode_row.cpu().half()

            result_layers.append(layer_attn)

        return torch.stack(result_layers, dim=0)  # (L, H, S, S)

    # ------------------------------------------------------------------
    # Token region identification
    # ------------------------------------------------------------------

    def get_token_regions(self, token_ids: List[int], input_len: int) -> TokenRegions:
        """Identify image / prompt / think / answer token regions.

        Uses the model's special token IDs (vision_start, vision_end) and the
        think/end-think tokens for reasoning models.  Falls back to string-based
        scanning of decoded tokens when the think tokens are not registered as
        single token IDs in the tokenizer.
        """
        image_start = 0
        image_end = 0

        # Find vision token boundaries
        for i, tid in enumerate(token_ids):
            if tid == self.vision_start_id:
                image_start = i + 1  # tokens after <|vision_start|>
                break

        for i in range(image_start, len(token_ids)):
            if token_ids[i] == self.vision_end_id:
                image_end = i
                break

        prompt_start = image_end + 1 if image_end > 0 else 0
        prompt_end = input_len
        generation_start = input_len

        # Locate <think>...</think> in generated tokens
        think_start = None
        think_end = None
        answer_start = None  # only set when a thinking phase is found

        # Strategy 1: token-ID-based lookup (fast, works when <think> is a
        # single registered token)
        if self.think_start_id is not None:
            for i in range(generation_start, len(token_ids)):
                if token_ids[i] == self.think_start_id and think_start is None:
                    think_start = i
                if (token_ids[i] == self.think_end_id
                        and think_start is not None
                        and think_end is None):
                    think_end = i
                    answer_start = i + 1
                    break

        # Strategy 2: string-based fallback — decode the generated portion
        # and search for <think> / </think> substrings in the running text.
        if think_start is None:
            tokenizer = self.processor.tokenizer
            gen_ids = token_ids[generation_start:]
            running_text = ""
            for j, tid in enumerate(gen_ids):
                running_text += tokenizer.decode([tid])
                if think_start is None and "<think>" in running_text:
                    think_start = generation_start + j
                if (think_start is not None
                        and think_end is None
                        and "</think>" in running_text):
                    think_end = generation_start + j
                    answer_start = think_end + 1
                    break

        return TokenRegions(
            image_start=image_start,
            image_end=image_end,
            prompt_start=prompt_start,
            prompt_end=prompt_end,
            generation_start=generation_start,
            think_start=think_start,
            think_end=think_end,
            answer_start=answer_start,
            total_len=len(token_ids),
        )

    # ------------------------------------------------------------------
    # Head / layer aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_heads(attn: torch.Tensor, method: str = "mean") -> torch.Tensor:
        """Aggregate attention across heads.

        Args:
            attn: (..., num_heads, seq_len, seq_len)
            method: "mean" or "max"

        Returns:
            Tensor with the head dimension removed.
        """
        head_dim = -3
        if method == "mean":
            return attn.mean(dim=head_dim)
        elif method == "max":
            return attn.max(dim=head_dim).values
        else:
            raise ValueError(f"Unknown head aggregation: {method}")

    @staticmethod
    def attention_rollout(
        attention_matrices: torch.Tensor,
        head_agg: str = "mean",
        start_layer: int = 0,
        end_layer: Optional[int] = None,
        residual_weight: float = 0.5,
    ) -> torch.Tensor:
        """Compute attention rollout across layers.

        For each layer, the residual connection is modeled by blending the
        attention matrix with the identity matrix.  Matrices are then
        multiplied across layers to get the effective attention.

        Args:
            attention_matrices: (num_layers, num_heads, seq_len, seq_len)
            head_agg: How to aggregate heads ("mean" or "max").
            start_layer: First layer (inclusive).
            end_layer: Last layer (exclusive). None = all.
            residual_weight: Weight for the identity (residual) component.

        Returns:
            rollout: (seq_len, seq_len) effective attention after rollout.
        """
        if end_layer is None:
            end_layer = attention_matrices.shape[0]

        layers = attention_matrices[start_layer:end_layer]

        # Aggregate heads -> (L, S, S)
        if head_agg == "mean":
            layers_agg = layers.float().mean(dim=1)
        elif head_agg == "max":
            layers_agg = layers.float().max(dim=1).values
        else:
            layers_agg = layers.float().mean(dim=1)

        seq_len = layers_agg.shape[-1]
        eye = torch.eye(seq_len, device=layers_agg.device)

        rollout = eye.clone()
        for layer_attn in layers_agg:
            attn_res = (1 - residual_weight) * layer_attn + residual_weight * eye
            attn_res = attn_res / (attn_res.sum(dim=-1, keepdim=True) + 1e-12)
            rollout = rollout @ attn_res

        return rollout

    # ------------------------------------------------------------------
    # Per-token causal attention (Mode A: Deep Inspection)
    # ------------------------------------------------------------------

    def get_causal_attention_for_token(
        self, result: AttentionResult, token_idx: int
    ) -> Dict[str, Any]:
        """For token i, extract its causal attention from every layer and head.

        Because the model is decoder-only with a causal mask, row i of the
        attention matrix at any layer is the distribution over tokens 0..i-1
        that contributed to token i.

        Args:
            result: AttentionResult from extract().
            token_idx: Index of the token to inspect.

        Returns:
            dict with:
                per_layer_head: (L, H, token_idx+1) raw causal attention
                per_layer:      (L, token_idx+1) head-averaged
                region_breakdown: {region_name -> (L,) attention mass per layer}
        """
        attn = result.attention_matrices  # (L, H, S, S)
        regions = result.token_regions

        # Clamp to valid range (the last generated token may not have an
        # attention row if it was never fed back through the model).
        seq_len = attn.shape[2]
        if token_idx < 0:
            token_idx = seq_len + token_idx
        token_idx = min(token_idx, seq_len - 1)

        # Row for this token from each layer/head: (L, H, token_idx+1)
        causal_row = attn[:, :, token_idx, :token_idx + 1].float()

        # Head-averaged: (L, token_idx+1)
        causal_row_avg = causal_row.mean(dim=1)

        # Region breakdown
        region_breakdown = self._compute_region_breakdown(
            causal_row_avg, token_idx, regions
        )

        return {
            "per_layer_head": causal_row.numpy(),
            "per_layer": causal_row_avg.numpy(),
            "region_breakdown": region_breakdown,
        }

    def _compute_region_breakdown(
        self,
        causal_row_avg: torch.Tensor,
        token_idx: int,
        regions: TokenRegions,
    ) -> Dict[str, np.ndarray]:
        """Sum causal attention mass by region for each layer."""
        breakdown = {}
        L = causal_row_avg.shape[0]

        region_ranges = [
            ("system", 0, regions.image_start),
            ("image", regions.image_start, regions.image_end),
            ("prompt", regions.prompt_start, regions.prompt_end),
        ]

        # Handle thinking / answer / generated regions
        if regions.think_start is not None and regions.think_end is not None:
            region_ranges.append(
                ("thinking", regions.think_start, regions.think_end + 1)
            )
            if regions.answer_start is not None:
                region_ranges.append(
                    ("answer", regions.answer_start, regions.total_len)
                )
        else:
            region_ranges.append(
                ("generated", regions.generation_start, regions.total_len)
            )

        for name, start, end in region_ranges:
            eff_start = max(start, 0)
            eff_end = min(end, token_idx + 1)
            if eff_start < eff_end:
                breakdown[name] = (
                    causal_row_avg[:, eff_start:eff_end].sum(dim=-1).numpy()
                )
            else:
                breakdown[name] = np.zeros(L)

        return breakdown

    def compute_causal_contribution(
        self,
        result: AttentionResult,
        token_idx: int,
        head_agg: str = "mean",
    ) -> np.ndarray:
        """Effective causal contribution to token i via attention rollout.

        Propagates attention through all layers to compute the effective
        influence of each input token on token i.

        Returns:
            contribution: (token_idx+1,) array of effective attention weights.
        """
        rollout = self.attention_rollout(
            result.attention_matrices, head_agg=head_agg
        )
        return rollout[token_idx, :token_idx + 1].numpy()

    # ------------------------------------------------------------------
    # Generation summary (Mode B)
    # ------------------------------------------------------------------

    def compute_generation_summary(
        self, result: AttentionResult, head_agg: str = "mean"
    ) -> List[Dict[str, Any]]:
        """For every generated token, compute a compact causal summary.

        Returns a list of dicts (one per generated token) containing:
            token_idx, token_str, image_attn_ratio, prompt_attn_ratio,
            generated_attn_ratio, entropy, top_k_tokens, phase,
            and optionally thinking_attn_ratio / answer_attn_ratio.
        """
        regions = result.token_regions
        attn = result.attention_matrices.float()  # (L, H, S, S)
        seq_len = attn.shape[2]  # may be < total_len (last token has no attn row)

        # Average across layers and heads -> (S, S)
        attn_avg = attn.mean(dim=(0, 1))

        summary = []
        gen_start = regions.generation_start

        for i in range(gen_start, min(regions.total_len, seq_len)):
            row = attn_avg[i, :i + 1]

            # Region attention masses
            img_mass = self._slice_sum(row, regions.image_start, regions.image_end)
            prompt_mass = self._slice_sum(
                row, regions.prompt_start, min(regions.prompt_end, i + 1)
            )
            gen_mass = self._slice_sum(row, gen_start, i) if i > gen_start else 0.0
            total = img_mass + prompt_mass + gen_mass + 1e-12

            # Entropy
            p = row.clamp(min=1e-12)
            entropy = -(p * p.log()).sum().item()

            # Top-5
            k = min(5, len(row))
            topk_vals, topk_ids = row.topk(k)
            top_k = [
                (int(idx), float(val), result.token_strings[int(idx)])
                for idx, val in zip(topk_ids, topk_vals)
            ]

            # Phase label
            phase = self._get_phase(i, regions)

            entry = {
                "token_idx": i,
                "token_str": result.token_strings[i],
                "image_attn_ratio": img_mass / total,
                "prompt_attn_ratio": prompt_mass / total,
                "generated_attn_ratio": gen_mass / total,
                "entropy": entropy,
                "top_k_tokens": top_k,
                "phase": phase,
            }

            # Thinking model breakdown
            if regions.think_start is not None:
                ts = regions.think_start
                te = regions.think_end if regions.think_end is not None else i
                think_mass = (
                    self._slice_sum(row, ts, min(te + 1, i + 1))
                    if ts < i + 1 else 0.0
                )
                ans_start = regions.answer_start or gen_start
                ans_mass = (
                    self._slice_sum(row, ans_start, i)
                    if ans_start < i else 0.0
                )
                entry["thinking_attn_ratio"] = think_mass / total
                entry["answer_attn_ratio"] = ans_mass / total

            summary.append(entry)

        return summary

    @staticmethod
    def _slice_sum(row: torch.Tensor, start: int, end: int) -> float:
        if start >= end or start >= len(row):
            return 0.0
        return row[start:end].sum().item()

    @staticmethod
    def _get_phase(idx: int, regions: TokenRegions) -> str:
        if regions.think_start is not None and regions.think_end is not None:
            if regions.think_start <= idx <= regions.think_end:
                return "thinking"
            if regions.answer_start is not None and idx >= regions.answer_start:
                return "answer"
        return "generated"

    # ------------------------------------------------------------------
    # Quantitative metrics
    # ------------------------------------------------------------------

    def compute_metrics(self, result: AttentionResult) -> Dict[str, np.ndarray]:
        """Compute quantitative attention metrics.

        Returns dict with per-layer and per-head metrics:
            entropy_per_layer:          (L,)
            entropy_per_layer_head:     (L, H)
            image_attn_ratio_per_layer: (L,)
            image_attn_ratio_per_layer_head: (L, H)
            sparsity_per_layer:         (L,)
            gini_image_per_layer:       (L,)
            head_specialization:        (H,)
        """
        attn = result.attention_matrices.float()  # (L, H, S, S)
        regions = result.token_regions
        L, H, S, _ = attn.shape
        metrics: Dict[str, np.ndarray] = {}

        gen_start = regions.generation_start
        if gen_start >= S:
            return metrics

        # Attention from generated tokens only
        gen_attn = attn[:, :, gen_start:, :]  # (L, H, gen_len, S)

        # --- Entropy ---
        p = gen_attn.clamp(min=1e-12)
        ent = -(p * p.log()).sum(dim=-1)  # (L, H, gen_len)
        metrics["entropy_per_layer_head"] = ent.mean(dim=-1).numpy()
        metrics["entropy_per_layer"] = ent.mean(dim=(1, 2)).numpy()

        # --- Image attention ratio ---
        img_s, img_e = regions.image_start, regions.image_end
        if img_e > img_s:
            img_mass = gen_attn[:, :, :, img_s:img_e].sum(dim=-1)
            total_mass = gen_attn.sum(dim=-1) + 1e-12
            ratio = img_mass / total_mass
            metrics["image_attn_ratio_per_layer_head"] = ratio.mean(dim=-1).numpy()
            metrics["image_attn_ratio_per_layer"] = ratio.mean(dim=(1, 2)).numpy()

        # --- Sparsity (fraction of weights above 1/S) ---
        threshold = 1.0 / S
        sparse = (gen_attn > threshold).float().mean(dim=-1)
        metrics["sparsity_per_layer"] = sparse.mean(dim=(1, 2)).numpy()

        # --- Gini coefficient on image attention ---
        if img_e > img_s:
            img_vals = gen_attn[:, :, :, img_s:img_e]
            gini = self._gini_coefficient(img_vals)
            metrics["gini_image_per_layer"] = gini.mean(dim=(1, 2)).numpy()

        # --- Head specialization ---
        if "image_attn_ratio_per_layer_head" in metrics:
            metrics["head_specialization"] = (
                metrics["image_attn_ratio_per_layer_head"].std(axis=0)
            )

        return metrics

    @staticmethod
    def _gini_coefficient(values: torch.Tensor) -> torch.Tensor:
        """Gini coefficient along the last dimension."""
        sorted_vals, _ = values.sort(dim=-1)
        n = sorted_vals.shape[-1]
        if n == 0:
            return torch.zeros_like(values[..., 0])
        indices = torch.arange(1, n + 1, device=sorted_vals.device, dtype=sorted_vals.dtype)
        weighted = (indices * sorted_vals).sum(dim=-1)
        total = sorted_vals.sum(dim=-1) + 1e-12
        return (2 * weighted) / (n * total) - (n + 1) / n

    # ------------------------------------------------------------------
    # Cross-modal attention
    # ------------------------------------------------------------------

    @staticmethod
    def get_cross_modal_attention(
        attn_matrix: torch.Tensor,
        token_regions: TokenRegions,
    ) -> Dict[str, torch.Tensor]:
        """Extract cross-modal attention sub-matrices.

        Args:
            attn_matrix: (seq_len, seq_len) or (H, seq_len, seq_len)
            token_regions: TokenRegions

        Returns:
            dict with 'text_to_image': attention from text tokens to image tokens
        """
        r = token_regions
        if attn_matrix.dim() == 2:
            text_to_image = attn_matrix[r.prompt_start:, r.image_start:r.image_end]
        else:
            text_to_image = attn_matrix[:, r.prompt_start:, r.image_start:r.image_end]

        return {"text_to_image": text_to_image}


# ------------------------------------------------------------------
# Per-token attention ratios (used by eval scripts)
# ------------------------------------------------------------------

def _find_answer_tag_span(token_strings, gen_start, seq_len):
    """Scan generated tokens for <answer>...</answer> and return (content_start, content_end).

    Returns the token indices of the first content token after <answer> and
    the first token of </answer>.  Returns (None, None) if not found.
    """
    tag_start = None
    tag_end = None
    running = ""
    for i in range(gen_start, min(len(token_strings), seq_len)):
        running += token_strings[i]
        if tag_start is None and "<answer>" in running:
            tag_start = i + 1
        elif tag_start is not None and "</answer>" in running:
            tag_text = ""
            for j in range(i, tag_start - 1, -1):
                tag_text = token_strings[j] + tag_text
                if "</answer>" in tag_text:
                    tag_end = j
                    break
            if tag_end is None:
                tag_end = i
            break
    return tag_start, tag_end


def compute_per_token_attention_ratios(result: "AttentionResult") -> dict:
    """Compute per-token, per-layer attention ratios to image / prompt / reasoning.

    All ratios are head-averaged: for each generated token *i* and layer *l*,
    the ratio is ``attn[l, :, i, region].sum(-1).mean(heads)``.

    Returns a dict with two sub-dicts::

        {
            "all_per_token_metrics": {
                "image_attn_per_token":     (gen_len, L) nested list,
                "prompt_attn_per_token":    (gen_len, L),
                "reasoning_attn_per_token": (gen_len, L) or [] if no thinking,
                "generated_tokens":         list[str] of length gen_len,
            },
            "per_answer_token_metrics": {
                "image_attn_per_ans_token":     (ans_len, L),
                "prompt_attn_per_ans_token":    (ans_len, L),
                "reasoning_attn_per_ans_token": (ans_len, L) or [],
                "generated_tokens":             list[str] of length ans_len,
            },
        }
    """
    attn = result.attention_matrices.float()  # (L, H, S, S)
    regions = result.token_regions
    L, H, S, _ = attn.shape
    gen_start = regions.generation_start

    img_s, img_e = regions.image_start, regions.image_end
    prompt_s, prompt_e = regions.prompt_start, regions.prompt_end

    # --- All generated tokens ---
    gen_attn = attn[:, :, gen_start:S, :]  # (L, H, gen_len, S)

    image_ratio = gen_attn[:, :, :, img_s:img_e].sum(dim=-1).mean(dim=1)   # (L, gen_len)
    prompt_ratio = gen_attn[:, :, :, prompt_s:prompt_e].sum(dim=-1).mean(dim=1)

    has_thinking = (regions.think_start is not None
                    and regions.think_end is not None)
    if has_thinking:
        ts, te = regions.think_start, regions.think_end
        reasoning_ratio = gen_attn[:, :, :, ts:te + 1].sum(dim=-1).mean(dim=1)
        reasoning_list = reasoning_ratio.T.tolist()
    else:
        reasoning_list = []

    gen_tokens = result.token_strings[gen_start:S]

    all_metrics = {
        "image_attn_per_token": image_ratio.T.tolist(),
        "prompt_attn_per_token": prompt_ratio.T.tolist(),
        "reasoning_attn_per_token": reasoning_list,
        "generated_tokens": gen_tokens,
    }

    # --- Answer tokens only ---
    ans_start, ans_end = _find_answer_tag_span(
        result.token_strings, gen_start, S,
    )
    if ans_start is None:
        if regions.answer_start is not None:
            ans_start = regions.answer_start
        else:
            ans_start = gen_start
    if ans_end is None:
        ans_end = S

    ans_start = max(ans_start, gen_start)
    ans_end = min(ans_end, S)

    if ans_start < ans_end:
        ans_offset = ans_start - gen_start
        ans_len = ans_end - ans_start
        ans_image = image_ratio[:, ans_offset:ans_offset + ans_len].T.tolist()
        ans_prompt = prompt_ratio[:, ans_offset:ans_offset + ans_len].T.tolist()
        if has_thinking:
            ans_reasoning = reasoning_ratio[:, ans_offset:ans_offset + ans_len].T.tolist()
        else:
            ans_reasoning = []
        ans_tokens = result.token_strings[ans_start:ans_end]
    else:
        ans_image = []
        ans_prompt = []
        ans_reasoning = []
        ans_tokens = []

    answer_metrics = {
        "image_attn_per_ans_token": ans_image,
        "prompt_attn_per_ans_token": ans_prompt,
        "reasoning_attn_per_ans_token": ans_reasoning,
        "generated_tokens": ans_tokens,
    }

    return {
        "all_per_token_metrics": all_metrics,
        "per_answer_token_metrics": answer_metrics,
    }
