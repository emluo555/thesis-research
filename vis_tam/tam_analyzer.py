"""
tam_analyzer.py - Core TAM (Token Activation Map) extraction and analysis.

Extracts hidden-state activations from Qwen3-VL models and provides:
- Per-token activation analysis (which input positions activate for token i)
- Cross-modal activation (image <-> text)
- Quantitative metrics (entropy, image activation ratio, sparsity)

Mirrors attention_map.py structure for direct comparison between
hidden-state activations (TAM) and attention weights.

Key structural differences from attention:
    - No head dimension: hidden states are full-rank vectors, not split into heads.
    - No rollout: TAM has per-layer logit projections, not composable matrices.
    - Score scale: TAM scores are clipped-ReLU logit values, not probabilities.
    - Model loading: Does NOT require attn_implementation="eager".
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

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
        return {
            "system": "#888888",
            "image": "#4A90D9",
            "prompt": "#50B848",
            "thinking": "#E74C3C",
            "answer": "#F39C12",
            "generated": "#F39C12",
        }


@dataclass
class TAMResult:
    """Structured result from TAM extraction.

    raw_scores[gen_idx][layer_idx] is a numpy array of shape
    (input_len + gen_idx,) containing the clipped-ReLU logit values for the
    predicted class at each sequence position.  This is the core TAM signal:
    at position j, the hidden state's projection through lm_head for the
    target class measures how much that position "activates" the prediction.
    """
    raw_scores: List[List[np.ndarray]]
    full_token_ids: List[int]
    token_strings: List[str]
    token_regions: TokenRegions
    num_layers: int
    num_generated: int
    vision_shape: Optional[Tuple[int, ...]] = None
    input_len: int = 0


# ------------------------------------------------------------------
# Analyzer
# ------------------------------------------------------------------

class TAMAnalyzer:
    """Extracts and analyzes Token Activation Maps from Qwen3-VL models.

    Usage:
        analyzer = TAMAnalyzer(model, processor)
        result = analyzer.extract(inputs, max_new_tokens=256)
        activation = analyzer.get_activation_for_token(result, gen_idx=0)
        summary = analyzer.compute_generation_summary(result)
        metrics = analyzer.compute_metrics(result)
    """

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

        text_config = model.config.text_config
        self.num_hidden_layers = text_config.num_hidden_layers

        self.vision_start_id = model.config.vision_start_token_id
        self.vision_end_id = model.config.vision_end_token_id

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
    # Vision shape inference
    # ------------------------------------------------------------------

    def _infer_vision_shape(self, inputs: Dict) -> Optional[Tuple[int, ...]]:
        """Infer vision token grid shape from processor outputs."""
        if "image_grid_thw" in inputs:
            thw = inputs["image_grid_thw"][0].cpu().tolist()
            merge = getattr(
                self.model.config.vision_config, "spatial_merge_size", 2
            )
            if thw[0] == 1:
                return (int(thw[1]) // merge, int(thw[2]) // merge)
            else:
                return (int(thw[0]), int(thw[1]) // merge, int(thw[2]) // merge)
        if "video_grid_thw" in inputs:
            thw = inputs["video_grid_thw"][0].cpu().tolist()
            merge = getattr(
                self.model.config.vision_config, "spatial_merge_size", 2
            )
            return (int(thw[0]), int(thw[1]) // merge, int(thw[2]) // merge)
        return None

    # ------------------------------------------------------------------
    # Token region identification
    # ------------------------------------------------------------------

    def get_token_regions(
        self, token_ids: List[int], input_len: int
    ) -> TokenRegions:
        """Identify image / prompt / think / answer token regions.

        Uses the model's special token IDs (vision_start, vision_end) and the
        think/end-think tokens for reasoning models.  Falls back to string-based
        scanning of decoded tokens when the think tokens are not registered as
        single token IDs in the tokenizer.
        """
        image_start = 0
        image_end = 0

        for i, tid in enumerate(token_ids):
            if tid == self.vision_start_id:
                image_start = i + 1
                break

        for i in range(image_start, len(token_ids)):
            if token_ids[i] == self.vision_end_id:
                image_end = i
                break

        prompt_start = image_end + 1 if image_end > 0 else 0
        prompt_end = input_len
        generation_start = input_len

        think_start = None
        think_end = None
        answer_start = None

        # Strategy 1: token-ID-based lookup
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

        # Strategy 2: string-based fallback
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
    # Extraction
    # ------------------------------------------------------------------

    def extract(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int = 256,
        do_sample: bool = False,
    ) -> TAMResult:
        """Extract TAM scores via model.generate() with hidden states.

        Strategy:
            1. Run model.generate() with output_hidden_states=True.
            2. Project each layer's hidden state through lm_head to get logits.
            3. For each generated token, extract the logit for the predicted
               class at every sequence position -- this is the TAM activation.
            4. Clip negative values (ReLU) since only positive activations
               are meaningful for the TAM signal.

        Args:
            inputs: Tokenized inputs dict (input_ids, pixel_values, etc.)
            max_new_tokens: Maximum tokens to generate.
            do_sample: Whether to sample (False = greedy for reproducibility).

        Returns:
            TAMResult with raw TAM scores and metadata.
        """
        input_len = inputs["input_ids"].shape[1]
        vision_shape = self._infer_vision_shape(inputs)

        print(f"[TAMAnalyzer] Generating with output_hidden_states=True "
              f"(input_len={input_len}, max_new_tokens={max_new_tokens})...")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                use_cache=True,
                do_sample=do_sample,
            )

        generated_ids = outputs.sequences
        full_token_ids = generated_ids[0].cpu().tolist()
        num_generated = len(full_token_ids) - input_len
        num_steps = len(outputs.hidden_states)
        # Includes embedding layer (index 0) + transformer layers
        num_layers = len(outputs.hidden_states[0])

        print(f"[TAMAnalyzer] Generated {num_generated} tokens "
              f"(total_len={len(full_token_ids)}). "
              f"Computing per-layer logits ({num_layers} layers)...")

        # Compute per-layer logits for all generation steps.
        # per_layer_logits[layer][step] = (1, seq_len_at_step, vocab_size)
        per_layer_logits: List[List[torch.Tensor]] = [
            [] for _ in range(num_layers)
        ]
        for t in range(num_steps):
            for layer in range(num_layers):
                layer_hs = outputs.hidden_states[t][layer]
                logits = self.model.lm_head(layer_hs)
                per_layer_logits[layer].append(logits)

        del outputs
        torch.cuda.empty_cache()

        # Token metadata
        token_regions = self.get_token_regions(full_token_ids, input_len)
        tokenizer = self.processor.tokenizer
        token_strings = [tokenizer.decode([tid]) for tid in full_token_ids]

        # Extract TAM scores for each generated token at each layer.
        # For generated token gen_idx, the target class is the actual token that
        # was generated. We extract the logit for that class at every position
        # from rounds 0..gen_idx, concatenated.
        print(f"[TAMAnalyzer] Extracting TAM scores for {num_generated} tokens "
              f"x {num_layers} layers...")

        raw_scores: List[List[np.ndarray]] = []

        for gen_idx in range(num_generated):
            cls_id = full_token_ids[input_len + gen_idx]
            token_scores: List[np.ndarray] = []

            for layer_idx in range(num_layers):
                parts = []
                for step in range(gen_idx + 1):
                    part = per_layer_logits[layer_idx][step][0, :, cls_id]
                    parts.append(part.cpu().float())
                scores = torch.cat(parts, dim=0).clamp(min=0).detach().numpy()
                token_scores.append(scores)

            raw_scores.append(token_scores)

        del per_layer_logits
        torch.cuda.empty_cache()

        print(f"[TAMAnalyzer] TAM extraction complete.")

        return TAMResult(
            raw_scores=raw_scores,
            full_token_ids=full_token_ids,
            token_strings=token_strings,
            token_regions=token_regions,
            num_layers=num_layers,
            num_generated=num_generated,
            vision_shape=vision_shape,
            input_len=input_len,
        )

    # ------------------------------------------------------------------
    # Per-token activation (Mode A: Deep Inspection)
    # ------------------------------------------------------------------

    def get_activation_for_token(
        self, result: TAMResult, gen_idx: int
    ) -> Dict[str, Any]:
        """For generated token gen_idx, extract per-layer TAM activations.

        Returns normalized (min-max) activation scores at each layer and
        their layer-average.

        Args:
            result: TAMResult from extract().
            gen_idx: Index within generated tokens (0 = first generated).

        Returns:
            dict with:
                per_layer: (L, score_len) normalized scores at each layer
                avg_full:  (score_len,)   layer-averaged normalized scores
        """
        num_layers = result.num_layers
        per_layer = []

        for layer_idx in range(num_layers):
            scores = result.raw_scores[gen_idx][layer_idx].copy()
            s_min, s_max = scores.min(), scores.max()
            if s_max - s_min > 1e-8:
                scores_norm = (scores - s_min) / (s_max - s_min)
            else:
                scores_norm = np.zeros_like(scores)
            per_layer.append(scores_norm)

        per_layer_arr = np.array(per_layer)  # (L, score_len)
        avg_full = per_layer_arr.mean(axis=0)  # (score_len,)

        return {
            "per_layer": per_layer_arr,
            "avg_full": avg_full,
        }

    # ------------------------------------------------------------------
    # Generation summary (Mode B)
    # ------------------------------------------------------------------

    def compute_generation_summary(
        self, result: TAMResult
    ) -> List[Dict[str, Any]]:
        """For every generated token, compute a compact activation summary.

        Returns a list of dicts (one per generated token) containing:
            token_idx, token_str, image/prompt/generated activation ratios,
            entropy, top_k_tokens, phase, and optionally thinking/answer ratios.
        """
        regions = result.token_regions
        summary: List[Dict[str, Any]] = []

        for gen_idx in range(result.num_generated):
            abs_idx = result.input_len + gen_idx

            # Average raw scores across all layers
            layer_scores = [
                result.raw_scores[gen_idx][l]
                for l in range(result.num_layers)
            ]
            avg_scores = np.mean(layer_scores, axis=0)
            score_len = len(avg_scores)

            # Region activation masses
            img_mass = _safe_sum(
                avg_scores, regions.image_start, regions.image_end
            )
            prompt_mass = _safe_sum(
                avg_scores, regions.prompt_start,
                min(regions.prompt_end, score_len),
            )
            gen_mass = (
                _safe_sum(
                    avg_scores, regions.generation_start,
                    min(regions.generation_start + gen_idx, score_len),
                )
                if gen_idx > 0
                else 0.0
            )
            region_total = img_mass + prompt_mass + gen_mass + 1e-12

            # Entropy of the normalized distribution
            total = avg_scores.sum() + 1e-12
            prob = avg_scores / total
            p = prob[prob > 0]
            entropy = float(-(p * np.log(p + 1e-12)).sum())

            # Top-5 most activated positions
            k = min(5, len(avg_scores))
            top_indices = np.argsort(avg_scores)[-k:][::-1]
            top_k = [
                (
                    int(idx),
                    float(avg_scores[idx]),
                    (result.token_strings[idx]
                     if idx < len(result.token_strings) else "?"),
                )
                for idx in top_indices
            ]

            phase = _get_phase(abs_idx, regions)

            entry: Dict[str, Any] = {
                "token_idx": abs_idx,
                "gen_idx": gen_idx,
                "token_str": (
                    result.token_strings[abs_idx]
                    if abs_idx < len(result.token_strings)
                    else ""
                ),
                "image_activation_ratio": img_mass / region_total,
                "prompt_activation_ratio": prompt_mass / region_total,
                "generated_activation_ratio": gen_mass / region_total,
                "entropy": entropy,
                "top_k_tokens": top_k,
                "phase": phase,
            }

            # Thinking model breakdown
            if regions.think_start is not None:
                ts = regions.think_start
                te = (
                    regions.think_end
                    if regions.think_end is not None
                    else min(abs_idx, score_len)
                )
                think_end_eff = min(te + 1, score_len)
                think_mass = (
                    _safe_sum(avg_scores, ts, think_end_eff)
                    if ts < score_len
                    else 0.0
                )
                ans_start = regions.answer_start or regions.generation_start
                ans_end = min(regions.generation_start + gen_idx, score_len)
                ans_mass = (
                    _safe_sum(avg_scores, ans_start, ans_end)
                    if ans_start < score_len
                    else 0.0
                )
                entry["thinking_activation_ratio"] = think_mass / region_total
                entry["answer_activation_ratio"] = ans_mass / region_total

            summary.append(entry)

        return summary

    # ------------------------------------------------------------------
    # Quantitative metrics
    # ------------------------------------------------------------------

    def compute_metrics(
        self, result: TAMResult
    ) -> Dict[str, np.ndarray]:
        """Compute per-layer aggregate activation metrics.

        Returns dict with:
            entropy_per_layer:                    (L,)
            image_activation_ratio_per_layer:     (L,)
            sparsity_per_layer:                   (L,)
        """
        regions = result.token_regions
        num_layers = result.num_layers
        metrics: Dict[str, np.ndarray] = {}

        if result.num_generated == 0:
            return metrics

        entropy_per_layer = np.zeros(num_layers)
        image_ratio_per_layer = np.zeros(num_layers)
        sparsity_per_layer = np.zeros(num_layers)

        img_s, img_e = regions.image_start, regions.image_end

        for layer_idx in range(num_layers):
            entropies = []
            img_ratios = []
            sparsities = []

            for gen_idx in range(result.num_generated):
                scores = result.raw_scores[gen_idx][layer_idx]
                total = scores.sum() + 1e-12
                prob = scores / total

                # Shannon entropy
                p = prob[prob > 0]
                ent = -(p * np.log(p + 1e-12)).sum()
                entropies.append(ent)

                # Image activation ratio
                if img_e > img_s:
                    img_mass = scores[img_s:min(img_e, len(scores))].sum()
                    img_ratios.append(img_mass / total)

                # Sparsity (fraction of positions above uniform threshold)
                threshold = 1.0 / max(len(scores), 1)
                sparsities.append(float((prob > threshold).mean()))

            entropy_per_layer[layer_idx] = np.mean(entropies)
            if img_ratios:
                image_ratio_per_layer[layer_idx] = np.mean(img_ratios)
            sparsity_per_layer[layer_idx] = np.mean(sparsities)

        metrics["entropy_per_layer"] = entropy_per_layer
        if img_e > img_s:
            metrics["image_activation_ratio_per_layer"] = image_ratio_per_layer
        metrics["sparsity_per_layer"] = sparsity_per_layer

        return metrics


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _safe_sum(arr: np.ndarray, start: int, end: int) -> float:
    """Sum an array slice with bounds checking."""
    start = max(start, 0)
    end = min(end, len(arr))
    if start >= end:
        return 0.0
    return float(arr[start:end].sum())


def _get_phase(idx: int, regions: TokenRegions) -> str:
    """Determine which generation phase a token belongs to."""
    if regions.think_start is not None and regions.think_end is not None:
        if regions.think_start <= idx <= regions.think_end:
            return "thinking"
        if regions.answer_start is not None and idx >= regions.answer_start:
            return "answer"
    return "generated"
