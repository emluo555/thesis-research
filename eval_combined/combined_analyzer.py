"""
combined_analyzer.py - Single-pass extraction of both attention weights and TAM scores.

Runs one model.generate() call with output_attentions=True AND output_hidden_states=True,
then reconstructs:
    - Attention matrices (L, H, S, S) in fp16
    - TAM scores per generated token per layer (ReLU-clipped logit for predicted class)

Memory-optimised post-processing:
    1. Reconstruct attention matrices, move to CPU, delete raw attentions.
    2. Project hidden states through lm_head ONE LAYER AT A TIME, extract target-class
       logit, immediately discard the full vocab-sized tensor.
    3. Delete raw hidden states.

Requires the model to be loaded with attn_implementation='eager'.
"""

import gc
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class TokenRegions:
    """Index ranges for different token regions in the full sequence."""
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


@dataclass
class CombinedResult:
    """Everything returned by a single-pass extraction."""
    # Attention
    attention_matrices: torch.Tensor       # (L, H, S, S) fp16, on CPU
    # TAM
    tam_scores: List[List[np.ndarray]]     # tam_scores[gen_idx][layer_idx] = (seq_len,)
    # Shared metadata
    full_token_ids: List[int]
    token_strings: List[str]
    token_regions: TokenRegions
    num_layers: int
    num_heads: int
    num_generated: int
    input_len: int
    vision_shape: Optional[Tuple[int, ...]] = None
    selected_layers: Optional[List[int]] = None


# ------------------------------------------------------------------
# Analyzer
# ------------------------------------------------------------------

class CombinedAnalyzer:
    """Single-pass extraction of attention weights and TAM activation scores."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

        text_config = model.config.text_config
        self.num_layers = text_config.num_hidden_layers
        self.num_heads = text_config.num_attention_heads

        self.vision_start_id = model.config.vision_start_token_id
        self.vision_end_id = model.config.vision_end_token_id

        self.think_start_id = self._get_token_id("<think>")
        self.think_end_id = self._get_token_id("</think>")

    # ------------------------------------------------------------------
    # Token ID helpers
    # ------------------------------------------------------------------

    def _get_token_id(self, token_str: str) -> Optional[int]:
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
        self, token_ids: List[int], input_len: int,
    ) -> TokenRegions:
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
    # Single-pass extraction
    # ------------------------------------------------------------------

    def extract(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int = 256,
        selected_layers: Optional[List[int]] = None,
        do_sample: bool = False,
    ) -> CombinedResult:
        """Run model.generate() once, return both attention and TAM data.

        Post-processing is sequenced for minimal peak GPU memory:
            1. Reconstruct attention -> move to CPU -> delete raw attentions.
            2. Extract TAM one layer at a time -> delete raw hidden states.
        """
        input_len = inputs["input_ids"].shape[1]
        vision_shape = self._infer_vision_shape(inputs)

        print(f"[CombinedAnalyzer] Generating (input_len={input_len}, "
              f"max_new_tokens={max_new_tokens})...")

        generate_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_attentions=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            use_cache=True,
            do_sample=do_sample,
        )

        with torch.no_grad():
            outputs = self.model.generate(**generate_kwargs)

        full_token_ids = outputs.sequences[0].cpu().tolist()
        num_generated = len(full_token_ids) - input_len

        print(f"[CombinedAnalyzer] Generated {num_generated} tokens. "
              f"Reconstructing attention...")

        # --- Phase 1: Attention reconstruction -> CPU -----------------
        attention_matrices = self._reconstruct_attention(
            outputs.attentions, input_len, selected_layers,
        )
        attention_matrices = attention_matrices.cpu()

        # Free raw attentions
        del outputs.attentions
        gc.collect()
        torch.cuda.empty_cache()

        # --- Phase 2: TAM extraction (layer-by-layer) ----------------
        print(f"[CombinedAnalyzer] Extracting TAM scores "
              f"({num_generated} tokens)...")

        raw_hidden_states = outputs.hidden_states
        num_steps = len(raw_hidden_states)
        num_all_layers = len(raw_hidden_states[0])

        layers_for_tam = (
            selected_layers if selected_layers is not None
            else list(range(num_all_layers))
        )

        tam_scores = self._extract_tam_scores(
            raw_hidden_states, full_token_ids, input_len,
            num_generated, layers_for_tam,
        )

        # Free raw hidden states and remaining outputs
        del raw_hidden_states, outputs
        gc.collect()
        torch.cuda.empty_cache()

        # --- Metadata ------------------------------------------------
        token_regions = self.get_token_regions(full_token_ids, input_len)
        tokenizer = self.processor.tokenizer
        token_strings = [tokenizer.decode([tid]) for tid in full_token_ids]

        print(f"[CombinedAnalyzer] Done. "
              f"Attention shape: {attention_matrices.shape}, "
              f"TAM: {num_generated} tokens x {len(layers_for_tam)} layers")

        return CombinedResult(
            attention_matrices=attention_matrices,
            tam_scores=tam_scores,
            full_token_ids=full_token_ids,
            token_strings=token_strings,
            token_regions=token_regions,
            num_layers=attention_matrices.shape[0],
            num_heads=attention_matrices.shape[1],
            num_generated=num_generated,
            input_len=input_len,
            vision_shape=vision_shape,
            selected_layers=selected_layers,
        )

    # ------------------------------------------------------------------
    # Attention reconstruction (from raw step-wise outputs)
    # ------------------------------------------------------------------

    def _reconstruct_attention(
        self,
        raw_attentions: Tuple,
        input_len: int,
        selected_layers: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Build full causal attention matrices (L, H, S, S) in fp16."""
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
            layer_attn = torch.zeros(
                H, total_len, total_len, dtype=torch.float16,
            )
            prefill = raw_attentions[0][l_idx][0]
            layer_attn[:, :input_len, :input_len] = prefill.cpu().half()

            for k in range(1, num_steps):
                row_idx = input_len + k - 1
                kv_len = raw_attentions[k][l_idx].shape[-1]
                decode_row = raw_attentions[k][l_idx][0, :, 0, :]
                layer_attn[:, row_idx, :kv_len] = decode_row.cpu().half()

            result_layers.append(layer_attn)

        return torch.stack(result_layers, dim=0)

    # ------------------------------------------------------------------
    # TAM extraction (memory-efficient, one layer at a time)
    # ------------------------------------------------------------------

    def _extract_tam_scores(
        self,
        hidden_states: Tuple,
        full_token_ids: List[int],
        input_len: int,
        num_generated: int,
        layers: List[int],
    ) -> List[List[np.ndarray]]:
        """Extract TAM scores without materialising all logits at once.

        Uses a layer-first loop so each (layer, step) pair is projected
        through lm_head exactly once.  For L layers and T steps this is
        L*T projections total, compared to L*sum(1..N) in the naive approach.

        Returns tam_scores[gen_idx][layer_local_idx] = numpy array (seq_len,).
        """
        num_steps = len(hidden_states)
        lm_head = self.model.lm_head

        # Initialise output: tam_scores[gen_idx][layer_local_idx]
        tam_scores: List[List[Optional[np.ndarray]]] = [
            [None] * len(layers) for _ in range(num_generated)
        ]

        for l_local, layer_idx in enumerate(layers):
            # Project every step for this layer once, keep on CPU
            step_class_scores: List[torch.Tensor] = []
            for step in range(num_steps):
                layer_hs = hidden_states[step][layer_idx]
                with torch.no_grad():
                    logits = lm_head(layer_hs)          # (1, seq_len_t, vocab)
                step_class_scores.append(logits[0].cpu().float())
                del logits

            # For each generated token, gather the target-class logit from
            # steps 0..gen_idx, concatenate, and ReLU-clip.
            for gen_idx in range(num_generated):
                cls_id = full_token_ids[input_len + gen_idx]
                parts = [
                    step_class_scores[s][:, cls_id]
                    for s in range(gen_idx + 1)
                ]
                scores = torch.cat(parts, dim=0).clamp(min=0).numpy()
                tam_scores[gen_idx][l_local] = scores

            del step_class_scores
            torch.cuda.empty_cache()

        return tam_scores

    # ------------------------------------------------------------------
    # Per-token ratio computation (attention)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_attention_ratios(result: CombinedResult) -> dict:
        """Per-token, per-layer attention ratios to 4 regions.

        Regions: system, image, prompt, generated (sums to 1.0).

        Returns dict with 'all_per_token_metrics' and
        'answer_token_offset'/'answer_token_count' (0-based offset into
        generated tokens list).
        """
        attn = result.attention_matrices.float()  # (L, H, S, S) on CPU
        regions = result.token_regions
        L, H, S, _ = attn.shape
        gen_start = regions.generation_start

        img_s, img_e = regions.image_start, regions.image_end
        prompt_s, prompt_e = regions.prompt_start, regions.prompt_end

        gen_attn = attn[:, :, gen_start:S, :]  # (L, H, gen_len, S)

        system_ratio = gen_attn[:, :, :, 0:img_s].sum(dim=-1).mean(dim=1)
        image_ratio = gen_attn[:, :, :, img_s:img_e].sum(dim=-1).mean(dim=1)
        prompt_ratio = gen_attn[:, :, :, prompt_s:prompt_e].sum(dim=-1).mean(dim=1)

        gen_len = S - gen_start
        generated_ratio = torch.zeros(L, gen_len)
        for i in range(gen_len):
            abs_i = gen_start + i
            if abs_i > gen_start:
                generated_ratio[:, i] = (
                    attn[:, :, abs_i, gen_start:abs_i]
                    .sum(dim=-1).mean(dim=1)
                )

        gen_tokens = result.token_strings[gen_start:S]

        all_metrics = {
            "system_attn_per_token": system_ratio.T.tolist(),
            "image_attn_per_token": image_ratio.T.tolist(),
            "prompt_attn_per_token": prompt_ratio.T.tolist(),
            "generated_attn_per_token": generated_ratio.T.tolist(),
            "generated_tokens": gen_tokens,
        }

        ans_start, ans_end = _find_answer_tag_span(
            result.token_strings, gen_start, S,
        )
        if ans_start is None:
            ans_start = (regions.answer_start
                         if regions.answer_start is not None else gen_start)
        if ans_end is None:
            ans_end = S
        ans_start = max(ans_start, gen_start)
        ans_end = min(ans_end, S)

        ans_offset = ans_start - gen_start
        ans_count = max(0, ans_end - ans_start)

        return {
            "all_per_token_metrics": all_metrics,
            "answer_token_offset": ans_offset,
            "answer_token_count": ans_count,
        }

    # ------------------------------------------------------------------
    # Per-token ratio computation (TAM)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_tam_ratios(result: CombinedResult) -> dict:
        """Per-token, per-layer TAM activation ratios to 4 regions.

        Regions: system, image, prompt, generated (sums to 1.0).
        Same structure as compute_attention_ratios for easy comparison.
        """
        regions = result.token_regions
        num_gen = result.num_generated
        num_layers = len(result.tam_scores[0]) if num_gen > 0 else 0
        gen_start = regions.generation_start
        S = gen_start + num_gen

        img_s, img_e = regions.image_start, regions.image_end
        prompt_s, prompt_e = regions.prompt_start, regions.prompt_end

        system_ratios_all = []    # (gen_len, num_layers)
        image_ratios_all = []
        prompt_ratios_all = []
        generated_ratios_all = []
        gen_tokens = []

        for gen_idx in range(num_gen):
            per_layer_sys = []
            per_layer_img = []
            per_layer_prompt = []
            per_layer_gen = []

            for l_idx in range(num_layers):
                scores = result.tam_scores[gen_idx][l_idx]
                total = float(scores.sum()) + 1e-12

                sys_mass = float(scores[0:img_s].sum())
                img_mass = float(scores[img_s:min(img_e, len(scores))].sum())
                prompt_mass = float(
                    scores[prompt_s:min(prompt_e, len(scores))].sum()
                )
                # Previously generated tokens: gen_start .. (gen_start + gen_idx - 1)
                gen_end_idx = min(gen_start + gen_idx, len(scores))
                gen_mass = (
                    float(scores[gen_start:gen_end_idx].sum())
                    if gen_idx > 0 else 0.0
                )

                per_layer_sys.append(sys_mass / total)
                per_layer_img.append(img_mass / total)
                per_layer_prompt.append(prompt_mass / total)
                per_layer_gen.append(gen_mass / total)

            system_ratios_all.append(per_layer_sys)
            image_ratios_all.append(per_layer_img)
            prompt_ratios_all.append(per_layer_prompt)
            generated_ratios_all.append(per_layer_gen)

            abs_idx = result.input_len + gen_idx
            gen_tokens.append(
                result.token_strings[abs_idx]
                if abs_idx < len(result.token_strings) else ""
            )

        all_metrics = {
            "system_tam_per_token": system_ratios_all,
            "image_tam_per_token": image_ratios_all,
            "prompt_tam_per_token": prompt_ratios_all,
            "generated_tam_per_token": generated_ratios_all,
            "generated_tokens": gen_tokens,
        }

        ans_start, ans_end = _find_answer_tag_span(
            result.token_strings, gen_start, S,
        )
        if ans_start is None:
            ans_start = (regions.answer_start
                         if regions.answer_start is not None else gen_start)
        if ans_end is None:
            ans_end = S
        ans_start = max(ans_start, gen_start)
        ans_end = min(ans_end, S)

        ans_offset = ans_start - gen_start
        ans_count = max(0, ans_end - ans_start)

        return {
            "all_per_token_metrics": all_metrics,
            "answer_token_offset": ans_offset,
            "answer_token_count": ans_count,
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _find_answer_tag_span(token_strings, gen_start, seq_len):
    """Scan generated tokens for <answer>...</answer>, return (start, end)."""
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


def extract_generated_answer(token_ids, token_regions, tokenizer):
    """Decode the generated text and split into thinking_trace and answer."""
    import re

    r = token_regions
    full_generated = tokenizer.decode(
        token_ids[r.generation_start:], skip_special_tokens=False,
    ).strip()

    answer_match = re.search(
        r"<answer>(.*?)</answer>", full_generated, re.DOTALL,
    )
    if answer_match:
        generated_answer = answer_match.group(1).strip()
        thinking_trace = full_generated[:answer_match.start()]
    else:
        think_match = re.search(r"</think>", full_generated)
        if think_match:
            thinking_trace = full_generated[:think_match.start()]
            generated_answer = full_generated[think_match.end():]
        else:
            thinking_trace = ""
            generated_answer = full_generated

    for tag in ["<think>", "</think>", "<answer>", "</answer>", "<|im_end|>"]:
        thinking_trace = thinking_trace.replace(tag, "")
        generated_answer = generated_answer.replace(tag, "")

    return generated_answer.strip(), thinking_trace.strip()
