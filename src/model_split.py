"""Decompose an HF causal-LM into an ordered list of single-tensor-in/out stages.

This is the shared backbone for BOTH pipeline frameworks (PROJECT_PLAN.md §4):
  - DeepSpeed wants a flat list of layers -> `build_ordered_layers(...)`.
  - torch.distributed.pipelining wants per-rank submodules -> `build_stage_modules(...)`.

Every stage passes only a single tensor to the next (`input_ids` -> hidden_states
-> ... -> logits). GPT-2 blocks build their own causal mask internally, so no
attention_mask needs to flow through the pipe — which keeps the plumbing simple.

Validated path: the **GPT-2 family** (`gpt2`, `gpt2-medium/large/xl`, `distilgpt2`).
Use `check_split.py` to prove a split reproduces the original model's logits before
trusting a multi-GPU run. OPT support is included but marked EXPERIMENTAL.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM


def _causal_additive_mask(hidden_states):
    """Build a (1, 1, T, T) additive causal mask (0 on/below diagonal, large
    negative above) in the hidden_states' dtype/device.

    Required because modern transformers (v5) build the causal mask in the top-
    level model and pass it DOWN to each block — a bare `attention_mask=None`
    call to a block would be **non-causal**. Passing this explicit mask is
    correct on both old (v4, internal bias buffer) and new (v5) transformers.
    """
    T = hidden_states.size(1)
    dtype = hidden_states.dtype
    min_val = torch.finfo(dtype).min
    mask = torch.full((T, T), min_val, device=hidden_states.device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)  # keep strictly-upper (future) = min; rest = 0
    return mask.view(1, 1, T, T)


# --------------------------------------------------------------------------- #
# GPT-2 stage wrappers                                                         #
# --------------------------------------------------------------------------- #
class GPT2EmbeddingStage(nn.Module):
    """input_ids (B,T) long -> hidden_states (B,T,H)."""

    def __init__(self, wte, wpe, drop):
        super().__init__()
        self.wte, self.wpe, self.drop = wte, wpe, drop

    def forward(self, input_ids):
        input_ids = input_ids.long()
        _, seqlen = input_ids.shape
        pos = torch.arange(seqlen, device=input_ids.device).unsqueeze(0)
        h = self.wte(input_ids) + self.wpe(pos)
        return self.drop(h)


class GPT2BlockStage(nn.Module):
    """hidden_states -> hidden_states for one transformer block. We pass an
    explicit causal mask (see `_causal_additive_mask`) so the block attends
    causally even when called outside the full model."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden_states):
        mask = _causal_additive_mask(hidden_states)
        out = self.block(hidden_states, attention_mask=mask)
        return out[0] if isinstance(out, tuple) else out


class GPT2HeadStage(nn.Module):
    """hidden_states -> logits (B,T,V). lm_head is *untied* from wte (cloned) so
    it can live on a different pipeline stage/device than the embedding."""

    def __init__(self, ln_f, lm_head):
        super().__init__()
        self.ln_f, self.lm_head = ln_f, lm_head

    def forward(self, hidden_states):
        return self.lm_head(self.ln_f(hidden_states))


def _gpt2_ordered_layers(model) -> List[nn.Module]:
    t = model.transformer
    layers: List[nn.Module] = [GPT2EmbeddingStage(t.wte, t.wpe, t.drop)]
    layers += [GPT2BlockStage(b) for b in t.h]
    # Untie lm_head from wte so embedding (stage 0) and head (last stage) are
    # independent params on separate devices.
    head = nn.Linear(model.lm_head.in_features, model.lm_head.out_features, bias=False)
    head.weight = nn.Parameter(model.lm_head.weight.detach().clone())
    layers.append(GPT2HeadStage(t.ln_f, head))
    return layers


# --------------------------------------------------------------------------- #
# OPT stage wrappers  (EXPERIMENTAL — validate with check_split.py first)      #
# --------------------------------------------------------------------------- #
class OPTEmbeddingStage(nn.Module):
    def __init__(self, embed_tokens, embed_positions, project_in):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.embed_positions = embed_positions
        self.project_in = project_in

    def forward(self, input_ids):
        input_ids = input_ids.long()
        h = self.embed_tokens(input_ids)
        attn = torch.ones_like(input_ids)
        # OPTLearnedPositionalEmbedding takes the attention mask and derives
        # absolute positions (with its internal +2 offset).
        pos = self.embed_positions(attn, 0)
        h = h + pos
        if self.project_in is not None:
            h = self.project_in(h)
        return h


class OPTBlockStage(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states):
        mask = _causal_additive_mask(hidden_states)
        out = self.layer(hidden_states, attention_mask=mask)
        return out[0] if isinstance(out, tuple) else out


class OPTHeadStage(nn.Module):
    def __init__(self, final_layer_norm, project_out, lm_head):
        super().__init__()
        self.final_layer_norm = final_layer_norm
        self.project_out = project_out
        self.lm_head = lm_head

    def forward(self, hidden_states):
        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
        return self.lm_head(hidden_states)


def _opt_ordered_layers(model) -> List[nn.Module]:
    dec = model.model.decoder
    project_in = getattr(dec, "project_in", None)
    project_out = getattr(dec, "project_out", None)
    final_ln = getattr(dec, "final_layer_norm", None)
    layers: List[nn.Module] = [OPTEmbeddingStage(dec.embed_tokens, dec.embed_positions, project_in)]
    layers += [OPTBlockStage(l) for l in dec.layers]
    head = nn.Linear(model.lm_head.in_features, model.lm_head.out_features, bias=False)
    head.weight = nn.Parameter(model.lm_head.weight.detach().clone())
    layers.append(OPTHeadStage(final_ln, project_out, head))
    return layers


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def build_ordered_layers(
    model_name: str,
    attn_implementation: str = "eager",
    dtype: str = "float16",
) -> Tuple[List[nn.Module], object]:
    """Build the model on CPU and return (ordered single-tensor stage list, config).

    `attn_implementation="eager"` is the default for the split because GPT-2's
    eager path always applies the registered causal-mask buffer regardless of
    attention_mask — the most predictable behaviour for a manual decomposition.
    Try `"sdpa"` to save activation memory once `check_split.py` passes.
    """
    torch_dtype = getattr(torch, dtype)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, attn_implementation=attn_implementation
    )
    model.eval()
    mt = config.model_type
    if mt == "gpt2":
        layers = _gpt2_ordered_layers(model)
    elif mt == "opt":
        layers = _opt_ordered_layers(model)
    else:
        raise ValueError(
            f"Unsupported model_type={mt!r}. Validated: GPT-2 family; experimental: OPT."
        )
    return layers, config


def partition_indices(n_layers: int, num_stages: int, param_counts=None,
                      balance: str = "uniform") -> List[Tuple[int, int]]:
    """Return `num_stages` contiguous [start, end) index ranges over the layer list.

    balance="uniform": even by layer count.
    balance="params" : greedy so each stage holds ~equal parameters (embedding +
                       lm_head are heavy, so this matters — pipelines run at the
                       speed of the slowest/heaviest stage; PROJECT_PLAN.md §5).
    """
    assert 1 <= num_stages <= n_layers, f"num_stages={num_stages} vs n_layers={n_layers}"
    if balance == "params" and param_counts is not None:
        total = sum(param_counts)
        target = total / num_stages
        ranges, start, acc, stages_left = [], 0, 0, num_stages
        for i, c in enumerate(param_counts):
            acc += c
            remaining_layers = n_layers - (i + 1)
            # Close this stage when it has ~target params, leaving enough layers
            # for the remaining stages.
            if (acc >= target and stages_left > 1
                    and remaining_layers >= stages_left - 1):
                ranges.append((start, i + 1))
                start, acc, stages_left = i + 1, 0, stages_left - 1
        ranges.append((start, n_layers))
        return ranges
    # uniform
    base, extra = divmod(n_layers, num_stages)
    ranges, start = [], 0
    for s in range(num_stages):
        end = start + base + (1 if s < extra else 0)
        ranges.append((start, end))
        start = end
    return ranges


def build_stage_modules(
    model_name: str,
    num_stages: int,
    attn_implementation: str = "eager",
    dtype: str = "float16",
    balance: str = "uniform",
) -> Tuple[List[nn.Module], object, List[Tuple[int, int]]]:
    """Return (list of `num_stages` nn.Sequential modules, config, index ranges).

    Each Sequential is one pipeline stage (single-tensor in/out). The caller keeps
    only the stage(s) for its rank and moves them to GPU; the rest can be freed.
    """
    layers, config = build_ordered_layers(model_name, attn_implementation, dtype)
    pcounts = [sum(p.numel() for p in m.parameters()) for m in layers]
    ranges = partition_indices(len(layers), num_stages, pcounts, balance)
    stages = [nn.Sequential(*layers[a:b]) for (a, b) in ranges]
    return stages, config, ranges
