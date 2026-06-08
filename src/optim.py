"""Shared optimizer factory.

For a fair convergence comparison, use the SAME `--optim` across all three
systems. `adamw8bit` (bitsandbytes) is the default because full fp32 AdamW
states don't fit a T4 for gpt2-xl; `adafactor` is the lightest fallback if you
still hit OOM. See PROJECT_PLAN.md §3/§4.
"""
from __future__ import annotations

import torch


def build_optimizer(params, name: str, lr: float):
    """`params` is an iterable of parameters (or param groups)."""
    params = list(params)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if name == "adamw8bit":
        import bitsandbytes as bnb  # int8 optimizer states -> ~4x lighter than fp32 Adam
        return bnb.optim.Adam8bit(params, lr=lr)
    if name == "adafactor":
        from transformers.optimization import Adafactor
        return Adafactor(params, lr=lr, scale_parameter=False, relative_step=False)
    raise ValueError(f"unknown optim {name!r}")
