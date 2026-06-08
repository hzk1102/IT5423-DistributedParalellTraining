"""Causal-LM loss shared by every training path.

The pipeline frameworks (torch.distributed.pipelining, DeepSpeed PipelineModule)
do not run HF's internal label-shift, so we shift here. The single-GPU baseline
also uses this so all three systems optimise the *identical* objective.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy with the standard shift-by-one.

    Args:
        logits: (B, T, V) float tensor from the final stage / lm_head.
        labels: (B, T) long tensor (the same token ids as the input; we shift here).

    Returns:
        Scalar mean cross-entropy over all (B*(T-1)) predicted positions.
    """
    # Predict token t+1 from position t. Upcast to fp32 for the softmax/CE so
    # fp16 pipeline activations don't make the loss over/underflow.
    logits = logits.float()
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
