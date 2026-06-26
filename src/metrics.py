"""Instrumentation: throughput, peak VRAM, pipeline-bubble theory, MFU, CSV log.

These are the graded artifacts of the study (PROJECT_PLAN.md §6) — capture them,
not just a working loop.
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

# T4 (Turing) peak fp16 matmul throughput, per NVIDIA datasheet: 65 TFLOP/s.
T4_PEAK_FP16_FLOPS = 65e12


# --------------------------------------------------------------------------- #
# Throughput                                                                   #
# --------------------------------------------------------------------------- #
class ThroughputMeter:
    """Measures steady-state throughput, dropping the first `warmup` steps so
    one-off CUDA/cuDNN autotuning and allocator warm-up don't pollute tokens/s."""

    def __init__(self, warmup: int = 3):
        self.warmup = warmup
        self.reset()

    def reset(self):
        self._steps = 0
        self._tokens = 0
        self._samples = 0
        self._t0 = None
        self._elapsed = 0.0

    def step(self, n_tokens: int, n_samples: int):
        self._steps += 1
        if self._steps <= self.warmup:
            # Still warming up; (re)start the clock after the last warmup step.
            self._t0 = time.perf_counter()
            return
        self._tokens += n_tokens
        self._samples += n_samples
        self._elapsed = time.perf_counter() - self._t0

    @property
    def tokens_per_sec(self) -> float:
        return self._tokens / self._elapsed if self._elapsed > 0 else 0.0

    @property
    def samples_per_sec(self) -> float:
        return self._samples / self._elapsed if self._elapsed > 0 else 0.0

    @property
    def measured_steps(self) -> int:
        return max(0, self._steps - self.warmup)


# --------------------------------------------------------------------------- #
# Memory                                                                       #
# --------------------------------------------------------------------------- #
def reset_peak_memory():
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb(device=None) -> float:
    import torch
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 3)


def all_reduce_max_gb(value_gb: float) -> float:
    """Max peak-memory across ranks (so we report the worst/heaviest stage)."""
    import torch
    import torch.distributed as dist
    if not dist.is_initialized():
        return value_gb
    t = torch.tensor([value_gb], device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


# --------------------------------------------------------------------------- #
# Pipeline bubble (theory) — PROJECT_PLAN.md §7                                #
# --------------------------------------------------------------------------- #
def bubble_fraction(num_stages: int, num_micro_batches: int) -> float:
    """Idle fraction of a synchronous (GPipe/1F1B) pipeline:
        (P - 1) / (M + P - 1).
    With P=2 this is 1/(M+1): 50% at M=1, ~6% at M=16."""
    P, M = num_stages, num_micro_batches
    return (P - 1) / (M + P - 1)


def interleaved_bubble_fraction(num_stages: int, num_micro_batches: int, v: int) -> float:
    """Interleaved 1F1B with `v` virtual stages per device cuts the bubble ~1/v."""
    return bubble_fraction(num_stages, num_micro_batches) / max(1, v)


# --------------------------------------------------------------------------- #
# Model FLOPs Utilisation (MFU) — nanoGPT / PaLM estimator                     #
# --------------------------------------------------------------------------- #
def flops_per_token(n_params: int, n_layer: int, n_head: int, head_dim: int, seq_len: int) -> float:
    """Fwd+bwd FLOPs per token: 6N (dense matmuls) + 12*L*H*Q*T (attention).

    N counts all params (the embedding term is a small over-estimate but standard
    and consistent across our three systems, which is what matters for comparison).
    """
    return 6 * n_params + 12 * n_layer * n_head * head_dim * seq_len


def mfu(tokens_per_sec: float, n_params: int, n_layer: int, n_head: int,
        head_dim: int, seq_len: int, world_size: int,
        peak_flops: float = T4_PEAK_FP16_FLOPS) -> float:
    """Achieved FLOP/s ÷ aggregate device peak FLOP/s, in [0, 1]."""
    achieved = tokens_per_sec * flops_per_token(n_params, n_layer, n_head, head_dim, seq_len)
    promised = peak_flops * world_size
    return achieved / promised if promised > 0 else 0.0


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def model_dims(hf_config):
    """Pull (n_layer, n_head, head_dim) from a GPT-2 or OPT HF config."""
    n_layer = getattr(hf_config, "n_layer", None) or getattr(hf_config, "num_hidden_layers", 0)
    n_head = getattr(hf_config, "n_head", None) or getattr(hf_config, "num_attention_heads", 0)
    hidden = getattr(hf_config, "n_embd", None) or getattr(hf_config, "hidden_size", 0)
    head_dim = hidden // n_head if n_head else 0
    return int(n_layer), int(n_head), int(head_dim)


# --------------------------------------------------------------------------- #
# Result logging                                                              #
# --------------------------------------------------------------------------- #
RESULT_FIELDS = [
    "timestamp", "system", "model", "dataset_config", "schedule",
    "world_size", "num_stages", "micro_batch_size", "num_micro_batches",
    "global_batch", "seq_len", "optim", "grad_checkpointing", "attn",
    "measured_steps", "tokens_per_sec", "samples_per_sec",
    "peak_mem_gb_rank0", "peak_mem_gb_max",
    "bubble_theory", "mfu", "final_loss", "eval_ppl",
    "torch_version", "notes",
]


@dataclass
class RunResult:
    system: str
    model: str
    dataset_config: str = ""
    schedule: str = ""
    world_size: int = 1
    num_stages: int = 1
    micro_batch_size: int = 0
    num_micro_batches: int = 1
    global_batch: int = 0
    seq_len: int = 0
    optim: str = ""
    grad_checkpointing: bool = False
    attn: str = ""
    measured_steps: int = 0
    tokens_per_sec: float = 0.0
    samples_per_sec: float = 0.0
    peak_mem_gb_rank0: float = 0.0
    peak_mem_gb_max: float = 0.0
    bubble_theory: float = 0.0
    mfu: float = 0.0
    final_loss: float = 0.0
    eval_ppl: float = 0.0
    torch_version: str = ""
    notes: str = ""
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def to_row(self) -> dict:
        return {k: getattr(self, k) for k in RESULT_FIELDS}


def log_result(result: RunResult, csv_path: str = "results/results.csv") -> None:
    """Append one run as a row to results.csv (creating header if needed) and
    drop a per-run JSON snapshot next to it. Call only on rank 0."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(result.to_row())

    json_dir = os.path.join(os.path.dirname(csv_path) or ".", "runs")
    os.makedirs(json_dir, exist_ok=True)
    tag = f"{result.system}_{result.schedule or 'na'}_M{result.num_micro_batches}_{result.timestamp.replace(':', '').replace(' ', '_')}"
    with open(os.path.join(json_dir, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)


# --------------------------------------------------------------------------- #
# Per-step loss logging (for convergence-curve figures)                       #
# --------------------------------------------------------------------------- #
class LossLogger:
    """Append (system, schedule, model, M, step, loss, tok/s) rows to a CSV so
    the report can plot loss-vs-step convergence curves across the three systems.

    A no-op when `path` is falsy, so trainers can always construct one and call
    `.log(...)` unconditionally. Only the rank that owns the loss (rank 0 /
    last pipeline stage) should hold a logger; others pass path=None.
    """

    def __init__(self, path, system="", schedule="", model="", num_micro_batches=0):
        self.path = path
        self._f = None
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        new = not os.path.exists(path)
        self._f = open(path, "a", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        if new:
            self._w.writerow(["system", "schedule", "model", "num_micro_batches",
                              "step", "loss", "tokens_per_sec"])
        self._meta = [system, schedule, model, num_micro_batches]

    def log(self, step: int, loss: float, tokens_per_sec: float = 0.0):
        if self._f is None:
            return
        self._w.writerow(self._meta + [step, f"{loss:.6f}", f"{tokens_per_sec:.1f}"])
        self._f.flush()

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None
