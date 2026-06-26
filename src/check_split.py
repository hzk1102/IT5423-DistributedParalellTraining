"""Correctness self-check for the model decomposition (model_split.py).

Runs on a SINGLE process (CPU is fine) and proves that pushing tokens through the
ordered stage list / partitioned stages reproduces the original HF model's logits.
ALWAYS run this for your model before trusting a multi-GPU pipeline run:

    python src/check_split.py --model gpt2 --num-stages 2

A PASS means the embedding/block/head split and the causal masking are wired
correctly; only then are throughput numbers from the pipeline meaningful.
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM

from model_split import build_ordered_layers, build_stage_modules


def _forward_chain(layers, x):
    for m in layers:
        x = m(x)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", help="use a small one (gpt2/distilgpt2) for the check")
    ap.add_argument("--num-stages", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--balance", default="uniform", choices=["uniform", "params"])
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-4)
    args = ap.parse_args()

    torch.manual_seed(0)
    # fp32 + eager so the comparison isn't masked by fp16 rounding.
    ref = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    vocab = ref.config.vocab_size
    input_ids = torch.randint(0, vocab, (args.batch, args.seq_len))

    with torch.no_grad():
        ref_logits = ref(input_ids).logits

        layers, _ = build_ordered_layers(args.model, attn_implementation="eager", dtype="float32")
        flat_logits = _forward_chain(layers, input_ids)

        stages, _, ranges = build_stage_modules(
            args.model, args.num_stages, attn_implementation="eager",
            dtype="float32", balance=args.balance,
        )
        staged_logits = _forward_chain(stages, input_ids)

    def report(name, a, b):
        max_abs = (a - b).abs().max().item()
        ok = torch.allclose(a, b, atol=args.atol, rtol=args.rtol)
        print(f"  {name:28s} max|Δ|={max_abs:.3e}  -> {'PASS' if ok else 'FAIL'}")
        return ok

    print(f"Model={args.model}  num_stages={args.num_stages}  ranges={ranges}")
    n_params = sum(p.numel() for m in layers for p in m.parameters())
    print(f"Ordered layers: {len(layers)}   total params: {n_params/1e6:.1f}M")
    ok1 = report("flat-chain vs HF model", flat_logits, ref_logits)
    ok2 = report("partitioned vs HF model", staged_logits, ref_logits)

    if ok1 and ok2:
        print("\nALL CHECKS PASSED — the split is faithful; safe to run the pipeline.")
    else:
        raise SystemExit("\nCHECK FAILED — do NOT trust pipeline results until this passes.")


if __name__ == "__main__":
    main()
