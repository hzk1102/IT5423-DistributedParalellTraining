"""Single-GPU baseline: gpt2-xl on ONE T4 via gradient checkpointing + a
memory-efficient optimizer (PROJECT_PLAN.md §4C, deliverable 4 control).

This isolates "what does adding the 2nd GPU via pipelining actually buy us." It
shares the dataset, tokenizer, seq_len, global batch, seed, fp16 setting and
optimizer-step count with the pipeline paths so the comparison is apples-to-apples.

Run (single process, one GPU):
    python src/train_singlegpu.py --model gpt2-xl --grad-checkpointing \
        --optim adamw8bit --micro-batch-size 1 --num-micro-batches 8 \
        --seq-len 512 --max-steps 50 --eval
"""
from __future__ import annotations

import argparse
import math

import torch
from transformers import AutoModelForCausalLM

from data import build_lm_dataset, get_tokenizer, make_dataloader
from metrics import (LossLogger, RunResult, ThroughputMeter, count_params,
                     log_result, mfu, model_dims, peak_memory_gb,
                     reset_peak_memory)
from optim import build_optimizer
from utils import set_seed


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 50) -> float:
    model.eval()
    losses = []
    for i, (input_ids, labels) in enumerate(loader):
        if i >= max_batches:
            break
        input_ids, labels = input_ids.to(device), labels.to(device)
        out = model(input_ids=input_ids, labels=labels)
        losses.append(out.loss.float().item())
    model.train()
    mean = sum(losses) / max(1, len(losses))
    return float(math.exp(min(20.0, mean)))  # clamp to avoid inf on short runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-xl")
    ap.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--micro-batch-size", type=int, default=1)
    ap.add_argument("--num-micro-batches", type=int, default=8,
                    help="grad-accum steps; global batch = micro_bs * this (matches pipeline M)")
    ap.add_argument("--max-steps", type=int, default=50, help="optimizer steps")
    ap.add_argument("--warmup-steps", type=int, default=3, help="steps dropped from throughput")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--optim", default="adamw8bit", choices=["adamw", "adamw8bit", "adafactor"])
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--loss-scale", type=float, default=1.0,
                    help="static fp16 loss scale (convergence runs); 1.0 = off")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--eval-config", default=None, help="defaults to --dataset-config")
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--results-csv", default="results/results.csv")
    ap.add_argument("--loss-log", default="", help="optional CSV for per-step loss (convergence curves)")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = get_tokenizer(args.model)
    train_ds = build_lm_dataset(args.model, args.dataset_config, "train",
                                args.seq_len, tokenizer, max_samples=args.max_train_samples)
    global_batch = args.micro_batch_size * args.num_micro_batches
    train_loader = make_dataloader(train_ds, global_batch, shuffle=True, seed=args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation=args.attn
    ).to(device)
    model.config.use_cache = False
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    optimizer = build_optimizer(model.parameters(), args.optim, args.lr)
    # The model is fully fp16, so torch's GradScaler (which assumes fp32 master
    # weights) refuses to unscale the fp16 grads. Use the same plain-fp16 +
    # optional static loss scaling path as train_torchpp.py instead.
    scale = args.loss_scale

    n_params = count_params(model)
    n_layer, n_head, head_dim = model_dims(model.config)
    meter = ThroughputMeter(warmup=args.warmup_steps)
    reset_peak_memory()

    print(f"[singlegpu] {args.model} params={n_params/1e9:.2f}B  global_batch={global_batch}"
          f"  grad_ckpt={args.grad_checkpointing}  optim={args.optim}", flush=True)

    loss_log = LossLogger(args.loss_log, system="singlegpu",
                          schedule="grad_ckpt" if args.grad_checkpointing else "dense",
                          model=args.model, num_micro_batches=args.num_micro_batches)
    step, last_loss = 0, 0.0
    data_iter = iter(train_loader)
    while step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.num_micro_batches):
            try:
                input_ids, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                input_ids, labels = next(data_iter)
            input_ids, labels = input_ids.to(device), labels.to(device)
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss / args.num_micro_batches * scale
            loss.backward()
            step_loss += out.loss.item()
        if scale != 1.0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(scale)
        optimizer.step()
        step += 1
        last_loss = step_loss / args.num_micro_batches
        meter.step(n_tokens=global_batch * args.seq_len, n_samples=global_batch)
        loss_log.log(step, last_loss, meter.tokens_per_sec)
        if step % 5 == 0 or step == 1:
            print(f"  step {step:4d}/{args.max_steps}  loss={last_loss:.4f}"
                  f"  tok/s={meter.tokens_per_sec:,.0f}", flush=True)
    loss_log.close()

    peak0 = peak_memory_gb(device)
    eval_ppl = 0.0
    if args.eval:
        eval_ds = build_lm_dataset(args.model, args.eval_config or args.dataset_config,
                                   "validation", args.seq_len, tokenizer)
        eval_loader = make_dataloader(eval_ds, global_batch, shuffle=False, drop_last=True)
        eval_ppl = evaluate(model, eval_loader, device)
        print(f"[singlegpu] eval perplexity = {eval_ppl:.2f}", flush=True)

    result = RunResult(
        system="singlegpu", model=args.model, dataset_config=args.dataset_config,
        schedule="grad_ckpt" if args.grad_checkpointing else "dense",
        world_size=1, num_stages=1, micro_batch_size=args.micro_batch_size,
        num_micro_batches=args.num_micro_batches, global_batch=global_batch,
        seq_len=args.seq_len, optim=args.optim, grad_checkpointing=args.grad_checkpointing,
        attn=args.attn, measured_steps=meter.measured_steps,
        tokens_per_sec=meter.tokens_per_sec, samples_per_sec=meter.samples_per_sec,
        peak_mem_gb_rank0=peak0, peak_mem_gb_max=peak0,
        bubble_theory=0.0,
        mfu=mfu(meter.tokens_per_sec, n_params, n_layer, n_head, head_dim, args.seq_len, world_size=1),
        final_loss=last_loss, eval_ppl=eval_ppl,
        torch_version=torch.__version__, notes=args.notes,
    )
    log_result(result, args.results_csv)
    print(f"[singlegpu] tok/s={result.tokens_per_sec:,.0f}  peak_mem={peak0:.2f}GB"
          f"  MFU={result.mfu*100:.1f}%  -> logged to {args.results_csv}", flush=True)


if __name__ == "__main__":
    main()
