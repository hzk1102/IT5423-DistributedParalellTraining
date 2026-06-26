"""PyTorch-native pipeline parallelism via **torch.distributed.pipelining**
(the modern PiPPy API; NOT the deprecated torch.distributed.pipeline).

This path benchmarks the full modern schedule ladder (PROJECT_PLAN.md §4B/§5/§7):
    gpipe -> 1f1b -> interleaved_1f1b -> interleaved_zb / zbv (zero-bubble)

Launch on 2 GPUs:
    torchrun --nproc_per_node=2 src/train_torchpp.py \
        --model gpt2-xl --schedule 1f1b --num-micro-batches 8 \
        --micro-batch-size 1 --seq-len 512 --max-steps 50

Notes
-----
- Each rank builds the model on CPU, keeps only its stage(s) on GPU, frees the rest.
- fp16 params; loss is computed in fp32 (see losses.py). For convergence runs add
  `--loss-scale 1024` (static loss scaling) for fp16 gradient stability — the
  scheduler runs backward internally so we can't use torch's GradScaler here.
- Zero-bubble schedules (`interleaved_zb`, `zbv`) require torch >= 2.6; if your
  Kaggle torch is older they simply won't be offered (the script will tell you).
"""
from __future__ import annotations

import argparse
import gc

import torch
import torch.distributed as dist

from data import build_lm_dataset, get_tokenizer, make_dataloader
from losses import causal_lm_loss
from metrics import (LossLogger, RunResult, ThroughputMeter, all_reduce_max_gb,
                     bubble_fraction, interleaved_bubble_fraction, log_result,
                     mfu, model_dims, peak_memory_gb, reset_peak_memory)
from model_split import build_stage_modules
from optim import build_optimizer
from utils import cleanup_distributed, set_seed, setup_distributed


def schedule_registry():
    """Map schedule name -> (layout_kind, class), only including those available
    in the installed torch version."""
    from torch.distributed.pipelining.schedules import ScheduleGPipe, Schedule1F1B
    reg = {"gpipe": ("single", ScheduleGPipe), "1f1b": ("single", Schedule1F1B)}
    for name, kind, attr in [
        ("interleaved_1f1b", "interleaved", "ScheduleInterleaved1F1B"),
        ("interleaved_zb", "interleaved", "ScheduleInterleavedZeroBubble"),
        ("zbv", "zbv", "ScheduleZBVZeroBubble"),
    ]:
        try:
            mod = __import__("torch.distributed.pipelining.schedules", fromlist=[attr])
            reg[name] = (kind, getattr(mod, attr))
        except (ImportError, AttributeError):
            pass
    return reg


def stage_layout(kind: str, rank: int, world: int, v: int):
    """Return (total_stages, sorted list of stage indices owned by this rank)."""
    if kind == "single":
        return world, [rank]
    if kind == "interleaved":
        total = world * v
        return total, sorted(rank + k * world for k in range(v))
    if kind == "zbv":  # V-layout, exactly 2 chunks/rank
        total = world * 2
        return total, sorted({rank, 2 * world - 1 - rank})
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-xl")
    ap.add_argument("--schedule", default="1f1b")
    ap.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--micro-batch-size", type=int, default=1)
    ap.add_argument("--num-micro-batches", type=int, default=8, help="M (pipeline micro-batches)")
    ap.add_argument("--virtual-stages", type=int, default=2, help="v for interleaved schedules")
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--optim", default="adamw8bit", choices=["adamw", "adamw8bit", "adafactor"])
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--balance", default="uniform", choices=["uniform", "params"])
    ap.add_argument("--loss-scale", type=float, default=1.0, help="static fp16 loss scale (convergence runs)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--results-csv", default="results/results.csv")
    ap.add_argument("--loss-log", default="", help="optional CSV for per-step loss (convergence curves)")
    ap.add_argument("--profile", action="store_true", help="dump a profiler trace for bubble analysis")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    set_seed(args.seed)
    rank, local_rank, world, device = setup_distributed("nccl")
    assert world >= 2, "torch.pipelining path needs >=2 ranks (torchrun --nproc_per_node=2)"

    reg = schedule_registry()
    if args.schedule not in reg:
        raise SystemExit(
            f"schedule {args.schedule!r} unavailable in torch {torch.__version__}. "
            f"Available: {sorted(reg)} (zero-bubble needs torch>=2.6)."
        )
    kind, ScheduleCls = reg[args.schedule]
    v = args.virtual_stages if kind == "interleaved" else (2 if kind == "zbv" else 1)
    total_stages, owned = stage_layout(kind, rank, world, v)

    # ---- data (replicated; identical order on every rank) ----------------- #
    tokenizer = get_tokenizer(args.model)
    train_ds = build_lm_dataset(args.model, args.dataset_config, "train",
                                args.seq_len, tokenizer, max_samples=args.max_train_samples)
    global_batch = args.micro_batch_size * args.num_micro_batches
    train_loader = make_dataloader(train_ds, global_batch, shuffle=True, seed=args.seed)

    # ---- build model, keep only this rank's stages on GPU ------------------ #
    all_stages, config, ranges = build_stage_modules(
        args.model, total_stages, attn_implementation=args.attn,
        dtype="float16", balance=args.balance,
    )
    hidden = getattr(config, "n_embd", None) or config.hidden_size
    vocab = config.vocab_size
    owned_mods = []
    for i in range(total_stages):
        if i in owned:
            all_stages[i].to(device).train()
            owned_mods.append(all_stages[i])
        else:
            all_stages[i] = None
    gc.collect()
    if rank == 0:
        print(f"[torchpp] schedule={args.schedule} total_stages={total_stages} "
              f"ranges={ranges} v={v}", flush=True)
    print(f"  rank{rank} owns stages {owned}", flush=True)

    # ---- PipelineStage objects (with example inputs for shape inference) --- #
    from torch.distributed.pipelining import PipelineStage

    def example_input(idx):
        if idx == 0:
            return torch.randint(0, vocab, (args.micro_batch_size, args.seq_len),
                                 dtype=torch.long, device=device)
        return torch.randn(args.micro_batch_size, args.seq_len, hidden,
                           dtype=torch.float16, device=device)

    def make_stage(idx, mod):
        # input_args (shape inference) is the torch>=2.4 path; fall back if the
        # installed version has a different signature.
        try:
            return PipelineStage(mod, idx, total_stages, device,
                                 input_args=(example_input(idx),))
        except TypeError:
            return PipelineStage(mod, idx, total_stages, device)

    stages = [make_stage(idx, mod) for idx, mod in zip(owned, owned_mods)]

    # ---- schedule + optimizer --------------------------------------------- #
    scale = args.loss_scale

    def loss_fn(output, target):
        return causal_lm_loss(output, target) * scale

    if kind == "single":
        schedule = ScheduleCls(stages[0], n_microbatches=args.num_micro_batches, loss_fn=loss_fn)
    else:
        schedule = ScheduleCls(stages, n_microbatches=args.num_micro_batches, loss_fn=loss_fn)

    params = [p for m in owned_mods for p in m.parameters()]
    optimizer = build_optimizer(params, args.optim, args.lr)

    is_first = 0 in owned
    is_last = (total_stages - 1) in owned

    # full-model param count for MFU (sum across ranks)
    local_n = sum(p.numel() for p in params)
    n_tensor = torch.tensor([local_n], device=device)
    dist.all_reduce(n_tensor, op=dist.ReduceOp.SUM)
    n_params = int(n_tensor.item())
    n_layer, n_head, head_dim = model_dims(config)

    meter = ThroughputMeter(warmup=args.warmup_steps)
    reset_peak_memory()

    def run_step(input_ids, labels):
        optimizer.zero_grad(set_to_none=True)
        losses = [] if is_last else None
        if is_first and is_last:
            schedule.step(input_ids, target=labels, losses=losses)
        elif is_first:
            schedule.step(input_ids)
        elif is_last:
            schedule.step(target=labels, losses=losses)
        else:
            schedule.step()
        if scale != 1.0:
            for p in params:
                if p.grad is not None:
                    p.grad.div_(scale)
        optimizer.step()
        if is_last and losses:
            return sum(l.item() for l in losses) / len(losses) / scale
        return 0.0

    profiler = None
    if args.profile and rank == 0:
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        )
        profiler.start()

    loss_log = LossLogger(args.loss_log if is_last else "", system="torchpp",
                          schedule=args.schedule, model=args.model,
                          num_micro_batches=args.num_micro_batches)
    step, last_loss = 0, 0.0
    data_iter = iter(train_loader)
    while step < args.max_steps:
        try:
            input_ids, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            input_ids, labels = next(data_iter)
        input_ids, labels = input_ids.to(device), labels.to(device)
        last_loss = run_step(input_ids, labels)
        step += 1
        meter.step(n_tokens=global_batch * args.seq_len, n_samples=global_batch)
        if profiler is not None:
            profiler.step()
        if is_last:
            loss_log.log(step, last_loss, meter.tokens_per_sec)
        if is_last and (step % 5 == 0 or step == 1):
            print(f"  step {step:4d}/{args.max_steps}  loss={last_loss:.4f}"
                  f"  tok/s={meter.tokens_per_sec:,.0f}", flush=True)
    loss_log.close()

    if profiler is not None:
        profiler.stop()
        trace = f"results/trace_torchpp_{args.schedule}_M{args.num_micro_batches}_rank{rank}.json"
        profiler.export_chrome_trace(trace)
        print(f"[torchpp] profiler trace -> {trace}", flush=True)

    # ---- gather memory across ranks --------------------------------------- #
    my_peak = peak_memory_gb(device)
    max_peak = all_reduce_max_gb(my_peak)
    t = torch.tensor([my_peak], device=device)
    dist.broadcast(t, src=0)
    rank0_peak = float(t.item())

    if kind == "interleaved":
        bub = interleaved_bubble_fraction(world, args.num_micro_batches, v)
    elif kind == "zbv":
        # ZBV's *design target* is ~0 bubble, but on 2x T4 (no NVLink) the split
        # backward rarely hides all idle time. Report the interleaved lower bound
        # (v=2) as the theoretical value and let the empirical-bubble plot
        # (bench/plot_analysis.py) show the real achieved gap -- do NOT log 0%.
        bub = interleaved_bubble_fraction(world, args.num_micro_batches, 2)
    else:
        bub = bubble_fraction(world, args.num_micro_batches)

    if is_last:
        result = RunResult(
            system="torchpp", model=args.model, dataset_config=args.dataset_config,
            schedule=args.schedule, world_size=world, num_stages=total_stages,
            micro_batch_size=args.micro_batch_size, num_micro_batches=args.num_micro_batches,
            global_batch=global_batch, seq_len=args.seq_len, optim=args.optim,
            grad_checkpointing=False, attn=args.attn, measured_steps=meter.measured_steps,
            tokens_per_sec=meter.tokens_per_sec, samples_per_sec=meter.samples_per_sec,
            peak_mem_gb_rank0=rank0_peak, peak_mem_gb_max=max_peak,
            bubble_theory=bub,
            mfu=mfu(meter.tokens_per_sec, n_params, n_layer, n_head, head_dim, args.seq_len, world),
            final_loss=last_loss, eval_ppl=0.0,
            torch_version=torch.__version__,
            notes=(args.notes + f" balance={args.balance} v={v}").strip(),
        )
        log_result(result, args.results_csv)
        print(f"[torchpp] schedule={args.schedule} M={args.num_micro_batches} "
              f"tok/s={result.tokens_per_sec:,.0f} peak/GPU_max={max_peak:.2f}GB "
              f"bubble~{bub*100:.1f}% MFU={result.mfu*100:.1f}% -> {args.results_csv}", flush=True)

    cleanup_distributed()


if __name__ == "__main__":
    main()
