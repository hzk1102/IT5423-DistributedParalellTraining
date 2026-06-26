"""DeepSpeed Pipeline Parallelism (PROJECT_PLAN.md §4A).

Builds an HF causal-LM as a DeepSpeed `PipelineModule` (a flat list of layers that
DeepSpeed partitions across stages), trains with the built-in **1F1B** schedule,
and sweeps `--num-micro-batches`. DeepSpeed manages fp16 master weights + dynamic
loss scaling, so this is the most robust convergence path on T4.

Launch on 2 GPUs:
    deepspeed --num_gpus 2 src/train_deepspeed.py \
        --model gpt2-xl --num-micro-batches 8 --micro-batch-size 1 \
        --seq-len 512 --max-steps 50 --eval

Note: DeepSpeed slices ONE global batch into `gradient_accumulation_steps`
micro-batches itself, so the DataLoader yields micro-batches (batch_size =
micro_batch_size), and one `engine.train_batch()` == one optimizer step.
"""
from __future__ import annotations

import argparse
import math
import time

import torch

from data import build_lm_dataset, get_tokenizer, make_dataloader
from losses import causal_lm_loss
from metrics import (LossLogger, RunResult, ThroughputMeter, all_reduce_max_gb,
                     bubble_fraction, log_result, mfu, model_dims,
                     peak_memory_gb, reset_peak_memory)
from model_split import build_ordered_layers
from optim import build_optimizer
from utils import env_local_rank, env_rank, env_world_size, set_seed


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-xl")
    ap.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--micro-batch-size", type=int, default=1)
    ap.add_argument("--num-micro-batches", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--optim", default="adamw8bit", choices=["adamw", "adamw8bit", "adafactor"])
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--partition", default="parameters",
                    help="DeepSpeed partition_method: parameters | uniform | type:<name>")
    ap.add_argument("--activation-checkpoint", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--eval-config", default=None)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--results-csv", default="results/results.csv")
    ap.add_argument("--loss-log", default="", help="optional CSV for per-step loss (convergence curves)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--local_rank", type=int, default=0)  # injected by the deepspeed launcher
    args = ap.parse_args()

    import deepspeed
    from deepspeed.pipe import PipelineModule

    set_seed(args.seed)
    deepspeed.init_distributed(dist_backend="nccl")
    rank, local_rank, world = env_rank(), env_local_rank(), env_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    assert world >= 2, "DeepSpeed PP path needs >=2 GPUs (deepspeed --num_gpus 2)"

    # ---- data: DataLoader yields MICRO-batches; DeepSpeed accumulates M ---- #
    tokenizer = get_tokenizer(args.model)
    train_ds = build_lm_dataset(args.model, args.dataset_config, "train",
                                args.seq_len, tokenizer, max_samples=args.max_train_samples)
    train_loader = make_dataloader(train_ds, args.micro_batch_size, shuffle=True, seed=args.seed)
    train_iter = infinite(train_loader)

    # ---- model -> flat layer list -> PipelineModule ----------------------- #
    layers, config = build_ordered_layers(args.model, attn_implementation=args.attn, dtype="float16")
    n_params = sum(p.numel() for m in layers for p in m.parameters())
    n_layer, n_head, head_dim = model_dims(config)

    net = PipelineModule(
        layers=layers,
        num_stages=world,
        loss_fn=causal_lm_loss,
        partition_method=args.partition,
        activation_checkpoint_interval=1 if args.activation_checkpoint else 0,
    )

    ds_config = {
        "train_micro_batch_size_per_gpu": args.micro_batch_size,
        "gradient_accumulation_steps": args.num_micro_batches,
        "fp16": {"enabled": True, "loss_scale": 0, "initial_scale_power": 12,
                 "loss_scale_window": 1000, "hysteresis": 2, "min_loss_scale": 1},
        "gradient_clipping": 1.0,
        "steps_per_print": 10_000,
        "wall_clock_breakdown": False,
    }

    client_opt = build_optimizer(net.parameters(), args.optim, args.lr)
    engine, _, _, _ = deepspeed.initialize(
        args=args, model=net, optimizer=client_opt, config=ds_config
    )

    global_batch = args.micro_batch_size * args.num_micro_batches
    meter = ThroughputMeter(warmup=args.warmup_steps)
    reset_peak_memory()
    if rank == 0:
        print(f"[deepspeed] {args.model} params={n_params/1e9:.2f}B world={world} "
              f"partition={args.partition} M={args.num_micro_batches} "
              f"global_batch={global_batch}", flush=True)

    loss_log = LossLogger(args.loss_log if rank == 0 else "", system="deepspeed",
                          schedule="1f1b", model=args.model,
                          num_micro_batches=args.num_micro_batches)
    last_loss = 0.0
    for step in range(1, args.max_steps + 1):
        loss = engine.train_batch(data_iter=train_iter)
        last_loss = float(loss.item()) if hasattr(loss, "item") else float(loss)
        meter.step(n_tokens=global_batch * args.seq_len, n_samples=global_batch)
        if rank == 0:
            loss_log.log(step, last_loss, meter.tokens_per_sec)
        if rank == 0 and (step % 5 == 0 or step == 1):
            print(f"  step {step:4d}/{args.max_steps}  loss={last_loss:.4f}"
                  f"  tok/s={meter.tokens_per_sec:,.0f}", flush=True)
    loss_log.close()

    # ---- eval perplexity via DeepSpeed pipeline eval ---------------------- #
    eval_ppl = 0.0
    if args.eval:
        eval_ds = build_lm_dataset(args.model, args.eval_config or args.dataset_config,
                                   "validation", args.seq_len, tokenizer)
        eval_loader = make_dataloader(eval_ds, args.micro_batch_size, shuffle=False, drop_last=True)
        eval_iter = infinite(eval_loader)
        losses = []
        for _ in range(10):
            try:
                l = engine.eval_batch(data_iter=eval_iter)
                losses.append(float(l.item()) if hasattr(l, "item") else float(l))
            except Exception as e:  # eval_batch availability varies by DS version
                if rank == 0:
                    print(f"[deepspeed] eval_batch unavailable ({e}); skipping ppl", flush=True)
                break
        if losses:
            eval_ppl = float(math.exp(min(20.0, sum(losses) / len(losses))))
            if rank == 0:
                print(f"[deepspeed] eval perplexity = {eval_ppl:.2f}", flush=True)

    my_peak = peak_memory_gb(device)
    max_peak = all_reduce_max_gb(my_peak)
    rank0_t = torch.tensor([my_peak], device=device)
    torch.distributed.broadcast(rank0_t, src=0)
    rank0_peak = float(rank0_t.item())

    if rank == 0:
        result = RunResult(
            system="deepspeed", model=args.model, dataset_config=args.dataset_config,
            schedule="1f1b", world_size=world, num_stages=world,
            micro_batch_size=args.micro_batch_size, num_micro_batches=args.num_micro_batches,
            global_batch=global_batch, seq_len=args.seq_len, optim=args.optim,
            grad_checkpointing=args.activation_checkpoint, attn=args.attn,
            measured_steps=meter.measured_steps, tokens_per_sec=meter.tokens_per_sec,
            samples_per_sec=meter.samples_per_sec, peak_mem_gb_rank0=rank0_peak,
            peak_mem_gb_max=max_peak, bubble_theory=bubble_fraction(world, args.num_micro_batches),
            mfu=mfu(meter.tokens_per_sec, n_params, n_layer, n_head, head_dim, args.seq_len, world),
            final_loss=last_loss, eval_ppl=eval_ppl, torch_version=torch.__version__,
            notes=(args.notes + f" partition={args.partition}").strip(),
        )
        log_result(result, args.results_csv)
        print(f"[deepspeed] M={args.num_micro_batches} tok/s={result.tokens_per_sec:,.0f} "
              f"peak/GPU_max={max_peak:.2f}GB bubble~{result.bubble_theory*100:.1f}% "
              f"MFU={result.mfu*100:.1f}% -> {args.results_csv}", flush=True)


if __name__ == "__main__":
    main()
