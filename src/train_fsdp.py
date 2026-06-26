"""PyTorch-native Data Parallelism via **torch.distributed.fsdp** (FSDP).

Fully Sharded Data Parallel: mỗi GPU xử lý một shard của batch riêng, nhưng
parameters / gradients / optimizer states được sharded đều giữa tất cả các rank.
Khác với Pipeline Parallelism (model split theo layer), FSDP split theo parameter
tensors → mỗi rank giữ toàn bộ kiến trúc nhưng chỉ 1/world_size parameters.

Benchmark axis (Data Parallelism):
    fsdp_full   - FULL_SHARD  : shard params + grads + optimizer (ZeRO-3)
    fsdp_grad   - SHARD_GRAD_OP: shard grads + optimizer only   (ZeRO-2)
    fsdp_no     - NO_SHARD    : replicate toàn bộ (= DDP baseline)

Launch:
    torchrun --standalone --nproc_per_node=2 src/train_fsdp.py \\
        --model gpt2-xl --shard-strategy full \\
        --micro-batch-size 2 --num-micro-batches 4 --seq-len 512 --max-steps 30

Notes
-----
- fp16 mixed-precision qua FSDP MixedPrecision (không dùng GradScaler vì FSDP
  tự xử lý reduce precision).
- gradient checkpointing tương thích FSDP khi wrap TRƯỚC khi gọi checkpoint.
- bubble_theory = 0.0 (DP không có pipeline bubble).
- Dùng DistributedSampler để mỗi rank thấy shard data riêng (khác PP dùng
  identical data order trên mọi rank).
"""
from __future__ import annotations

import argparse
import functools

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler

from data import build_lm_dataset, get_tokenizer, collate
from losses import causal_lm_loss
from metrics import (
    LossLogger, RunResult, ThroughputMeter,
    all_reduce_max_gb, log_result, mfu, model_dims,
    peak_memory_gb, reset_peak_memory,
)
from optim import build_optimizer
from utils import cleanup_distributed, set_seed, setup_distributed


# ---- Sharding strategy registry -------------------------------------------- #
SHARD_MAP = {
    "full": ShardingStrategy.FULL_SHARD,       # ZeRO-3: params+grads+optim sharded
    "grad": ShardingStrategy.SHARD_GRAD_OP,    # ZeRO-2: grads+optim sharded
    "no":   ShardingStrategy.NO_SHARD,         # DDP-equivalent (replicated)
}


def get_transformer_layer_cls(model):
    """Trả về class của transformer block để FSDP wrap từng block riêng."""
    # GPT-2 family
    try:
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block
        if any(isinstance(m, GPT2Block) for m in model.modules()):
            return GPT2Block
    except ImportError:
        pass
    # OPT family
    try:
        from transformers.models.opt.modeling_opt import OPTDecoderLayer
        if any(isinstance(m, OPTDecoderLayer) for m in model.modules()):
            return OPTDecoderLayer
    except ImportError:
        pass
    return None


def build_fsdp_model(model, device, shard_strategy: ShardingStrategy,
                     grad_checkpointing: bool = False):
    """Wrap model với FSDP. Nếu grad_checkpointing=True thì apply trước khi wrap."""
    layer_cls = get_transformer_layer_cls(model)

    # Gradient checkpointing phải apply trên raw module TRƯỚC khi FSDP wrap
    if grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Mixed precision: compute+communication fp16, master weights fp32
    mp_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

    wrap_policy = None
    if layer_cls is not None:
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={layer_cls},
        )

    fsdp_model = FSDP(
        model,
        sharding_strategy=shard_strategy,
        mixed_precision=mp_policy,
        auto_wrap_policy=wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=device,
        use_orig_params=True,   # tương thích bitsandbytes optimizer
    )
    return fsdp_model


def make_distributed_dataloader(dataset, global_batch: int, rank: int,
                                 world: int, seed: int = 1234):
    """Mỗi rank thấy global_batch // world samples (DistributedSampler).
    Khác PP: ở đó mỗi rank thấy *toàn bộ* batch vì model-parallel."""
    sampler = DistributedSampler(
        dataset, num_replicas=world, rank=rank,
        shuffle=True, seed=seed, drop_last=True,
    )
    per_rank_batch = max(1, global_batch // world)
    return DataLoader(
        dataset,
        batch_size=per_rank_batch,
        sampler=sampler,
        drop_last=True,
        num_workers=2,
        collate_fn=collate,
        pin_memory=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-xl")
    ap.add_argument("--shard-strategy", default="full",
                    choices=list(SHARD_MAP), help="full=ZeRO-3 | grad=ZeRO-2 | no=DDP")
    ap.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--micro-batch-size", type=int, default=2,
                    help="per-rank batch size (không phải micro-batch PP)")
    ap.add_argument("--num-micro-batches", type=int, default=4,
                    help="dùng để tính global_batch = micro_batch_size * num_micro_batches "
                         "(để so sánh cùng global batch với PP runs)")
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--optim", default="adamw8bit",
                    choices=["adamw", "adamw8bit", "adafactor"])
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--results-csv", default="results/results.csv")
    ap.add_argument("--loss-log", default="",
                    help="optional CSV cho per-step loss (convergence curves)")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    set_seed(args.seed)
    rank, local_rank, world, device = setup_distributed("nccl")
    assert world >= 1, "cần ít nhất 1 GPU"

    shard_strategy = SHARD_MAP[args.shard_strategy]
    # schedule label để so sánh với torchpp rows trong results.csv
    schedule_label = f"fsdp_{args.shard_strategy}"

    # global_batch giống cách tính của PP để so sánh apples-to-apples
    global_batch = args.micro_batch_size * args.num_micro_batches

    # ---- data ---------------------------------------------------------------- #
    tokenizer = get_tokenizer(args.model)
    train_ds = build_lm_dataset(
        args.model, args.dataset_config, "train",
        args.seq_len, tokenizer,
        max_samples=args.max_train_samples,
    )
    train_loader = make_distributed_dataloader(
        train_ds, global_batch, rank, world, args.seed,
    )

    # ---- model --------------------------------------------------------------- #
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained(args.model)
    if args.attn == "sdpa":
        config._attn_implementation = "sdpa"

    # Load trên CPU trước, FSDP sẽ tự shard lên GPU
    if rank == 0:
        print(f"[fsdp] Loading {args.model} on CPU ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        dtype=torch.float16,
    )
    model = build_fsdp_model(
        model, device, shard_strategy,
        grad_checkpointing=args.grad_checkpointing,
    )

    # ---- optimizer ----------------------------------------------------------- #
    # use_orig_params=True cho phép truyền model.parameters() trực tiếp
    optimizer = build_optimizer(model.parameters(), args.optim, args.lr)

    # ---- param count (all_reduce vì FSDP shards params) --------------------- #
    local_n = sum(p.numel() for p in model.parameters())
    if world > 1 and shard_strategy != ShardingStrategy.NO_SHARD:
        # FSDP FULL_SHARD: mỗi rank chỉ giữ 1/world params → nhân world lại
        # dùng all_reduce để lấy tổng thực
        n_tensor = torch.tensor([local_n], dtype=torch.long, device=device)
        dist.all_reduce(n_tensor, op=dist.ReduceOp.SUM)
        n_params = int(n_tensor.item())
    else:
        n_params = local_n
    n_layer, n_head, head_dim = model_dims(config)

    if rank == 0:
        strategy_name = {
            ShardingStrategy.FULL_SHARD:    "FULL_SHARD (ZeRO-3)",
            ShardingStrategy.SHARD_GRAD_OP: "SHARD_GRAD_OP (ZeRO-2)",
            ShardingStrategy.NO_SHARD:      "NO_SHARD (DDP)",
        }[shard_strategy]
        print(
            f"[fsdp] {args.model} params={n_params/1e9:.2f}B  world={world}  "
            f"strategy={strategy_name}  global_batch={global_batch}  "
            f"grad_ckpt={args.grad_checkpointing}",
            flush=True,
        )

    # ---- training loop ------------------------------------------------------- #
    meter = ThroughputMeter(warmup=args.warmup_steps)
    reset_peak_memory()

    loss_log = LossLogger(
        args.loss_log if rank == 0 else "",
        system="fsdp", schedule=schedule_label,
        model=args.model, num_micro_batches=args.num_micro_batches,
    )

    step, last_loss = 0, 0.0
    data_iter = iter(train_loader)

    while step < args.max_steps:
        try:
            input_ids, labels = next(data_iter)
        except StopIteration:
            # epoch ended — reshuffle sampler cho epoch mới
            train_loader.sampler.set_epoch(step)
            data_iter = iter(train_loader)
            input_ids, labels = next(data_iter)

        input_ids = input_ids.to(device)
        labels    = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # dùng causal_lm_loss từ losses.py — giữ fp16, đồng bộ với train_torchpp.py
        outputs = model(input_ids=input_ids)
        loss = causal_lm_loss(outputs.logits, labels)

        loss.backward()              # FSDP reduce-scatter grads tự động

        optimizer.step()

        last_loss = loss.item()
        step += 1
        # global tokens = per_rank_batch * world * seq_len
        per_rank_batch = input_ids.size(0)
        meter.step(
            n_tokens=per_rank_batch * world * args.seq_len,
            n_samples=per_rank_batch * world,
        )

        if rank == 0:
            loss_log.log(step, last_loss, meter.tokens_per_sec)
            if step % 5 == 0 or step == 1:
                print(
                    f"  step {step:4d}/{args.max_steps}  "
                    f"loss={last_loss:.4f}  tok/s={meter.tokens_per_sec:,.0f}",
                    flush=True,
                )

    loss_log.close()

    # ---- gather metrics ------------------------------------------------------ #
    my_peak  = peak_memory_gb(device)
    max_peak = all_reduce_max_gb(my_peak)
    t = torch.tensor([my_peak], device=device)
    dist.broadcast(t, src=0)
    rank0_peak = float(t.item())

    if rank == 0:
        result = RunResult(
            system="fsdp",
            model=args.model,
            dataset_config=args.dataset_config,
            schedule=schedule_label,          # e.g. "fsdp_full"
            world_size=world,
            num_stages=world,                  # DP: mỗi GPU là 1 "stage" độc lập
            micro_batch_size=args.micro_batch_size,
            num_micro_batches=args.num_micro_batches,
            global_batch=global_batch,
            seq_len=args.seq_len,
            optim=args.optim,
            grad_checkpointing=args.grad_checkpointing,
            attn=args.attn,
            measured_steps=meter.measured_steps,
            tokens_per_sec=meter.tokens_per_sec,
            samples_per_sec=meter.samples_per_sec,
            peak_mem_gb_rank0=rank0_peak,
            peak_mem_gb_max=max_peak,
            bubble_theory=0.0,                 # DP không có pipeline bubble
            mfu=mfu(
                meter.tokens_per_sec, n_params,
                n_layer, n_head, head_dim, args.seq_len, world,
            ),
            final_loss=last_loss,
            eval_ppl=0.0,
            torch_version=torch.__version__,
            notes=(
                f"shard={args.shard_strategy} "
                + args.notes
            ).strip(),
        )
        log_result(result, args.results_csv)
        print(
            f"[fsdp] strategy={args.shard_strategy} "
            f"tok/s={result.tokens_per_sec:,.0f} "
            f"peak/GPU_max={max_peak:.2f}GB "
            f"bubble~0.0% "
            f"MFU={result.mfu*100:.1f}% "
            f"-> {args.results_csv}",
            flush=True,
        )

    cleanup_distributed()


if __name__ == "__main__":
    main()
