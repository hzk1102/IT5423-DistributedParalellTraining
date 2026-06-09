# Topic 2 — Pipeline Parallelism for LLMs Exceeding GPU Memory

Course project (GenAI Master / Distributed & Parallel Training). We fine-tune
**`gpt2-xl` (1.5B)** — too large for one 16 GB T4 with a full Adam optimizer — using
**pipeline parallelism** on **2× T4**, and compare three systems on identical data:

1. **DeepSpeed Pipeline Parallelism** (`src/train_deepspeed.py`)
2. **`torch.distributed.pipelining`** across the modern schedule ladder
   GPipe → 1F1B → Interleaved → **Zero-Bubble** (`src/train_torchpp.py`)
3. **1-GPU + gradient checkpointing** baseline / control (`src/train_singlegpu.py`)

Metrics: throughput (tokens/s), peak VRAM/GPU, the pipeline **bubble** (theory vs
measured), MFU, and perplexity.

## Start here

- **Running it on Kaggle:** read **[KAGGLE_GUIDE.md](KAGGLE_GUIDE.md)** — a complete,
  copy-paste, zero-prior-knowledge walkthrough.
- **Why each decision was made:** read **[PROJECT_PLAN.md](PROJECT_PLAN.md)** (the
  authoritative design doc).


## Layout

```
src/      data, model-split, losses, metrics, optim, check_split, 3 training scripts
bench/    run_sweep.sh, aggregate_results.py, plots.py
configs/  ds_pp.json (DeepSpeed)
results/  CSV + per-run JSON + figures (created at runtime)
```

## Quick local sanity (no GPU)

```bash
uv sync
uv run --no-project python src/check_split.py --model gpt2 --num-stages 2  # needs torch
```

`check_split.py` requires torch (not installed locally by design — it's a Kaggle/GPU
concern). Locally you can build the data pipeline (`datasets`/`transformers` only); the
full training runs target the Kaggle 2× T4 notebook described in the guide.

> Hardware reality: at only P=2 over PCIe, pipeline parallelism is **memory-scaling, not
> speed-scaling** — its value is making the model trainable, not a ~2× speedup. The study
> quantifies exactly this. See PROJECT_PLAN.md §7.
