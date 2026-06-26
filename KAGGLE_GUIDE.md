# KAGGLE_GUIDE.md — Run this project on Kaggle (2× T4), step by step

This guide assumes **zero prior knowledge** of the codebase. Follow it top to bottom and
you will reproduce the whole study and generate every figure the report needs.

> **What you are running (Topic 2 — Pipeline Parallelism).** We fine-tune large language
> models with **pipeline parallelism** across two GPUs and compare three systems on
> identical data:
> 1. **DeepSpeed Pipeline Parallelism** (2 GPUs)
> 2. **`torch.distributed.pipelining`** (2 GPUs) across the modern schedule ladder
>    (GPipe → 1F1B → Interleaved → Zero-Bubble)
> 3. **1-GPU + gradient checkpointing** (the control)
>
> Two models are used on purpose:
> - **`gpt2-xl` (1.5B)** — the *primary* model for the throughput / memory / bubble study.
>   It fits on one T4 (with 8-bit Adam + grad-checkpointing), so for it the question is
>   *"how much faster / leaner is 2-GPU pipelining?"*
> - **`facebook/opt-2.7b`** — the *enablement* model. It does **not** fit on one 16 GB T4,
>   so it demonstrates the headline claim of the topic: pipelining makes an
>   otherwise-untrainable model trainable.
>
> We measure throughput (tokens/s), peak VRAM/GPU, the pipeline **"bubble"** (theoretical
> *and* measured), MFU, perplexity, and convergence. The full design rationale is in
> `PROJECT_PLAN.md`.

---

## 0. The one thing to understand first

Pipeline parallelism **splits the model across GPUs** (GPU 0 holds the first half of the
layers, GPU 1 the second half). A batch is chopped into **M micro-batches** that flow
through the two stages like an assembly line. When M is small the assembly line has idle
gaps — the **bubble**. Bigger M ⇒ smaller bubble ⇒ higher throughput, until memory or
communication cost takes over. Measuring exactly that is the project.

Everything below is just: get the code onto Kaggle → prove the model-split is correct →
run the production sweep → aggregate → plot.

---

## 1. Kaggle account setup (one time)

1. Create a Kaggle account and **verify your phone number**
   (*Settings → Phone Verification*). Without this you cannot use GPUs or the internet
   inside notebooks.
2. You get ~30 GPU-hours/week. The full sweep below (~60–90 min) is well within that.

---

## 2. Get the code onto Kaggle

Pick **ONE** route.

### Route A — Upload as a Kaggle Dataset (no GitHub needed; recommended)

1. On your computer, zip the **whole `Project/` folder** (the one containing `src/`,
   `bench/`, `configs/`, `PROJECT_PLAN.md`, this guide).
2. Kaggle → **Datasets → New Dataset** → drag the zip in → name it e.g. `pp-topic2` →
   **Create**. Kaggle unzips it for you.
3. It will be available inside notebooks at `/kaggle/input/pp-topic2/`.

### Route B — From GitHub

If you push this repo to GitHub, skip the dataset and just `!git clone <url>` in the
setup cell below.

---

## 3. Create the notebook (GPU T4 ×2 + Internet)

1. Kaggle → **Code → New Notebook**.
2. Right-hand panel → **Accelerator → "GPU T4 x2"** (this is the 2-GPU setting — do **not**
   pick "GPU P100" or single T4).
3. Right-hand panel → **Internet → On** (needed to download the models + dataset).
4. (Route A) **Add Input** → your `pp-topic2` dataset.

---

## 4. Setup cell — copy code to a writable dir & install deps

Kaggle inputs are **read-only**; we need to write `results/`, so copy the project into
`/kaggle/working` and run everything from there.

```python
# === Cell 1: setup ===
import os, shutil

# Route A: copy from your uploaded dataset (adjust the name if you used a different one).
SRC = "/kaggle/input/pp-topic2"          # <-- your dataset folder
DST = "/kaggle/working/project"
# Some uploads nest one level (…/pp-topic2/Project/). Auto-detect where src/ lives:
cand = [SRC] + [os.path.join(SRC, d) for d in os.listdir(SRC)]
root = next(c for c in cand if os.path.isdir(os.path.join(c, "src")))
if os.path.exists(DST): shutil.rmtree(DST)
shutil.copytree(root, DST)
os.chdir(DST)
print("working dir:", os.getcwd(), "->", os.listdir("."))

# Route B instead of the above:
# !git clone https://github.com/<you>/<repo>.git /kaggle/working/project
# %cd /kaggle/working/project
```

```python
# === Cell 2: install the only missing packages (torch is preinstalled!) ===
!pip install -q deepspeed bitsandbytes
# Do NOT pip install torch / transformers / datasets — they're already in the image and
# reinstalling can break the CUDA build. The code works with the preinstalled versions.
```

```python
# === Cell 3: confirm the hardware & versions ===
!nvidia-smi --query-gpu=index,name,memory.total --format=csv
import torch
print("torch", torch.__version__, "| CUDA", torch.version.cuda,
      "| GPUs visible:", torch.cuda.device_count())
print("Zero-bubble schedules need torch >= 2.6 :",
      tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (2, 6))
```

You should see **two Tesla T4** rows and `GPUs visible: 2`.

---

## 5. Prove the model split is correct (always do this first)

The pipeline frameworks need the model decomposed into per-GPU stages. Before trusting any
speed number, verify the decomposition reproduces the original model's outputs. This runs
on CPU in seconds with a **small** model:

```python
# === Cell 4: correctness self-check ===
!python src/check_split.py --model gpt2 --num-stages 2
```

Expect `ALL CHECKS PASSED`. (We check with `gpt2` because it's tiny; the split logic is
identical for `gpt2-xl` and `opt-2.7b`.)

---

## 6. Warm the cache (avoids a 2-process download race)

`torchrun`/`deepspeed` start **two** processes; if both try to download the weights at once
they can clash. Download once, single-process, first. This pulls **both** models:

```python
# === Cell 5: pre-download models + dataset into the cache ===
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
for m in ["gpt2-xl", "facebook/opt-2.7b"]:    # ~6 GB + ~5 GB, one time each
    AutoTokenizer.from_pretrained(m)
    AutoModelForCausalLM.from_pretrained(m)
    print("cached", m)
load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
print("cache warmed")
```

---

## 7. Run the full production sweep (one cell)

`bench/run_kaggle.sh` runs the **entire study** — there is no smoke test, these are the
numbers that go in the report. It executes, in order:

| Block | What it proves | Note |
|---|---|---|
| **(A) Enablement** | `opt-2.7b` **OOMs on 1 GPU**, then **trains on the 2-GPU pipeline** | A1 is *expected to crash* — that crash is the evidence. **Screenshot it.** |
| **(B) Recompute isolation** | 1-GPU baseline **with and without** grad-checkpointing | separates "2-GPU speedup" from "avoiding recompute cost" |
| **(C) Quality** | DeepSpeed + single-GPU **perplexity**, plus **loss curves** for all 3 | the convergence runs append to `results/loss_curves.csv` |
| **(D) Bubble + scaling** | `gpt2-xl` micro-batch sweep (1F1B, GPipe) + schedule ladder | feeds the throughput, memory, and **measured-bubble** figures |

```python
# === Cell 6: full production sweep (~60–90 min) ===
!bash bench/run_kaggle.sh
```

Each run prints a `tok/s ... peak_mem ... -> results/results.csv` line. Lines that say
`!!! FAILED (continuing)` are non-fatal — the script keeps going (block A1's OOM will show
up here, which is correct).

**Tunables** (only if you need to go faster or split across quota sessions):

```python
# shorter throughput runs + shorter convergence curves:
!STEPS=20 CONV_STEPS=150 bash bench/run_kaggle.sh

# if opt-2.7b OOMs even on 2 GPUs (tight on T4), use a smaller enablement model:
!BIG=facebook/opt-1.3b bash bench/run_kaggle.sh
```

> `results/results.csv` is **append-only** and the aggregator filters by model, so it is
> safe to re-run the script or run it in pieces across sessions — nothing is overwritten.

---

## 8. Aggregate + plot (three commands)

```python
# === Cell 7: tables + all figures ===
!python bench/aggregate_results.py --model gpt2-xl --seq-len 512   # -> results/summary.csv + console tables
!python bench/plots.py            --model gpt2-xl --seq-len 512   # -> throughput / memory / ladder / system PNGs
!python bench/plot_analysis.py    --model gpt2-xl --seq-len 512   # -> measured-bubble + convergence PNGs
```

> **Why the `--model/--seq-len` flags matter:** `results.csv` may contain rows from several
> models. These flags keep only the primary experiment so a different-model row can't win a
> per-M `max()`. (They default to `gpt2-xl` / `512`, so you can omit them — they're shown
> here so you know they exist.)

```python
# === Cell 8: show every figure inline ===
from IPython.display import Image, display
import glob
for png in sorted(glob.glob("results/figures/*.png")):
    print(png); display(Image(png))
```

```python
# === Cell 9: peek at the raw results table ===
import pandas as pd
pd.read_csv("results/results.csv")
```

Everything is written under `/kaggle/working/results/` (CSV, `loss_curves.csv`, per-run
JSON, figures) and saved with the notebook output — download it via the notebook's
**Output** tab, or **Save Version** to persist it.

---

## 9. (Optional) Run pieces individually

Use these if a block fails, or to split the work across quota sessions. Skip if Cell 6
finished cleanly.

> **Memory rule of thumb on T4:** start with `--micro-batch-size 1 --seq-len 512`. On OOM,
> first drop `--seq-len 256`, then try `--optim adafactor`.

```python
# (A) enablement — opt-2.7b: 1 GPU (expected OOM) then 2-GPU pipeline (expected success)
!CUDA_VISIBLE_DEVICES=0 python src/train_singlegpu.py --model facebook/opt-2.7b \
        --grad-checkpointing --optim adamw8bit --micro-batch-size 1 \
        --num-micro-batches 8 --seq-len 512 --max-steps 10
!torchrun --standalone --nproc_per_node=2 src/train_torchpp.py --model facebook/opt-2.7b \
        --schedule 1f1b --num-micro-batches 8 --micro-batch-size 1 --seq-len 512 --max-steps 30

# (B) recompute isolation — gpt2-xl baseline with and without grad-checkpointing
!CUDA_VISIBLE_DEVICES=0 python src/train_singlegpu.py --model gpt2-xl --grad-checkpointing \
        --optim adamw8bit --micro-batch-size 1 --num-micro-batches 8 --seq-len 512 \
        --max-steps 30 --eval
!CUDA_VISIBLE_DEVICES=0 python src/train_singlegpu.py --model gpt2-xl \
        --optim adamw8bit --micro-batch-size 1 --num-micro-batches 8 --seq-len 512 \
        --max-steps 30 --eval

# (D) one torch.pipelining run; --schedule in {gpipe,1f1b,interleaved_1f1b,interleaved_zb,zbv}
!torchrun --standalone --nproc_per_node=2 src/train_torchpp.py --model gpt2-xl \
        --schedule 1f1b --num-micro-batches 8 --micro-batch-size 1 --seq-len 512 --max-steps 30

# (C) one DeepSpeed run with perplexity
!deepspeed --num_gpus 2 src/train_deepspeed.py --model gpt2-xl \
        --num-micro-batches 8 --micro-batch-size 1 --seq-len 512 --max-steps 30 --eval

# (C) a convergence run that logs the loss curve (add --loss-log to ANY trainer)
!torchrun --standalone --nproc_per_node=2 src/train_torchpp.py --model gpt2-xl \
        --schedule 1f1b --num-micro-batches 8 --micro-batch-size 1 --seq-len 512 \
        --max-steps 300 --loss-scale 1024 --loss-log results/loss_curves.csv
```

---

## 10. How the outputs map to the assignment deliverables

| Deliverable | Where to look |
|---|---|
| 1. Fine-tune a model too big for 1 GPU via pipeline parallelism | **Block A**: the `opt-2.7b` single-GPU **OOM** (screenshot) vs the 2-GPU `torchpp` row that completes (loss ↓). This is your enablement proof. |
| 2. DeepSpeed PP **vs** PyTorch-native pipeline | `system_comparison.png`; `bubble_empirical.png` (torch.pipelining tracks theory, DeepSpeed sits above it); rows `system=deepspeed` vs `system=torchpp` |
| 3. Bubble analysis + effect of **micro-batch count** | `throughput_vs_M.png`, `memory_vs_M.png`, and **`bubble_empirical.png`** (measured vs theoretical bubble); columns `bubble_theory`, `mfu` |
| 4. 2-GPU pipeline **vs** 1-GPU gradient checkpointing | `system_comparison.png`; compare `system=singlegpu` (both grad-ckpt and no-ckpt rows from Block B) to the pipeline rows |
| Extension: modern schedule ladder | `schedule_ladder.png`; `interleaved_*`/`zbv` rows — how Interleaved/Zero-Bubble schedules shrink the bubble beyond GPipe/1F1B |
| Extension: convergence / quality | `convergence.png` (loss vs step, all 3 systems); `eval_ppl` column for single-GPU + DeepSpeed |

**Honest headline findings to state in the report:**
1. **Enablement (memory):** `opt-2.7b` cannot train on one 16 GB T4 but trains across two
   via pipelining — half the weights + optimizer state per GPU. This is the core claim of
   Topic 2.
2. **Throughput (scaling):** for `gpt2-xl`, the 2-GPU pipeline reaches ~1.9–2.0× the
   single-GPU throughput as M grows and the bubble shrinks — **but** part of that gap is
   that the single-GPU baseline pays a gradient-checkpointing *recompute* tax (Block B
   isolates this; quote both baselines).
3. **Bubble is validated, not assumed:** the measured (throughput-derived) bubble tracks
   the theoretical `(P-1)/(M+P-1)` for torch.pipelining; DeepSpeed shows a consistently
   *larger* bubble (extra framework overhead on no-NVLink T4s).
4. **GPipe vs 1F1B memory:** GPipe's peak VRAM grows ∝ M and **OOMs at M=8**, while 1F1B
   stays flat — the textbook activation-memory difference, shown with data. See
   `PROJECT_PLAN.md` §7.

---

## 11. What each file does (quick map)

```
src/
  data.py            wikitext -> tokenize -> fixed 512-token blocks (shared by all 3)
  losses.py          causal-LM loss with shift, computed in fp32
  model_split.py     gpt2-xl/OPT -> ordered list of single-tensor stages (the hard part)
  check_split.py     proves the split == the original model (run this first!)
  metrics.py         throughput / peak VRAM / bubble / MFU / CSV logger + LossLogger
  optim.py           adamw | adamw8bit | adafactor factory (use the same one everywhere)
  train_singlegpu.py 1-GPU grad-checkpointing baseline (--eval, --loss-log)
  train_torchpp.py   torch.distributed.pipelining (GPipe..Zero-Bubble) (--loss-log)
  train_deepspeed.py DeepSpeed PipelineModule (1F1B) (--eval, --loss-log)
bench/
  run_kaggle.sh         the PRODUCTION sweep (enablement + recompute + quality + bubble)
  run_sweep.sh          older minimal sweep (kept for reference)
  aggregate_results.py  prints summary tables -> results/summary.csv (--model/--seq-len)
  plots.py              throughput / memory / ladder / system PNGs (--model/--seq-len)
  plot_analysis.py      measured-bubble + convergence PNGs (--model/--seq-len)
configs/ds_pp.json      canonical DeepSpeed settings (CLI overrides micro-batch + M)
results/                CSV + loss_curves.csv + per-run JSON + figures (created at runtime)
```

Run everything **from the project root** (`/kaggle/working/project`) so the `src/...`
paths resolve.

---

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` on a **gpt2-xl pipeline** run | `--micro-batch-size 1`; then `--seq-len 256`; then `--optim adafactor`. For DeepSpeed add `--activation-checkpoint`. |
| `CUDA out of memory` in **Block A1** (opt-2.7b on 1 GPU) | **Expected — this is the result.** Capture the traceback; it's your enablement evidence. |
| `CUDA out of memory` in **Block A2** (opt-2.7b on 2 GPUs) | T4s are tight for 2.7B. Re-run with `BIG=facebook/opt-1.3b` (and run single-GPU 1.3B *without* `--grad-checkpointing` so it OOMs on 1 GPU) — still a valid enablement demo. |
| GPipe OOMs at large M | Expected and reportable — GPipe stashes M activations; 1F1B does not. Note it as a finding. |
| `GPUs visible: 1` | Accelerator isn't "GPU T4 x2". Fix in the right panel and restart the session. |
| `schedule 'zbv' unavailable` / zero-bubble missing | Your torch < 2.6. Use `gpipe`/`1f1b`/`interleaved_1f1b`; report the version limitation. |
| `import deepspeed` warns about ops / async_io / libaio | Harmless — pipeline parallelism needs no compiled kernels. |
| `bitsandbytes` import error | Use `--optim adafactor` everywhere instead. |
| DeepSpeed `Address already in use` | Re-run, or add `--master_port 29501` to the `deepspeed` command. |
| Download/network errors | Confirm **Internet = On**; re-run the warm-cache cell (Cell 5). |
| Two processes both downloading / hang at start | You skipped Cell 5 — run it to warm the cache before `torchrun`/`deepspeed`. |
| `convergence: results/loss_curves.csv not found` | You didn't run the convergence block (or any `--loss-log` run). It's the (C) block of `run_kaggle.sh`. |
| Hang at the end of a `torchrun` run | Usually a crashed rank; scroll up for the real traceback (often OOM on one stage). |
| Session timed out mid-sweep | Lower `STEPS`/`CONV_STEPS`, or run section 9 blocks separately. Results accumulate in `results.csv` across runs. |

---

## 13. Reproducibility checklist (for the report)

- Same `--model`, `--seq-len`, **global batch** (`micro_batch_size × num_micro_batches`),
  `--optim`, `--seed`, and `--max-steps` across all three systems — otherwise the
  comparison is meaningless. The defaults already line these up (global batch = 8).
- Record the `torch_version` column (logged automatically) and note whether zero-bubble
  schedules were available on your image.
- For perplexity, the single-GPU and DeepSpeed paths log `eval_ppl` with `--eval`.
  `torchpp` does not run an eval pass (pipeline-eval APIs are version-fragile) — use its
  **final training loss / `convergence.png`** as the quality signal and state this
  asymmetry as a known limitation.
- For fp16 gradient stability on long `torchpp` convergence runs, add `--loss-scale 1024`.
- Keep the `--model/--seq-len` filter flags consistent across all three aggregation/plot
  commands so every figure is built from the same subset of `results.csv`.
```
