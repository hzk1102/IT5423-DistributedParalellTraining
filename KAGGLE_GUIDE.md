# KAGGLE_GUIDE.md — Run this project on Kaggle (2× T4), step by step

This guide assumes **zero prior knowledge** of the codebase. Follow it top to bottom
and you will reproduce the whole study: a 1.5B model (`gpt2-xl`) fine-tuned with
**pipeline parallelism** across two GPUs, compared three ways.

> **What you are running (Topic 2).** We fine-tune `gpt2-xl` — too big for one 16 GB
> T4 with a full Adam optimizer — and compare three systems on identical data:
> 1. **DeepSpeed Pipeline Parallelism** (2 GPUs)
> 2. **`torch.distributed.pipelining`** (2 GPUs) across the modern schedule ladder
>    (GPipe → 1F1B → Interleaved → Zero-Bubble)
> 3. **1-GPU + gradient checkpointing** (the control)
>
> We measure throughput (tokens/s), peak VRAM/GPU, the pipeline **"bubble"**, MFU,
> and perplexity. The full design rationale is in `PROJECT_PLAN.md`.

---

## 0. The one thing to understand first

Pipeline parallelism **splits the model across GPUs** (GPU 0 holds the first half of
the layers, GPU 1 the second half). A batch is chopped into **M micro-batches** that
flow through the two stages like an assembly line. When M is small the assembly line
has idle gaps — the **bubble**. Bigger M ⇒ smaller bubble ⇒ higher throughput, until
memory or communication cost takes over. Measuring exactly that is the project.

Everything below is just: get the code onto Kaggle → prove the model-split is correct →
run the three systems → aggregate → plot.

---

## 1. Kaggle account setup (one time)

1. Create a Kaggle account and **verify your phone number**
   (*Settings → Phone Verification*). Without this you cannot use GPUs or the internet
   inside notebooks.
2. You get ~30 GPU-hours/week. The full sweep below is well within that.

---

## 2. Get the code onto Kaggle

Pick **ONE** route.

### Route A — Upload as a Kaggle Dataset (no GitHub needed; recommended)

1. On your computer, zip the **whole `Project/` folder** (the one containing `src/`,
   `bench/`, `configs/`, `PROJECT_PLAN.md`, this guide).
2. Kaggle → **Datasets → New Dataset** → drag the zip in → name it e.g.
   `pp-topic2` → **Create**. Kaggle unzips it for you.
3. It will be available inside notebooks at `/kaggle/input/pp-topic2/`.

### Route B — From GitHub

If you push this repo to GitHub, skip the dataset and just `!git clone <url>` in the
setup cell below.

---

## 3. Create the notebook (GPU T4 ×2 + Internet)

1. Kaggle → **Code → New Notebook**.
2. Right-hand panel → **Accelerator → "GPU T4 x2"** (this is the 2-GPU setting — do not
   pick "GPU P100" or single T4).
3. Right-hand panel → **Internet → On** (needed to download the model + dataset).
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
# === Cell 2: install the only two missing packages (torch is preinstalled!) ===
!pip install -q deepspeed bitsandbytes
# Do NOT pip install torch/transformers/datasets — they're already in the image and
# reinstalling can break the CUDA build.
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

The pipeline frameworks need `gpt2-xl` decomposed into per-GPU stages. Before trusting
any speed number, verify the decomposition reproduces the original model's outputs.
This runs on CPU in seconds with a **small** model:

```python
# === Cell 4: correctness self-check ===
!python src/check_split.py --model gpt2 --num-stages 2
```

Expect `ALL CHECKS PASSED`. (We check with `gpt2` because it's tiny; the split logic is
identical for `gpt2-xl`.)

---

## 6. Warm the cache (avoids a 2-process download race)

`torchrun`/`deepspeed` start **two** processes; if both try to download `gpt2-xl` at
once they can clash. Download once, single-process, first:

```python
# === Cell 5: pre-download model + dataset into the cache ===
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
MODEL = "gpt2-xl"
AutoTokenizer.from_pretrained(MODEL)
AutoModelForCausalLM.from_pretrained(MODEL)            # ~6 GB download, one time
load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
print("cache warmed for", MODEL)
```

---

## 7. Fast end-to-end smoke test (do this before the real run)

Run all three systems on the **small `gpt2` model** for a few steps to confirm the
whole pipeline works on your session, before spending time on `gpt2-xl`:

```python
# === Cell 6: 2-minute smoke test on gpt2 (small) ===
!python   src/train_singlegpu.py --model gpt2 --grad-checkpointing --optim adamw8bit \
          --micro-batch-size 2 --num-micro-batches 4 --seq-len 256 --max-steps 10
!torchrun --standalone --nproc_per_node=2 src/train_torchpp.py --model gpt2 \
          --schedule 1f1b --micro-batch-size 2 --num-micro-batches 4 --seq-len 256 --max-steps 10
!deepspeed --num_gpus 2 src/train_deepspeed.py --model gpt2 \
          --micro-batch-size 2 --num-micro-batches 4 --seq-len 256 --max-steps 10
```

Each prints a `tok/s ... peak_mem ... -> results/results.csv` line. If all three finish,
you're ready for the real model.

---

## 8. The real runs (`gpt2-xl`)

> **Memory rule of thumb on T4:** start with `--micro-batch-size 1 --seq-len 512`.
> If you hit CUDA out-of-memory, first drop `--seq-len 256`, then try `--optim adafactor`.

### 8a. Single-GPU baseline (deliverable 4 control)

```python
!python src/train_singlegpu.py --model gpt2-xl --grad-checkpointing --optim adamw8bit \
        --micro-batch-size 1 --num-micro-batches 8 --seq-len 512 --max-steps 40 --eval
```

### 8b. torch.distributed.pipelining — one schedule

```python
!torchrun --standalone --nproc_per_node=2 src/train_torchpp.py --model gpt2-xl \
        --schedule 1f1b --num-micro-batches 8 --micro-batch-size 1 --seq-len 512 --max-steps 40
```
Valid `--schedule` values: `gpipe`, `1f1b`, `interleaved_1f1b`, `interleaved_zb`, `zbv`
(the last two need torch ≥ 2.6 — Cell 3 told you if you have it).

### 8c. DeepSpeed Pipeline Parallelism (deliverable 2)

```python
!deepspeed --num_gpus 2 src/train_deepspeed.py --model gpt2-xl \
        --num-micro-batches 8 --micro-batch-size 1 --seq-len 512 --max-steps 40 --eval
```

---

## 9. Run the full sweep (all experiments at once)

This runs the correctness check, the baseline, the micro-batch sweep for both
frameworks, and the schedule ladder — then you aggregate and plot.

```python
# === Cell 7: full sweep (≈30–60 min on gpt2-xl; set STEPS lower to go faster) ===
!MODEL=gpt2-xl SEQ=512 STEPS=30 MBS=1 bash bench/run_sweep.sh
```

```python
# === Cell 8: aggregate + plot ===
!python bench/aggregate_results.py
!python bench/plots.py
```

```python
# === Cell 9: show the figures inline ===
from IPython.display import Image, display
import glob
for png in sorted(glob.glob("results/figures/*.png")):
    print(png); display(Image(png))
```

```python
# === Cell 10: peek at the raw results table ===
import pandas as pd
pd.read_csv("results/results.csv")
```

Everything is written under `/kaggle/working/results/` (CSV, per-run JSON, figures) and
is saved with the notebook output — download it via the notebook's **Output** tab, or
**Save Version** to persist it.

---

## 10. How the outputs map to the 4 assignment deliverables

| Deliverable | Where to look |
|---|---|
| 1. Fine-tune a model too big for 1 GPU with pipeline parallelism | any `torchpp`/`deepspeed` run completing on `gpt2-xl` (loss ↓), `peak_mem_gb_max` per GPU ≈ ½ the model |
| 2. DeepSpeed PP **vs** PyTorch-native pipeline | `system_comparison.png`; rows `system=deepspeed` vs `system=torchpp` in `results.csv` |
| 3. Bubble analysis + effect of **micro-batch count** | `throughput_vs_M.png`, `bubble_vs_M.png`, `memory_vs_M.png`; column `bubble_theory` |
| 4. 2-GPU pipeline **vs** 1-GPU gradient checkpointing | `system_comparison.png`; compare `system=singlegpu` to the pipeline rows |

The **schedule ladder** (`schedule_ladder.png`, `interleaved_*`/`zbv` rows) is the
modern-technique extension — it shows how Interleaved and Zero-Bubble schedules shrink
the bubble beyond classic GPipe/1F1B.

**Expected headline finding (state this in the report):** at only P=2 over PCIe, the
2-GPU pipeline is **not** ~2× faster than 1 GPU — its value is **making the model
trainable** (half the weights/optimizer per GPU) and allowing a larger batch, i.e.
pipeline at this scale is *memory-scaling, not speed-scaling*. See `PROJECT_PLAN.md` §7.

---

## 11. What each file does (quick map)

```
src/
  data.py            wikitext -> tokenize -> fixed 512-token blocks (shared by all 3)
  losses.py          causal-LM loss with shift, computed in fp32
  model_split.py     gpt2-xl/OPT -> ordered list of single-tensor stages (the hard part)
  check_split.py     proves the split == the original model (run this first!)
  metrics.py         throughput / peak VRAM / bubble / MFU / CSV logger
  optim.py           adamw | adamw8bit | adafactor factory (use the same one everywhere)
  train_singlegpu.py 1-GPU grad-checkpointing baseline
  train_torchpp.py   torch.distributed.pipelining (GPipe..Zero-Bubble)
  train_deepspeed.py DeepSpeed PipelineModule (1F1B)
bench/
  run_sweep.sh       runs the whole experiment matrix
  aggregate_results.py  prints summary tables -> results/summary.csv
  plots.py           writes results/figures/*.png
configs/ds_pp.json   canonical DeepSpeed settings (CLI overrides micro-batch + M)
results/             CSV + per-run JSON + figures (created at runtime)
```

Run everything **from the project root** (`/kaggle/working/project`) so the `src/...`
paths resolve.

---

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | `--micro-batch-size 1`; then `--seq-len 256`; then `--optim adafactor`. For DeepSpeed add `--activation-checkpoint`. GPipe OOMs sooner than 1F1B at large M (that's an expected result — note it). |
| `GPUs visible: 1` | Accelerator isn't "GPU T4 x2". Fix in the right panel and restart the session. |
| `schedule 'zbv' unavailable` / zero-bubble missing | Your torch < 2.6. Use `gpipe`/`1f1b`/`interleaved_1f1b`; report the version limitation. |
| `import deepspeed` warns about ops / async_io / libaio | Harmless — pipeline parallelism needs no compiled kernels. |
| `bitsandbytes` import error | Use `--optim adafactor` everywhere instead. |
| DeepSpeed `Address already in use` | Re-run, or add `--master_port 29501` to the `deepspeed` command. |
| Download/network errors | Confirm **Internet = On**; re-run the warm-cache cell (Cell 5). |
| Two processes both downloading / hang at start | You skipped Cell 5 — run it to warm the cache before `torchrun`/`deepspeed`. |
| Hang at the end of a `torchrun` run | Usually a crashed rank; scroll up for the real traceback (often OOM on one stage). |
| Session timed out mid-sweep | Lower `STEPS` (e.g. `STEPS=15`) and/or run sections 8a–8c separately. Results accumulate in `results.csv` across runs. |

---

## 13. Reproducibility checklist (for the report)

- Same `--model`, `--seq-len`, **global batch** (`micro_batch_size × num_micro_batches`),
  `--optim`, `--seed`, and `--max-steps` across all three systems — otherwise the
  comparison is meaningless. The defaults already line these up (global batch = 8).
- Record the `torch_version` column (it's logged automatically) and note whether
  zero-bubble schedules were available on your image.
- For a convergence (perplexity) comparison, run `--eval` on the single-GPU and
  DeepSpeed paths and use the **same optimizer** on all systems; for `torchpp` add
  `--loss-scale 1024` for fp16 gradient stability.
```
