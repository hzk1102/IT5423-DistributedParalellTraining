# Project Plan — Topic 2: Pipeline Parallelism for Models Exceeding GPU Memory

> Master-level project, GenAI Master / Distributed & Parallel Training.
> Status: **planning only, no code yet.** This plan supersedes `research_plan.md`
> (which is kept as a generic literature checklist; several of its assumptions are
> wrong for our hardware — see §10).

---

## 1. What the assignment actually asks (from `project-20252.pdf`, Topic 2)

Four mandatory deliverables:

1. Fine-tune a model that **cannot fully fit on one GPU** (suggested: `facebook/opt-6.7b`
   or `gpt2-xl`) using **pipeline parallelism** on a **2-GPU** environment.
2. **Compare** DeepSpeed Pipeline Parallelism **vs** the PyTorch-native pipeline solution
   (the PDF says `torch.distributed.pipeline` — that API is now deprecated; see §4).
3. **Analyze the pipeline "bubble"** and the **effect of the number of micro-batches** on
   performance.
4. **Compare 2-GPU pipeline vs 1-GPU with gradient checkpointing** (throughput + memory).

The graded artifact is a **comparative benchmark study** — the measurements, their analysis,
and the framing matter as much as a working training loop.

---

## 2. Binding hardware constraints (Kaggle 2× T4)

These dictate every downstream decision:

| Constraint | Value | Consequence |
|---|---|---|
| GPUs | 2× NVIDIA T4, 16 GB each (32 GB total) | Model+optimizer+grads must fit in ~30 GB usable |
| Interconnect | **PCIe, no NVLink** | Pipeline P2P activation transfers are slow → comm overhead is *real and measurable* (good for the study) |
| Arch | Turing, **SM 7.5** | **No bf16 hardware** → use **fp16** + loss scaling. **No FlashAttention-2** (needs SM80+) → use PyTorch SDPA mem-efficient backend |
| Runtime | Kaggle session ~9–12 h, can stop | Keep each experiment short; checkpoint; log everything |
| Dev vs run | Dev locally (Windows + `uv`), **run on Kaggle Linux** | Launch scripts must be portable; no Windows-only assumptions in training code |

**Key implication:** at only **P = 2 pipeline stages over PCIe**, pipeline parallelism is
primarily a **memory-scaling** technique, not a throughput-scaling one. The honest expected
finding (see §7) is that 2-GPU pipeline will *not* be ~2× faster than 1 GPU — it makes a model
*trainable* that otherwise wouldn't fit. Framing the report around this is what makes it strong.

---

## 3. Model & dataset selection (with the memory math)

Full fine-tuning with Adam costs ≈ **16 bytes/param**:
fp16 weight (2) + fp16 grad (2) + fp32 master (4) + Adam m,v fp32 (4+4).

| Model | Params | FP16 weights | Full-Adam total (16 B/p) | Fits 1×T4 (16 GB)? | Fits 2×T4 pipeline (per-GPU ≈ ½)? |
|---|---|---|---|---|---|
| `distilgpt2` | 82 M | 0.16 GB | 1.3 GB | yes (too small to be interesting) | trivial |
| `facebook/opt-1.3b` | 1.3 B | 2.6 GB | 21 GB | **no** | ~10.5 GB/GPU → yes |
| **`gpt2-xl`** | **1.5 B** | **3.0 GB** | **24 GB** | **no** | **~12 GB/GPU → yes** ✅ |
| `facebook/opt-6.7b` | 6.7 B | 13.4 GB | 107 GB | no | ~53 GB/GPU → **NO** ❌ |

**Recommendation: `gpt2-xl` (1.5 B) as the primary model.** It is the sweet spot:
- Full-Adam fine-tuning (24 GB) **exceeds a single 16 GB T4** → satisfies "doesn't fit on 1 GPU."
- Splits cleanly to ~12 GB/GPU across a 2-stage pipeline → comfortably trains on 2× T4.
- The 1-GPU baseline becomes feasible *only* with memory tricks (gradient checkpointing
  + 8-bit Adam / Adafactor) → this **is exactly the comparison the assignment wants**.

**Do not use `opt-6.7b`:** full fine-tuning needs ~107 GB; even with 8-bit Adam (~10 B/p → 67 GB)
it is ~33 GB/GPU on 2 GPUs — over budget. It is only reachable via QLoRA/ZeRO-offload, which
*dilutes the pipeline study* (frozen 4-bit weights, tiny optimizer state). **Include the
infeasibility analysis in the report** — it demonstrates understanding of the memory model.

- **Secondary/sanity model:** `opt-1.3b` (24 layers, hidden 2048) for fast iteration and to show
  the method generalizes across architectures (GPT-2 vs OPT). Both `gpt2-xl` (48 layers, hidden
  1600) and `opt-1.3b` were **verified callable** from the Hub (config loads).

### Dataset — verified loadable (this was tested, not assumed)

**Primary: `Salesforce/wikitext`** — the standard LM perplexity benchmark, served as parquet,
**no auth / no `trust_remote_code` / no gating**. Verified locally:

```python
from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
# -> train 36,718 / validation 3,760 / test 4,358 rows  (✓ loaded)
# configs available: wikitext-2-raw-v1, wikitext-2-v1, wikitext-103-raw-v1, wikitext-103-v1
```

> Note: the bare `"wikitext"` id is the old/deprecated path — use **`Salesforce/wikitext`**.
> Use `wikitext-2-raw-v1` for the fast micro-batch/schedule sweeps and
> `wikitext-103-raw-v1` (~100× larger) for the longer convergence run. `-raw-` keeps real
> text (no `<unk>` preprocessing) → tokenize with the model's own BPE tokenizer.

**Modern alternative (optional): `HuggingFaceFW/fineweb-edu`** (2024, very current web-text
corpus). Verified reachable via **streaming** (the full set is TB-scale — never download it):

```python
ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
# first record loaded ✓ (fields: text, id, url, score, token_count, ...)
```

Recommendation: keep **wikitext-103-raw-v1** as the headline dataset (comparable to published
perplexity numbers, fully reproducible, tiny), and mention fineweb-edu streaming as a "modern
corpus" extension. For a *pipeline-parallelism systems study* the dataset is largely orthogonal —
reliability and reproducibility beat novelty here.

- **Objective:** causal LM. Fixed sequence length **512** primary; sweep 256/1024 as a secondary
  axis (activation-memory pressure). Pack/chunk concatenated tokens to fixed length; model's own tokenizer.

---

## 4. The three systems under comparison

### A. DeepSpeed Pipeline Parallelism
- `deepspeed.pipe.PipelineModule` built from a **`LayerSpec` list** (delays construction so each
  rank allocates only its layers). HF GPT-2 must be **decomposed into sequential stages**:
  `[embedding(wte+wpe)] → [block_0 … block_N] → [final_ln + lm_head]`, with a tied **`loss_fn`**
  that does the causal shift. Only `hidden_states` is passed stage-to-stage (GPT-2 blocks build
  their own causal mask internally), which keeps the pipeline plumbing simple.
- Schedule: DeepSpeed uses **1F1B** by default; `num_micro_batches` is the swept knob.
- fp16 via DeepSpeed config (`"fp16": {"enabled": true}`).

### B. PyTorch-native pipeline — **`torch.distributed.pipelining`** (NOT `torch.distributed.pipeline`)
- ⚠️ **Correction:** the PDF's `torch.distributed.pipeline.sync.Pipe` (torchgpipe) is
  **deprecated and superseded**. The current API is **`torch.distributed.pipelining`** (the PiPPy
  project, upstreamed into PyTorch ≥2.4). It provides `GPipe`, `1F1B`, `Interleaved1F1B`,
  `LoopedBFS` schedules. Verify the Kaggle torch version first; if it ships the legacy module,
  document the deprecation and use the new one.
- Approach: **manual stage split** (split the `nn.Module` into per-stage submodules + wrap with
  `PipelineStage`) is more robust for HF models than the tracing/`pipeline()` path (HF forward has
  dynamic control flow that can break the tracer).
- **Schedule suite to compare — this is the "newest technique" axis.** `torch.distributed.pipelining`
  exposes the full modern ladder; we benchmark across it:
  - `ScheduleGPipe` — classic all-forward-then-all-backward (baseline).
  - `Schedule1F1B` — one-forward-one-backward (same bubble as GPipe, lower activation memory).
  - `ScheduleInterleaved1F1B` — virtual stages per rank, **smaller bubble**.
  - `ScheduleInterleavedZeroBubble` / `ScheduleZBVZeroBubble` — **Zero-Bubble (2024, ICLR)**: splits
    the backward into input-grad (B) and weight-grad (W) and uses W to fill the bubble → bubble → ~0.
    `ScheduleZBVZeroBubble` requires exactly 2 stages per rank (with 2 ranks ⇒ 4 model chunks).
  - Requires **torch ≥ 2.6** (zero-bubble schedules landed in 2.6) — verify Kaggle's torch version.

### C. Single-GPU baseline (the control)
- Same `gpt2-xl`, 1× T4, **`model.gradient_checkpointing_enable()`** + a memory-efficient
  optimizer (8-bit Adam via `bitsandbytes`, or `Adafactor`) to make it fit at all.
- This isolates "what does adding the 2nd GPU via pipelining buy us."

All three share **identical**: dataset, tokenizer, seq len, effective global batch size, seed,
fp16 setting, and number of optimizer steps — otherwise the comparison is meaningless.

---

## 5. Experimental matrix

1. **Micro-batch sweep** (core of deliverable 3): M ∈ {1, 2, 4, 8, 16, 32} for both DeepSpeed
   and torch.pipelining at fixed global batch. Measure throughput, peak VRAM, bubble.
2. **Schedule comparison (the modern-technique axis):** GPipe → 1F1B → Interleaved1F1B →
   **Zero-Bubble (ZBV)**. Expectation: 1F1B ≈ same bubble as GPipe but **lower peak activation
   memory**; interleaved lowers the bubble; zero-bubble drives it toward ~0 at the cost of more
   scheduling complexity and W-grad bookkeeping. Empirically validating this ladder on 2× T4 is the
   project's most current contribution.
3. **2-GPU pipeline vs 1-GPU grad-checkpointing** (deliverable 4): best pipeline config vs the
   baseline — throughput, peak VRAM/GPU, time-to-fixed-loss.
4. **Framework comparison** (deliverable 2): DeepSpeed vs torch.pipelining at matched configs —
   throughput, memory, *and developer ergonomics* (LOC, conversion effort, stability).
5. **Stage-balancing** (secondary): naive even-layer split vs parameter/FLOP-balanced split;
   show effect on the slowest stage (pipelines run at the speed of the slowest stage).
6. **Seq-len axis** (secondary): 256/512/1024 to show activation-memory pressure.

---

## 6. Metrics & instrumentation

- **Throughput:** tokens/sec (primary) and samples/sec, steady-state (drop warm-up steps).
- **Peak VRAM per GPU:** `torch.cuda.max_memory_allocated()` per rank.
- **Pipeline bubble fraction:** *theoretical* (formula §7) **and measured** (idle/forward-backward
  time from the PyTorch profiler / NVTX ranges or DeepSpeed timers).
- **Communication:** P2P activation transfer time/volume between stages (profiler).
- **MFU (Model FLOPs Utilization):** achieved FLOP/s ÷ T4 peak fp16 (~65 TFLOP/s) — strong,
  master-level normalized efficiency metric.
- **Convergence:** training loss + **eval perplexity** on wikitext, plotted vs **wall-clock** and
  vs **optimizer steps** (both axes — steps shows algorithmic equivalence, wall-clock shows system cost).
- **Reproducibility:** fixed seeds, logged configs/versions, deterministic data order.

Log to CSV/JSON + a Kaggle-friendly notebook that regenerates all plots.

---

## 7. The bubble — theory to validate

For a synchronous schedule with **P** stages and **M** micro-batches the idle fraction is:

```
bubble_fraction = (P - 1) / (M + P - 1)
```

With **P = 2** this is **1 / (M + 1)**:

| M | bubble |
|---|---|
| 1 | 50% |
| 2 | 33% |
| 4 | 20% |
| 8 | 11% |
| 16 | 5.9% |

**Hypotheses to test:** throughput rises with M and asymptotes as the bubble shrinks, but (a)
GPipe activation memory grows ∝ M (eventually OOM) while 1F1B caps it, and (b) PCIe P2P overhead
and per-microbatch launch cost erode the gains at large M. **Interleaved 1F1B** with `v` virtual
stages per device cuts the bubble by ≈ `1/v` at the cost of more comm — worth one data point.

**The full bubble ladder we're validating (newest-technique framing):**

| Schedule | Bubble (idealized) | Trade-off |
|---|---|---|
| GPipe / 1F1B | `(P-1)/M` | simplest; 1F1B lowers *memory* not bubble |
| Interleaved 1F1B (`v` chunks/rank) | `(1/v)·(P-1)/M` | smaller bubble, more P2P comm |
| **Zero-Bubble (ZB/ZBV, 2024)** | **≈ 0** | split backward into B (input-grad) + W (weight-grad), W fills the bubble |
| DualPipe (DeepSeek-V3, 2025) | `(P/2−1)(β+γ)` | bidirectional full overlap; built for large-scale MoE/EP |

**DualPipe** (DeepSeek-V3, Feb 2025) is the current SOTA and worth citing as related work, but it
targets cross-node expert-parallel MoE training with full comp/comm overlap — **out of scope to
implement** on 2× dense-GPT-2 T4s. Include it in the discussion to show awareness of the frontier;
**implement and measure up to Zero-Bubble**, which is feasible here via `torch.distributed.pipelining`.

**Expected headline result:** at P=2 over PCIe, the 1-GPU grad-checkpointing baseline may match or
beat 2-GPU pipeline raw throughput (no bubble, no P2P), but cannot fit the model without heavy
memory tricks / a smaller batch — so pipeline's value is **enabling the model + larger batch**, not
speed. Stating and quantifying this is the report's core contribution.

---

## 8. Implementation plan (structure for when coding starts — not yet)

```
Project/
  configs/            # ds_config_*.json, run configs (yaml)
  src/
    data.py           # wikitext load + tokenize + fixed-len chunking
    model_split.py    # HF GPT-2/OPT → ordered stage list (shared by DeepSpeed & torch.pipelining)
    train_deepspeed.py
    train_torchpp.py
    train_singlegpu.py # grad-checkpointing baseline
    metrics.py        # throughput, peak mem, bubble, MFU helpers
  bench/
    run_microbatch_sweep.sh
    aggregate_results.py
    plots.ipynb
  results/            # csv/json + figures
  PROJECT_PLAN.md
  CLAUDE.md
```

Launch:
```bash
deepspeed --num_gpus 2 src/train_deepspeed.py --deepspeed configs/ds_pp.json --micro-batches 8
torchrun --nproc_per_node=2 src/train_torchpp.py --schedule 1f1b --micro-batches 8
python src/train_singlegpu.py --grad-checkpointing --optim adamw8bit
```

**Build order:** (1) data + single-GPU baseline (proves the loop/metrics), (2) `model_split.py`
shared decomposition, (3) torch.pipelining path, (4) DeepSpeed path, (5) sweeps + plots, (6) report.

---

## 9. Risks & pitfalls (anticipate these)

- **HF → sequential decomposition** for DeepSpeed `PipelineModule` is the hardest part; the
  causal-shift loss must live in the pipeline `loss_fn`. Budget time here.
- **Deprecated API trap:** don't call `torch.distributed.pipeline`; use `torch.distributed.pipelining`.
- **T4 limits:** fp16 only (bf16 will silently fall back or error); no FlashAttention-2.
- **OOM at large M with GPipe** (activations ∝ M) — switch to 1F1B / lower M.
- **Stage imbalance** (embedding+head are heavy) inflates the bubble — balance by params/FLOPs.
- **Apples-to-apples:** identical global batch, steps, seq len, seed across all three systems.
- **Kaggle:** version drift (torch/DeepSpeed), session timeouts → checkpoint + keep runs short.

---

## 10. Corrections to `research_plan.md`

Keep it as a background literature checklist, but for *this* project:
- **Drop FlashAttention** as a method — FA2 needs SM80+, T4 is SM75. Use SDPA mem-efficient backend.
- **bf16 → fp16** everywhere (T4 has no bf16 hardware).
- The "Project Design Options" (FSDP-vs-DDP, ZeRO, QLoRA, MoE) belong to *other* assignment topics —
  we are locked to **Topic 2 (pipeline)**. Use them only as comparative background, not as the build.
- TorchCompile / CUDA graphs / Liger Kernel: optional extensions only; fragile in combination with
  pipeline + DeepSpeed on T4 — do not put them on the critical path.

**What was modernized vs the original hints:** the original plan stopped at GPipe/1F1B. This plan
extends the comparison up the current scheduling ladder — **Interleaved 1F1B and Zero-Bubble (2024)
schedules** via `torch.distributed.pipelining`, with **DualPipe (DeepSeek-V3, 2025)** cited as
frontier related work — and uses the **current `Salesforce/wikitext`** dataset path plus an optional
**FineWeb-Edu (2024)** modern corpus. Both datasets and both models were verified callable.

---

## 11. Milestones

1. **M1 — Foundation:** uv env, data pipeline, single-GPU baseline + metrics harness working on Kaggle.
2. **M2 — Pipelines:** `model_split.py`; torch.pipelining run; DeepSpeed PP run; correctness (loss ↓) matches baseline.
3. **M3 — Experiments:** micro-batch sweep, schedule comparison, 1-GPU vs 2-GPU, framework comparison.
4. **M4 — Analysis & report:** bubble theory vs measured, plots, MFU, ergonomics discussion, write-up.

## 12. Final deliverables

- Reproducible code (3 training paths + shared split + bench scripts) runnable on Kaggle 2× T4.
- Results CSVs + plots: throughput vs M, bubble (theory vs measured), memory, perplexity vs
  wall-clock & steps, framework comparison.
- Written report covering all four assignment deliverables, the OPT-6.7B infeasibility analysis,
  and the "pipeline = memory-scaling, not speed-scaling at P=2/PCIe" conclusion.
