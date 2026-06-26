# How to Build the Presentation Slides — Topic 2 (Pipeline Parallelism)

> A guide for turning `REPORT.md` / the DOCX into a defensible ~12–15 minute talk.
> Written in a strict-grading voice: every slide must **earn its place** by advancing
> one logical step. If a slide doesn't move the argument forward, cut it.

---

## 0. The one rule

**A presentation is an argument, not a data dump.** Your job is to walk the audience
from *"big models don't fit on one GPU"* to *"here is exactly what pipeline parallelism
buys you on 2× T4, measured and honest."* Every slide is one link in that chain. If you
can't say in one sentence *why a slide exists*, delete it.

The spine of the argument (memorize this — it is also the order of the deck):

> **Problem → Why it's hard → Our approach → What we measured → What it means → Honest limits → Takeaway.**

Each of the assignment's four deliverables must be visibly answered. Map them as you go:
- D1 Enablement, D2 Framework comparison, D3 Bubble & micro-batches, D4 2-GPU vs 1-GPU.

---

## 1. Audience, time, format

- **Audience:** a mixed room — the grader knows distributed training; some classmates do
  not. Open accessible, then earn depth. Do **not** assume everyone knows "the bubble."
- **Time:** assume **12–15 min talk + Q&A**. That is **~13–15 slides**. ~1 min/slide.
  Going over time reads as poor preparation — rehearse with a timer.
- **One idea per slide. One figure per slide.** Never two plots competing for attention.
- **Slides carry images and ≤ 6 lines of text; your voice carries the sentences.** A slide
  full of paragraphs means you will read it, and reading slides loses the room.

---

## 2. Slide-by-slide blueprint

Each entry: **[slide title]** — *purpose (why it exists)* — what to put on it — the
figure/table to use — the one line you must say out loud.

### Slide 1 — Title
*Purpose: set the topic and credibility.*
- Project title, "Topic 2: Pipeline Parallelism", team names, course, date.
- Say: *"We study how to train a model too big for one GPU by splitting it across two."*

### Slide 2 — The problem (the hook)
*Purpose: make the audience feel why this matters in 30 seconds.*
- One visual: the memory math. "gpt2-xl needs ~24 GB to train; one T4 has 16 GB."
- A 3-row mini table: model → memory needed → fits one T4? (No.)
- Say: *"Full-Adam training costs ~16 bytes per parameter — 1.5B params overflows a 16 GB card before we even start."*

### Slide 3 — The assignment, as four questions
*Purpose: tell the grader you know exactly what you must deliver.*
- List D1–D4 in plain language (enable, compare frameworks, bubble, 2-vs-1).
- Say: *"Everything that follows answers one of these four."* (Reference them later by number.)

### Slide 4 — How pipeline parallelism works (the analogy)
*Purpose: the single most important teaching slide — get the mental model in.*
- The **assembly-line** picture: GPU1 = first half of layers, GPU2 = second half;
  micro-batches flow through like cars down a line.
- A simple 2-stage diagram (draw it; arrows for micro-batches).
- Say: *"While GPU2 works on micro-batch 1, GPU1 already starts micro-batch 2 — that overlap is the whole point."*

### Slide 5 — The bubble (the core concept of D3)
*Purpose: define the one metric the whole study revolves around.*
- The idle-time picture (start-up fill + drain), the formula `(P−1)/(M+P−1)`, and the
  P=2 table (50% → 5.9% as M goes 1 → 16).
- Say: *"Two stages means the first and last micro-batches always leave a GPU idle; more micro-batches dilute that idle time."*

### Slide 6 — Setup (so results are trustworthy)
*Purpose: pre-empt "is this a fair comparison?"*
- Hardware (2× T4, **PCIe / no NVLink** — flag this, it explains later results), model,
  dataset, and the "same data/seed/batch/steps for all three systems" fairness line.
- Say: *"The only thing we change between systems is how the model sits on the GPUs."*

### Slide 7 — D1: Enablement (first result, the clearest win)
*Purpose: a concrete, undeniable success to open the results.*
- Table: opt-2.7b → one GPU = **OOM**; two GPUs = **trains, 10.6 GB/GPU, 1,701 tok/s**.
- Say: *"A model that crashes one GPU trains fine across two — this is what pipeline parallelism is for."*

### Slide 8 — D3: Micro-batches shrink the bubble
*Purpose: show theory meets measurement.*
- Figure: `throughput_vs_M.png` **or** `bubble_empirical.png` (pick ONE — I'd use the
  bubble one, it's the more sophisticated claim).
- Say: *"Throughput climbs as predicted; the measured bubble sits just above theory — the gap is the PCIe transfer cost."*

### Slide 9 — D3: 1F1B vs GPipe is a memory story
*Purpose: the cleanest, most quotable finding.*
- Figure: `memory_vs_M.png` (1F1B flat; GPipe climbs and OOMs at M=8).
- Say: *"Same bubble, very different memory — GPipe runs out at M=8 while 1F1B stays flat. That's why 1F1B is the default."*

### Slide 10 — D3: Does the newest schedule win? (honesty + nuance)
*Purpose: show critical thinking, not hype.*
- Figure: `schedule_ladder.png` + the +0.1% / +6.2% numbers.
- Say: *"Zero-bubble is the 2024 state of the art, but on our slow PCIe link it buys only ~6% — the technique is interconnect-bound."* (Graders love an honest negative result.)

### Slide 11 — D2: DeepSpeed vs PyTorch
*Purpose: the framework verdict.*
- Figure: `system_comparison.png` + a 3-row table (tok/s, GB/GPU, schedules/automation).
- Say: *"PyTorch was faster and leaner here; DeepSpeed repays that with automation — loss scaling and partitioning for free."*

### Slide 12 — D4: Two GPUs vs one (with the honesty note)
*Purpose: answer D4 AND demonstrate scientific integrity.*
- Table: 1 GPU (~1,221 corrected) vs 2-GPU pipeline (1,835). Include a one-line
  "honesty box": *we found a metering bug, corrected it exactly ×8, and the residual
  caveat biases against us.*
- Say: *"~1.5× faster and less memory per GPU — and we're transparent about the one number we had to correct."*
- **This slide raises your grade.** Hiding the bug and getting caught in Q&A would sink it.

### Slide 13 — Conclusion (the takeaway sentence)
*Purpose: leave one idea in their heads.*
- 3 bullets max, then the headline.
- Say: *"At two GPUs over PCIe, pipeline parallelism is mainly a way to fit bigger models — a memory win and a modest ~1.5× speed win, not a magic 2×."*

### Slide 14 — (Optional) Limitations & future work
*Purpose: shows you know the edges of your own claims.*
- Bubble-sweep batch confound; no perplexity for the PyTorch path; P=2 only; short runs.
- Say: *"These don't change the conclusions, but a fixed-batch sweep and a longer run would sharpen them."*

### Backup slides (after the "Thank you" — for Q&A only)
- `convergence_vs_tokens.png` (the fair convergence view).
- The full results table from the report appendix.
- The memory-math table for opt-6.7b infeasibility.

---

## 3. Which figure goes where (don't overload)

| Slide | Figure | Why this one |
|---|---|---|
| 8 | `bubble_empirical.png` | measured vs theory — your strongest D3 evidence |
| 9 | `memory_vs_M.png` | the GPipe-OOM-vs-1F1B-flat contrast |
| 10 | `schedule_ladder.png` | the honest "newest ≠ much faster" |
| 11 | `system_comparison.png` | the framework verdict at a glance |
| 12 | (table, not a plot) | numbers + the honesty note |
| backup | `convergence_vs_tokens.png`, `throughput_vs_M.png` | for follow-up questions |

Rule: a figure you show, you must be able to **read aloud in one sentence** (axes + the
one trend). If you can't, it's a backup slide, not a main slide.

---

## 4. Design rules (strict)

- **Font ≥ 24 pt** for body, ≥ 32 pt for titles. If it doesn't fit, you have too much text.
- **Label every axis** and state units (tokens/sec, GB). Unlabeled plots lose marks.
- **One color = one system** across all slides (e.g. PyTorch = green, DeepSpeed = blue,
  1 GPU = orange — match the figures). Consistency = professionalism.
- **No screenshots of code** unless it makes a specific point (e.g. the causal-mask fix).
- **Numbers get units and context:** "1,835 tok/s" not "1835"; "+6.2% over 1F1B" not "1,949".
- Round in speech: say "about 1,800 tokens a second," show the exact number on the slide.

---

## 5. Rehearsal & defense (anticipate the grader)

Prepare a crisp answer to each — these are the questions a strict examiner *will* ask:

- **"Why is the pipeline not 2× faster with 2 GPUs?"** → the bubble (P=2 is the worst case
  for it) plus PCIe transfer overhead; quantify with the measured-vs-theory gap.
- **"Your single-GPU number looked wrong at first — explain."** → the batching bug; the
  exact ×8 correction; why the residual caveat biases *against* the pipeline.
- **"Why did zero-bubble barely help?"** → no NVLink; the split-backward work can't be
  hidden behind slow PCIe transfers. It shines on datacentre interconnects.
- **"Is the comparison fair?"** → same data/seed/batch/steps; the one caveat (batch grows
  with M in the sweep) is disclosed and doesn't affect the fixed-M=8 comparisons.
- **"Why gpt2-xl and not opt-6.7b?"** → memory math: 6.7B needs ~107 GB even split — over
  budget; gpt2-xl is the model that *just* doesn't fit one GPU, which is the point.

Rehearse the full talk **out loud, twice, against a timer.** Decide who presents which
slides and hand off explicitly. The person on the conclusion slide owns the takeaway
sentence — practice it until it's automatic.

---

## 6. Common mistakes that cost marks

1. **Reading the slides.** Slides are visual aids; you narrate.
2. **A wall of text.** If a slide has a paragraph, split or cut it.
3. **Hiding the limitation.** Owning the bug is a strength; getting caught hiding it is fatal.
4. **Claiming a 2× speed-up.** The data says ~1.5×; overclaiming invites a takedown.
5. **An unlabeled or unexplained figure.** Never show a plot you can't narrate in one line.
6. **Running over time.** Cut backup content from the main flow; rehearse to the clock.
7. **No clear "so what."** End every results slide with the one-line implication, not just the number.

---

> **Build order:** write the **takeaway sentence first** (Slide 13), then assemble the slides
> that earn their way to it. If a slide doesn't help you say that sentence, it's a backup.
