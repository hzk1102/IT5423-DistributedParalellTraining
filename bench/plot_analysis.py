"""Extra report figures that turn the weak spots into evidence.

    python bench/plot_analysis.py \
        [--csv results/results.csv] [--loss-csv results/loss_curves.csv] \
        [--out results/figures] [--model gpt2-xl] [--seq-len 512]

Produces:
  1. bubble_empirical.png  -- MEASURED bubble (derived from throughput scaling)
     overlaid on the theoretical (P-1)/(M+P-1). This is the figure that answers
     "analyze the bubble": we don't just plot a formula, we validate it against
     observed tokens/sec.
  2. convergence.png       -- training loss vs step for the three systems on the
     same axis (needs runs launched with --loss-log results/loss_curves.csv).

Why the empirical bubble is defensible
--------------------------------------
If a pipeline had no bubble, per-step time would be M-independent and throughput
would be flat in M. The bubble inflates per-step time by 1/(1-b(M)), so
    T(M) = T_ideal * (1 - b(M)).
We estimate the bubble-free rate T_ideal from the largest-M run (where the
theoretical bubble is smallest), then report the *measured* bubble
    b_emp(M) = 1 - T(M) / T_ideal
and compare it to theory. Tracking curves => the bubble model is validated.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path)


def _filter(df, model, seq_len):
    if model and "model" in df.columns:
        df = df[df["model"] == model]
    if seq_len and "seq_len" in df.columns:
        df = df[df["seq_len"] == seq_len]
    return df


def plot_bubble_empirical(df, out_dir):
    """Measured-vs-theoretical bubble for every 2-GPU schedule with >=2 M points."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    plotted = False
    # theory curve for P=2 over a smooth M range
    Ms = sorted(df["num_micro_batches"].unique())
    if Ms:
        theory = [(2 - 1) / (m + 2 - 1) * 100 for m in Ms]
        ax.plot(Ms, theory, "k--", lw=2, label="theory  (P-1)/(M+P-1),  P=2")

    pp = df[df["world_size"] == 2] if "world_size" in df.columns else df
    for (system, schedule), g in pp.groupby(["system", "schedule"]):
        g = g.sort_values("num_micro_batches")
        g = g.drop_duplicates("num_micro_batches")
        if g["num_micro_batches"].nunique() < 2:
            continue
        m_max = g["num_micro_batches"].max()
        t_at_max = g.loc[g["num_micro_batches"] == m_max, "tokens_per_sec"].iloc[0]
        b_max = (2 - 1) / (m_max + 2 - 1)              # theoretical bubble at M_max
        t_ideal = t_at_max / (1 - b_max)               # extrapolated bubble-free rate
        g = g.assign(b_emp=100 * (1 - g["tokens_per_sec"] / t_ideal))
        ax.plot(g["num_micro_batches"], g["b_emp"], marker="o",
                label=f"measured: {system}/{schedule}")
        plotted = True

    ax.set_xlabel("micro-batches M"); ax.set_ylabel("pipeline bubble (%)")
    ax.set_title("Pipeline bubble: measured (throughput-derived) vs theory")
    ax.set_xscale("log", base=2); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    ax.axhline(0, color="grey", lw=0.6)
    if plotted:
        _save(fig, out_dir, "bubble_empirical.png")
    else:
        plt.close(fig)
        print("[skip] bubble_empirical: need >=2 M points per schedule")


def _smooth(s, window=10):
    """Rolling mean so the noisy per-step loss reads as a trend, not a scribble."""
    return s.rolling(window=window, min_periods=1).mean()


def _nice_label(system, schedule):
    sysname = {"singlegpu": "1 GPU", "torchpp": "torch PP (2 GPU)",
               "deepspeed": "DeepSpeed (2 GPU)"}.get(str(system), str(system))
    return f"{sysname} / {schedule}"


def plot_convergence(loss_csv, out_dir):
    if not os.path.exists(loss_csv):
        print(f"[skip] convergence: {loss_csv} not found "
              f"(launch runs with --loss-log {loss_csv})")
        return
    df = pd.read_csv(loss_csv)
    if df.empty:
        print("[skip] convergence: loss CSV empty")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    keys = ["system", "schedule", "model", "num_micro_batches"]
    keys = [k for k in keys if k in df.columns]
    for vals, g in df.groupby(keys):
        g = g.sort_values("step")
        vals = vals if isinstance(vals, tuple) else (vals,)
        label = _nice_label(vals[0], vals[1] if len(vals) > 1 else "")
        line, = ax.plot(g["step"], _smooth(g["loss"]), label=label, lw=2)
        ax.plot(g["step"], g["loss"], color=line.get_color(), alpha=0.15)
    ax.set_xlabel("optimizer step"); ax.set_ylabel("training loss")
    ax.set_title("Convergence: training loss vs optimizer step")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_dir, "convergence.png")


def plot_convergence_vs_tokens(loss_csv, out_dir, seq_len=512, micro_batch_size=1):
    """Fair convergence view: loss vs the number of tokens actually processed.

    The single-GPU run had a batching bug (DataLoader batch = global_batch, then
    accumulated num_micro_batches of them), so per optimizer step it consumed
    num_micro_batches x more tokens than the 2-GPU pipeline runs. Plotting against
    *tokens seen* (not steps) puts all three systems on a common x-axis, removing
    that unfair advantage from the picture.
    """
    if not os.path.exists(loss_csv):
        print(f"[skip] convergence_vs_tokens: {loss_csv} not found")
        return
    df = pd.read_csv(loss_csv)
    if df.empty:
        print("[skip] convergence_vs_tokens: loss CSV empty")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    keys = [k for k in ["system", "schedule", "model", "num_micro_batches"] if k in df.columns]
    for vals, g in df.groupby(keys):
        g = g.sort_values("step")
        vals = vals if isinstance(vals, tuple) else (vals,)
        system = str(vals[0]); schedule = vals[1] if len(vals) > 1 else ""
        nmb = int(g["num_micro_batches"].iloc[0]) if "num_micro_batches" in g else 1
        if system == "singlegpu":
            # bug: each of nmb accumulation steps pulled a batch of (nmb*mbs)
            tokens_per_step = nmb * (nmb * micro_batch_size) * seq_len
            label = _nice_label(system, schedule) + f"  (saw {nmb}x tokens)"
        else:
            tokens_per_step = (nmb * micro_batch_size) * seq_len
            label = _nice_label(system, schedule)
        tokens_m = g["step"] * tokens_per_step / 1e6
        line, = ax.plot(tokens_m, _smooth(g["loss"]), label=label, lw=2)
        ax.plot(tokens_m, g["loss"], color=line.get_color(), alpha=0.15)
    ax.set_xlabel("tokens processed (millions)"); ax.set_ylabel("training loss")
    ax.set_title("Convergence: training loss vs tokens seen (fair across batch sizes)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_dir, "convergence_vs_tokens.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/results.csv")
    ap.add_argument("--loss-csv", default="results/loss_curves.csv")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--model", default="gpt2-xl")
    ap.add_argument("--seq-len", type=int, default=512)
    args = ap.parse_args()

    if os.path.exists(args.csv):
        df = _filter(pd.read_csv(args.csv), args.model, args.seq_len)
        if not df.empty:
            plot_bubble_empirical(df, args.out)
        else:
            print("[skip] bubble_empirical: no rows after model/seq-len filter")
    else:
        print(f"[skip] bubble_empirical: {args.csv} not found")

    plot_convergence(args.loss_csv, args.out)
    plot_convergence_vs_tokens(args.loss_csv, args.out, seq_len=args.seq_len)
    print("\nanalysis figures in", args.out)


if __name__ == "__main__":
    main()
