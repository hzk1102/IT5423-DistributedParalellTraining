"""Generate the report figures from results/results.csv.

    python bench/plots.py [--csv results/results.csv] [--out results/figures]

Figures (PNG):
  1. throughput vs M           (per system/schedule)  + theoretical bubble overlay
  2. peak VRAM/GPU vs M        (GPipe grows ~M; 1F1B flat)
  3. schedule-ladder bar       (tok/s at fixed M)
  4. 1-GPU vs 2-GPU            (throughput and peak memory)

Runs headless (Agg backend) so it works in a Kaggle cell; display the PNGs after.
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


def plot_throughput_vs_m(df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for (system, schedule), g in df.groupby(["system", "schedule"]):
        g = g.sort_values("num_micro_batches")
        if g["num_micro_batches"].nunique() < 2:
            continue
        ax.plot(g["num_micro_batches"], g["tokens_per_sec"], marker="o",
                label=f"{system}/{schedule}")
    ax.set_xlabel("micro-batches M"); ax.set_ylabel("tokens/sec")
    ax.set_title("Throughput vs micro-batch count"); ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_dir, "throughput_vs_M.png")


def plot_bubble(df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    sub = df[df.bubble_theory > 0].sort_values("num_micro_batches")
    if not sub.empty:
        b = sub.drop_duplicates("num_micro_batches")
        ax.plot(b["num_micro_batches"], b["bubble_theory"] * 100, "k--o",
                label="theoretical bubble (P=2)")
    ax.set_xlabel("micro-batches M"); ax.set_ylabel("bubble fraction (%)")
    ax.set_title("Pipeline bubble vs M  [ (P-1)/(M+P-1) ]")
    ax.set_xscale("log", base=2); ax.grid(True, alpha=0.3); ax.legend()
    _save(fig, out_dir, "bubble_vs_M.png")


def plot_mem_vs_m(df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for (system, schedule), g in df.groupby(["system", "schedule"]):
        g = g.sort_values("num_micro_batches")
        if g["num_micro_batches"].nunique() < 2:
            continue
        ax.plot(g["num_micro_batches"], g["peak_mem_gb_max"], marker="s",
                label=f"{system}/{schedule}")
    ax.axhline(16, color="r", ls=":", label="T4 16 GB")
    ax.set_xlabel("micro-batches M"); ax.set_ylabel("peak VRAM / GPU (GB)")
    ax.set_title("Peak memory vs M"); ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_dir, "memory_vs_M.png")


def plot_schedule_ladder(df, out_dir):
    pp = df[df.system == "torchpp"]
    if pp.empty:
        return
    best = pp.groupby("schedule")["tokens_per_sec"].max().sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(best.index, best.values)
    ax.set_xlabel("best tokens/sec"); ax.set_title("Schedule ladder (torch.pipelining)")
    ax.grid(True, axis="x", alpha=0.3)
    _save(fig, out_dir, "schedule_ladder.png")


def plot_1gpu_vs_2gpu(df, out_dir):
    best = df.groupby("system").agg(tok_s=("tokens_per_sec", "max"),
                                    mem=("peak_mem_gb_max", "max")).reset_index()
    if best.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(best["system"], best["tok_s"]); axes[0].set_ylabel("best tokens/sec")
    axes[0].set_title("Throughput by system")
    axes[1].bar(best["system"], best["mem"]); axes[1].set_ylabel("peak VRAM/GPU (GB)")
    axes[1].axhline(16, color="r", ls=":"); axes[1].set_title("Peak memory by system")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_dir, "system_comparison.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/results.csv")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--model", default="gpt2-xl",
                    help="keep only this model (drops smoke-test rows); '' = no filter")
    ap.add_argument("--seq-len", type=int, default=512,
                    help="keep only this seq_len (drops smoke-test rows); 0 = no filter")
    args = ap.parse_args()
    if not os.path.exists(args.csv):
        raise SystemExit(f"no results at {args.csv} -- run a training script first")
    df = pd.read_csv(args.csv)
    # Drop smoke-test rows (append-only CSV); see aggregate_results.py for why.
    n_all = len(df)
    if args.model:
        df = df[df["model"] == args.model]
    if args.seq_len:
        df = df[df["seq_len"] == args.seq_len]
    if len(df) < n_all:
        print(f"[filter] kept {len(df)}/{n_all} rows (model={args.model or 'any'}, "
              f"seq_len={args.seq_len or 'any'})")
    if df.empty:
        raise SystemExit("no rows left after filtering -- check --model/--seq-len")
    plot_throughput_vs_m(df, args.out)
    plot_bubble(df, args.out)
    plot_mem_vs_m(df, args.out)
    plot_schedule_ladder(df, args.out)
    plot_1gpu_vs_2gpu(df, args.out)
    print("\nall figures in", args.out)


if __name__ == "__main__":
    main()
