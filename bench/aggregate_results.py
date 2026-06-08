"""Summarise results/results.csv into the tables the report needs.

    python bench/aggregate_results.py [--csv results/results.csv]

Prints (and writes results/summary.csv):
  - micro-batch sweep per (system, schedule): tok/s, peak mem, bubble theory
  - schedule ladder comparison at fixed M
  - 1-GPU baseline vs best 2-GPU pipeline
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/results.csv")
    args = ap.parse_args()
    if not os.path.exists(args.csv):
        raise SystemExit(f"no results at {args.csv} -- run a training script first")

    df = pd.read_csv(args.csv)
    df = df.sort_values("timestamp")
    cols = ["system", "schedule", "num_micro_batches", "global_batch", "seq_len",
            "tokens_per_sec", "peak_mem_gb_max", "bubble_theory", "mfu",
            "final_loss", "eval_ppl"]
    cols = [c for c in cols if c in df.columns]

    print("\n========== ALL RUNS ==========")
    print(df[cols].to_string(index=False))

    print("\n========== MICRO-BATCH SWEEP (tok/s by system x schedule x M) ==========")
    sweep = (df.groupby(["system", "schedule", "num_micro_batches"])
               .agg(tok_s=("tokens_per_sec", "max"),
                    peak_gb=("peak_mem_gb_max", "max"),
                    bubble=("bubble_theory", "mean"),
                    mfu=("mfu", "max"))
               .reset_index())
    print(sweep.to_string(index=False))

    print("\n========== SCHEDULE LADDER (torchpp, by schedule) ==========")
    ladder = (df[df.system == "torchpp"]
                .groupby("schedule")
                .agg(best_tok_s=("tokens_per_sec", "max"),
                     peak_gb=("peak_mem_gb_max", "max"),
                     mfu=("mfu", "max"))
                .reset_index())
    print(ladder.to_string(index=False))

    print("\n========== 1-GPU vs 2-GPU (best per system) ==========")
    best = (df.groupby("system")
              .agg(best_tok_s=("tokens_per_sec", "max"),
                   peak_gb=("peak_mem_gb_max", "max"),
                   best_mfu=("mfu", "max"),
                   best_ppl=("eval_ppl", "max"))
              .reset_index())
    print(best.to_string(index=False))

    out = os.path.join(os.path.dirname(args.csv) or ".", "summary.csv")
    sweep.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
