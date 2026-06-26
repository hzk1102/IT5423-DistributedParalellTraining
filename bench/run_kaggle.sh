#!/usr/bin/env bash
# ===========================================================================
# PRODUCTION sweep for the Topic-2 report (NO smoke test) -- run on Kaggle 2x T4.
# Run from the PROJECT ROOT inside a notebook cell:   !bash bench/run_kaggle.sh
#
# Closes the four examiner gaps:
#   (A) enablement   : opt-2.7b OOMs on 1 GPU, trains on 2-GPU pipeline
#   (B) recompute    : 1-GPU baseline WITH and WITHOUT grad-ckpt (isolate the tax)
#   (C) quality      : DeepSpeed + single-GPU eval perplexity; loss curves for all 3
#   (D) bubble       : the M-sweep feeds the measured-vs-theory bubble plot
#
# Tunables (defaults are the report values -- only change to shorten):
#     STEPS=30 CONV_STEPS=300 bash bench/run_kaggle.sh
# Quota note: this whole script is ~60-90 min of T4 time. To split across
# sessions, comment out blocks you have already logged (results.csv is append-
# only and aggregate filters by model, so partial runs are safe).
# ===========================================================================
set -u
MODEL=${MODEL:-gpt2-xl}          # primary model (you already have most of this data)
BIG=${BIG:-facebook/opt-2.7b}    # enablement model: too big for one 16 GB T4
SEQ=${SEQ:-512}
STEPS=${STEPS:-30}               # throughput/memory steady-state (short is fine)
CONV_STEPS=${CONV_STEPS:-300}    # convergence curves (longer; quality signal)
MBS=${MBS:-1}
CFG=${CFG:-wikitext-2-raw-v1}
LOSSLOG=results/loss_curves.csv
echo "=== PRODUCTION sweep: model=$MODEL big=$BIG seq=$SEQ steps=$STEPS conv=$CONV_STEPS ==="

run() { echo; echo ">>> $*"; "$@" || echo "!!! FAILED (continuing): $*"; }

# (0) sanity: prove the 2-stage split reproduces HF logits before trusting numbers
run python src/check_split.py --model "$MODEL" --num-stages 2

# --------------------------------------------------------------------------- #
# (A) ENABLEMENT: the headline "model exceeds one GPU" demonstration           #
#     A1 is EXPECTED TO OOM -- that crash IS the evidence. Screenshot it.       #
# --------------------------------------------------------------------------- #
echo; echo "### (A1) opt-2.7b on ONE T4 -- expected to OOM (this is the point) ###"
run env CUDA_VISIBLE_DEVICES=0 python src/train_singlegpu.py --model "$BIG" \
    --grad-checkpointing --optim adamw8bit --micro-batch-size "$MBS" \
    --num-micro-batches 8 --seq-len "$SEQ" --max-steps 10 \
    --notes "enablement: single-GPU attempt (expected OOM)"

echo; echo "### (A2) opt-2.7b on TWO T4 via pipeline -- expected to SUCCEED ###"
run torchrun --standalone --nproc_per_node=2 src/train_torchpp.py --model "$BIG" \
    --schedule 1f1b --num-micro-batches 8 --micro-batch-size "$MBS" \
    --seq-len "$SEQ" --max-steps "$STEPS" --notes "enablement: 2-GPU pipeline success"

# --------------------------------------------------------------------------- #
# (B) RECOMPUTE ISOLATION: single-GPU baseline, with and without grad-ckpt      #
#     (without may OOM on gpt2-xl -- if so, that itself is a reportable result) #
# --------------------------------------------------------------------------- #
run env CUDA_VISIBLE_DEVICES=0 python src/train_singlegpu.py --model "$MODEL" \
    --grad-checkpointing --optim adamw8bit --micro-batch-size "$MBS" \
    --num-micro-batches 8 --seq-len "$SEQ" --max-steps "$STEPS" --eval \
    --notes "baseline +grad_ckpt"

run env CUDA_VISIBLE_DEVICES=0 python src/train_singlegpu.py --model "$MODEL" \
    --optim adamw8bit --micro-batch-size "$MBS" \
    --num-micro-batches 8 --seq-len "$SEQ" --max-steps "$STEPS" --eval \
    --notes "baseline NO grad_ckpt (isolates recompute tax; may OOM)"

# --------------------------------------------------------------------------- #
# (C/D) gpt2-xl micro-batch sweep on torch.pipelining 1F1B  (bubble + scaling)  #
# --------------------------------------------------------------------------- #
for M in 1 2 4 8 16; do
  run torchrun --standalone --nproc_per_node=2 src/train_torchpp.py --model "$MODEL" \
      --schedule 1f1b --num-micro-batches "$M" --micro-batch-size "$MBS" \
      --seq-len "$SEQ" --max-steps "$STEPS"
done

# GPipe sweep: same bubble as 1F1B but memory grows ~M (your best memory figure)
for M in 1 2 4 8; do
  run torchrun --standalone --nproc_per_node=2 src/train_torchpp.py --model "$MODEL" \
      --schedule gpipe --num-micro-batches "$M" --micro-batch-size "$MBS" \
      --seq-len "$SEQ" --max-steps "$STEPS"
done

# Modern schedule ladder at fixed M=8 (zero-bubble needs torch>=2.6)
for S in interleaved_1f1b interleaved_zb zbv; do
  run torchrun --standalone --nproc_per_node=2 src/train_torchpp.py --model "$MODEL" \
      --schedule "$S" --num-micro-batches 8 --micro-batch-size "$MBS" \
      --seq-len "$SEQ" --max-steps "$STEPS"
done

# DeepSpeed PP micro-batch sweep -- NOTE: --eval now ON (quality axis)
for M in 1 2 4 8 16; do
  run deepspeed --num_gpus 2 src/train_deepspeed.py --model "$MODEL" \
      --num-micro-batches "$M" --micro-batch-size "$MBS" \
      --seq-len "$SEQ" --max-steps "$STEPS" --eval
done

# --------------------------------------------------------------------------- #
# (C) CONVERGENCE: longer fixed-M=8 run per system -> overlaid loss curves      #
#     all three append to the same loss_curves.csv (tagged by system/schedule)  #
# --------------------------------------------------------------------------- #
run env CUDA_VISIBLE_DEVICES=0 python src/train_singlegpu.py --model "$MODEL" \
    --grad-checkpointing --optim adamw8bit --micro-batch-size "$MBS" \
    --num-micro-batches 8 --seq-len "$SEQ" --max-steps "$CONV_STEPS" \
    --loss-log "$LOSSLOG" --notes "convergence"

run torchrun --standalone --nproc_per_node=2 src/train_torchpp.py --model "$MODEL" \
    --schedule 1f1b --num-micro-batches 8 --micro-batch-size "$MBS" \
    --seq-len "$SEQ" --max-steps "$CONV_STEPS" --loss-scale 1024 \
    --loss-log "$LOSSLOG" --notes "convergence"

run deepspeed --num_gpus 2 src/train_deepspeed.py --model "$MODEL" \
    --num-micro-batches 8 --micro-batch-size "$MBS" \
    --seq-len "$SEQ" --max-steps "$CONV_STEPS" --loss-log "$LOSSLOG" --notes "convergence"

echo; echo "=== sweep done -> results/results.csv  +  $LOSSLOG ==="
echo "Now run the aggregation + figures:"
echo "  python bench/aggregate_results.py --model $MODEL --seq-len $SEQ"
echo "  python bench/plots.py            --model $MODEL --seq-len $SEQ"
echo "  python bench/plot_analysis.py    --model $MODEL --seq-len $SEQ"
