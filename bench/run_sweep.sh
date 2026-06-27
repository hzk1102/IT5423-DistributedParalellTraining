#!/usr/bin/env bash
# Full experimental sweep for Topic 2 (PROJECT_PLAN.md §5).
# Run from the PROJECT ROOT (so that `src/...` paths resolve):
#     bash bench/run_sweep.sh
# On Kaggle, run this inside a notebook cell:  !bash bench/run_sweep.sh
#
# Tunables via env vars (keep runs short on Kaggle):
#     MODEL=gpt2-xl SEQ=512 STEPS=30 bash bench/run_sweep.sh
# Use MODEL=gpt2 for a fast end-to-end smoke test before the real gpt2-xl run.
set -u
MODEL=${MODEL:-gpt2-xl}
SEQ=${SEQ:-512}
STEPS=${STEPS:-30}
MBS=${MBS:-1}
CFG=${CFG:-wikitext-2-raw-v1}
echo "=== sweep: model=$MODEL seq=$SEQ steps=$STEPS micro_bs=$MBS dataset=$CFG ==="

run() { echo; echo ">>> $*"; "$@" || echo "!!! FAILED (continuing): $*"; }

# (0) sanity: prove the split is faithful before trusting pipeline numbers
run python src/check_split.py --model "$MODEL" --num-stages 2

# (5) FSDP Data Parallelism sweep (ZeRO-3 - full / ZeRO-2 - grad / DDP baseline - no) 
# run first to test
# for SHARD in grad; do
#   run torchrun --standalone --nproc_per_node=2 src/train_fsdp.py \
#       --model "$MODEL" --shard-strategy "$SHARD" \
#       --micro-batch-size "$MBS" --num-micro-batches 8 \
#       --seq-len "$SEQ" --max-steps "$STEPS" --grad-checkpointing
# done

for M in 3 4 5 6; do
  run torchrun --standalone --nproc_per_node=2 src/train_fsdp.py \
      --model "$MODEL" --shard-strategy "grad" \
      --micro-batch-size "$MBS" --num-micro-batches "$M" \
      --seq-len "$SEQ" --max-steps "$STEPS"
done

# # (1) single-GPU grad-checkpointing baseline (deliverable 4 control)
# run env CUDA_VISIBLE_DEVICES=0 python src/train_singlegpu.py --model "$MODEL" \
#     --grad-checkpointing --optim adamw8bit --micro-batch-size "$MBS" \
#     --num-micro-batches 8 --seq-len "$SEQ" --max-steps "$STEPS" --eval

# (2) micro-batch sweep on torch.pipelining 1F1B (core of deliverable 3)
for M in 16 32; do
  run torchrun --standalone --nproc_per_node=2 src/train_torchpp.py --model "$MODEL" \
      --schedule 1f1b --num-micro-batches "$M" --micro-batch-size "$MBS" \
      --seq-len "$SEQ" --max-steps "$STEPS"
done

# (2b) GPipe for the same M (shows 1F1B==GPipe bubble but lower memory)
for M in 6 8; do
  run torchrun --standalone --nproc_per_node=2 src/train_torchpp.py --model "$MODEL" \
      --schedule gpipe --num-micro-batches "$M" --micro-batch-size "$MBS" \
      --seq-len "$SEQ" --max-steps "$STEPS"
done

# (3) modern schedule ladder at fixed M= (zero-bubble needs torch>=2.6)
for M in 12 16; do
  for S in interleaved_1f1b interleaved_zb zbv; do
    run torchrun --standalone --nproc_per_node=2 src/train_torchpp.py --model "$MODEL" \
        --schedule "$S" --num-micro-batches "$M" --micro-batch-size "$MBS" \
        --seq-len "$SEQ" --max-steps "$STEPS"
  done
done

# (4) DeepSpeed PP micro-batch sweep (deliverable 2 framework comparison)
for M in 24 32; do
  run deepspeed --num_gpus 2 src/train_deepspeed.py --model "$MODEL" \
      --num-micro-batches "$M" --micro-batch-size "$MBS" \
      --seq-len "$SEQ" --max-steps "$STEPS"
done

echo; echo "=== sweep done -> results/results.csv ==="
echo "Now run:  python bench/aggregate_results.py  &&  python bench/plots.py"
