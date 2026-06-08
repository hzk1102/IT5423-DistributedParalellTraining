"""Topic 2 — Pipeline Parallelism for LLMs exceeding GPU memory (Kaggle 2x T4).

Shared package for the three training paths compared in this study:
  - train_singlegpu.py   : 1-GPU gradient-checkpointing baseline (control)
  - train_torchpp.py     : torch.distributed.pipelining (GPipe/1F1B/Interleaved/Zero-Bubble)
  - train_deepspeed.py   : DeepSpeed Pipeline Parallelism (PipelineModule, 1F1B)

See PROJECT_PLAN.md for the design and KAGGLE_GUIDE.md to run it.
"""
