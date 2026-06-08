"""Small cross-cutting helpers: seeding, distributed setup, and rank-aware printing.

Kept torch-import-light at module top so the parts that don't need a GPU (e.g.
argparse defaults) can be imported on a CPU-only box. torch is imported lazily
inside the functions that actually need it.
"""
from __future__ import annotations

import os
import random


def set_seed(seed: int = 1234) -> None:
    """Seed python / numpy / torch for reproducible data order and init."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Distributed helpers (torchrun-style env: RANK / LOCAL_RANK / WORLD_SIZE)     #
# --------------------------------------------------------------------------- #
def env_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def env_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


def env_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main() -> bool:
    return env_rank() == 0


def setup_distributed(backend: str = "nccl"):
    """Init the default process group from torchrun env vars and bind this rank
    to its GPU. Returns (rank, local_rank, world_size, device)."""
    import torch
    import torch.distributed as dist

    rank, local_rank, world = env_rank(), env_local_rank(), env_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world)
    return rank, local_rank, world, device


def cleanup_distributed() -> None:
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def rank0_print(*args, **kwargs) -> None:
    if is_main():
        print(*args, **kwargs, flush=True)
