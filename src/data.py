"""WikiText loading -> tokenize -> fixed-length packing for causal LM.

Shared by all three training paths so every system trains on *identical* data
(same tokenizer, seq_len, packing, and order). See PROJECT_PLAN.md §3.

Design notes
------------
- Dataset id is **`Salesforce/wikitext`** (the bare `"wikitext"` id is the
  deprecated path). Configs: `wikitext-2-raw-v1` (fast sweeps),
  `wikitext-103-raw-v1` (longer convergence run).
- We **pack** concatenated token streams into contiguous blocks of `seq_len`
  (the standard GPT-2 LM recipe) — no padding, every position is a real token,
  so throughput/perplexity are directly comparable to published numbers.
- The HF `Dataset` returned here holds plain python lists (`input_ids`,
  `labels`). We deliberately do **not** call `.set_format("torch")` so this
  module imports with no torch installed (handy for local data checks). The
  torch-side helpers (`collate`, `make_dataloader`) import torch lazily.
"""
from __future__ import annotations

from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer

DATASET_ID = "Salesforce/wikitext"


def get_tokenizer(model_name: str):
    """Load the model's own BPE tokenizer; GPT-2/OPT have no pad token, so we
    alias it to EOS (harmless because we pack and never actually pad)."""
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_lm_dataset(
    model_name: str,
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "train",
    seq_len: int = 512,
    tokenizer=None,
    num_proc: int = 1,
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
):
    """Return an HF `Dataset` of fixed-length blocks with `input_ids` and
    `labels` (labels == input_ids; the shift happens in the loss).

    `max_samples` truncates to the first N blocks — keep Kaggle runs short.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(model_name)

    raw = load_dataset(DATASET_ID, dataset_config, split=split, cache_dir=cache_dir)

    def _tokenize(batch):
        # Only need input_ids; attention is full-causal over packed blocks.
        return {"input_ids": tokenizer(batch["text"])["input_ids"]}

    tokenized = raw.map(
        _tokenize,
        batched=True,
        remove_columns=raw.column_names,
        num_proc=num_proc,
        desc=f"tokenize[{dataset_config}/{split}]",
    )

    def _group(batch):
        # Concatenate all token ids in this map-batch, then chop into seq_len blocks.
        concat = []
        for ids in batch["input_ids"]:
            concat.extend(ids)
        total = (len(concat) // seq_len) * seq_len
        blocks = [concat[i : i + seq_len] for i in range(0, total, seq_len)]
        return {"input_ids": blocks, "labels": [b[:] for b in blocks]}

    lm = tokenized.map(
        _group,
        batched=True,
        num_proc=num_proc,
        desc=f"pack[{seq_len}]",
    )

    if max_samples is not None:
        lm = lm.select(range(min(max_samples, len(lm))))
    return lm


# --------------------------------------------------------------------------- #
# torch-side helpers (imported lazily so the build step above stays torch-free) #
# --------------------------------------------------------------------------- #
def collate(batch):
    """Stack a list of `{input_ids, labels}` rows into (input_ids, labels) tensors."""
    import torch

    input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    return input_ids, labels


def make_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 1234,
    drop_last: bool = True,
    num_workers: int = 2,
):
    """A plain (non-distributed) DataLoader. For pipeline parallelism every rank
    must iterate the dataset in the *same* order, so we shuffle deterministically
    with a fixed generator rather than a DistributedSampler (data is replicated,
    not sharded — pipeline splits the *model*, not the batch across data ranks)."""
    import torch
    from torch.utils.data import DataLoader

    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
