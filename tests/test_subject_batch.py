"""Subject identity survives lazy and parquet batch paths."""

import numpy as np
import torch

from src.utils import h5io
from src.utils.primitives import InputMode, OutputMode, PviChannelGroup


def test_slice_sequences_includes_subject_idx():
    period_length = 50
    t = period_length * 2
    data = {
        "bp": torch.randn(2, period_length),
        "stats": torch.randn(2, 5, 2),
        "pviHP": torch.randn(1, t),
        "pviLP": torch.randn(1, t),
    }
    sample = h5io.slice_sequences(
        data=data,
        bounds=(1, 2),
        period_length=period_length,
        subject="subject042",
    )
    assert int(sample["subject_idx"]) == 42
    assert set(sample.keys()) >= set(PviChannelGroup.keys()) | {"bp", "stats", "subject_idx"}


def test_subject_idx_collates_in_batch():
    from torch.utils.data import DataLoader, TensorDataset

    ds = TensorDataset(
        torch.zeros(4, 1),
        torch.tensor([1, 2, 3, 4], dtype=torch.long),
    )

    class _Wrap(torch.utils.data.Dataset):
        def __init__(self, base):
            self.base = base

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            _, subject_idx = self.base[idx]
            return {"bp": torch.zeros(50), "subject_idx": subject_idx}

    loader = DataLoader(_Wrap(ds), batch_size=4)
    batch = next(iter(loader))
    assert batch["subject_idx"].tolist() == [1, 2, 3, 4]
