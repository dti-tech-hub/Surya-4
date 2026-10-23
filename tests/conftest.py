"""Shared synthetic NSL-KDD-like fixtures (no external data required)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_frame(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = rng.choice(
        ["normal", "neptune", "satan", "guess_passwd", "buffer_overflow"],
        size=n,
        p=[0.45, 0.30, 0.15, 0.07, 0.03],
    )
    frame = pd.DataFrame({
        "duration": rng.integers(0, 100, n),
        "src_bytes": rng.integers(0, 5000, n),
        "dst_bytes": rng.integers(0, 5000, n),
        "count": rng.integers(1, 300, n),
        "serror_rate": rng.random(n),
        "protocol_type": rng.choice(["tcp", "udp", "icmp"], n),
        "service": rng.choice(["http", "private", "ftp_data", "eco_i"], n),
        "flag": rng.choice(["SF", "S0", "REJ"], n),
        "label": labels,
        "difficulty": rng.integers(1, 21, n),
    })
    from src.data.loader import harmonise_labels

    return harmonise_labels(frame)


@pytest.fixture(scope="session")
def nsl_frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture(scope="session")
def split_frames(nsl_frame):
    from src.data.splitting import stratified_split_indices

    idx = stratified_split_indices(nsl_frame["is_attack"].to_numpy(), 0.7, 0.15, 0.15, 42)
    return tuple(
        nsl_frame.iloc[idx[name]].reset_index(drop=True)
        for name in ("train", "validation", "test")
    )
