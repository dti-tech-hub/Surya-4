"""Reproducible splitting utilities.

Primary protocol
----------------
* Pool      : the KDDTrain+ file (the file NSL-KDD intends for model fitting).
* Train/Val : 70% / 15%, stratified on the binary label.
* Test      : 15%, stratified, touched only for final evaluation.
* External  : the official KDDTest+ file, never used for fitting or selection.

A separate *unseen attack-category* protocol removes whole attack categories
from the training pool (see ``build_unseen_category_split``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.io import save_json


class SplitError(RuntimeError):
    pass


def stratified_split_indices(
    y: np.ndarray,
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    if abs(train_size + validation_size + test_size - 1.0) > 1e-9:
        raise SplitError("train/validation/test fractions must sum to 1.0")
    y = np.asarray(y)
    indices = np.arange(len(y))

    # First carve out the test set, then split the remainder.
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=y, random_state=seed, shuffle=True
    )
    relative_val = validation_size / (train_size + validation_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val,
        stratify=y[train_val_idx],
        random_state=seed,
        shuffle=True,
    )
    return {
        "train": np.sort(train_idx),
        "validation": np.sort(val_idx),
        "test": np.sort(test_idx),
    }


def split_dataframe(df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    seed = int(cfg.get_path("project.seed", 42))
    fracs = {
        "train": float(cfg.get_path("split.train_size", 0.70)),
        "validation": float(cfg.get_path("split.validation_size", 0.15)),
        "test": float(cfg.get_path("split.test_size", 0.15)),
    }
    y = df["is_attack"].to_numpy()
    idx = stratified_split_indices(y, fracs["train"], fracs["validation"], fracs["test"], seed)

    overlap = {
        "train_validation": len(np.intersect1d(idx["train"], idx["validation"])),
        "train_test": len(np.intersect1d(idx["train"], idx["test"])),
        "validation_test": len(np.intersect1d(idx["validation"], idx["test"])),
    }
    if any(overlap.values()):
        raise SplitError(f"Split overlap detected: {overlap}")

    parts = {name: df.iloc[ix].reset_index(drop=True) for name, ix in idx.items()}
    meta = {
        "seed": seed,
        "fractions": fracs,
        "sizes": {k: int(len(v)) for k, v in parts.items()},
        "indices": {k: v.tolist() for k, v in idx.items()},
        "overlap": overlap,
        "stratified_on": "is_attack",
        "leakage_guard": "test indices are only materialised at final evaluation",
    }
    return parts["train"], parts["validation"], parts["test"], meta


def save_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    cfg,
    extra: Optional[dict] = None,
) -> dict:
    out_dir = Path(cfg.get_path("paths.data_splits"))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        path = out_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = str(path)
    meta = {
        "sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "files": paths,
        "columns": list(train.columns),
        **(extra or {}),
    }
    save_json(meta, out_dir / "split_metadata.json")
    return meta


def load_splits(cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path(cfg.get_path("paths.data_splits"))
    frames = []
    for name in ("train", "validation", "test"):
        path = base / f"{name}.parquet"
        if not path.exists():
            raise SplitError(
                f"Missing split file {path}. Run preprocessing first: "
                "python run_pipeline.py --mode quick --stage prepare"
            )
        frames.append(pd.read_parquet(path))
    return tuple(frames)  # type: ignore[return-value]


def stratified_subsample_indices(
    y: np.ndarray,
    n: Optional[int],
    seed: int = 42,
    ensure_both_classes: bool = True,
) -> np.ndarray:
    """Stratified subsample of at most ``n`` rows (deterministic for a seed)."""
    y = np.asarray(y)
    if n is None or n >= len(y):
        return np.arange(len(y))
    idx, _ = train_test_split(
        np.arange(len(y)), train_size=int(n), stratify=y, random_state=seed, shuffle=True
    )
    idx = np.sort(idx)
    if ensure_both_classes and len(np.unique(y[idx])) < 2:
        raise SplitError("Subsample contains a single class; increase the sample size.")
    return idx


def make_common_test_subset(
    test_df: pd.DataFrame,
    n: Optional[int],
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """The identical evaluation set every model is scored on."""
    idx = stratified_subsample_indices(test_df["is_attack"].to_numpy(), n, seed)
    return test_df.iloc[idx].reset_index(drop=True), idx


def build_unseen_category_split(
    df: pd.DataFrame,
    holdout_categories: Iterable[str],
    seed: int = 42,
    normal_ratio: float = 1.0,
    validation_fraction: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """Remove entire attack categories from training.

    Returns ``train`` (normal + known attacks), ``validation`` (normal + known
    attacks) and ``test`` (normal + *only* the held-out categories).
    """
    holdout = [str(c) for c in holdout_categories]
    df = df.copy()
    df["attack_category"] = df["attack_category"].astype(str)

    held = df[df["attack_category"].isin(holdout)].reset_index(drop=True)
    known = df[~df["attack_category"].isin(holdout)].reset_index(drop=True)
    if held.empty:
        raise SplitError(
            f"No samples found for holdout categories {holdout}. "
            "Check the dataset attack-category mapping."
        )

    # Split the known subset into train / validation (stratified).
    train_known, val_known = train_test_split(
        known,
        test_size=validation_fraction,
        stratify=known["is_attack"],
        random_state=seed,
        shuffle=True,
    )

    normal_pool = known[known["is_attack"] == 0]
    n_normal = int(round(len(held) * normal_ratio))
    n_normal = min(n_normal, len(normal_pool))
    normal_test = normal_pool.sample(n=n_normal, random_state=seed)

    test = pd.concat([held, normal_test], ignore_index=True).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)

    return {
        "train": train_known.reset_index(drop=True),
        "validation": val_known.reset_index(drop=True),
        "test": test,
        "holdout_categories": holdout,
        "held_out_counts": {str(k): int(v) for k, v in held["attack_category"].value_counts().items()},
        "test_label_counts": {str(k): int(v) for k, v in test["label"].value_counts().items()},
    }


def verify_no_category_leakage(train: pd.DataFrame, test: pd.DataFrame,
                               holdout_categories: Iterable[str]) -> dict:
    holdout = set(str(c) for c in holdout_categories)
    train_cats = set(train["attack_category"].astype(str).unique())
    leaked = sorted(train_cats & holdout)
    return {
        "holdout_categories": sorted(holdout),
        "train_categories": sorted(train_cats),
        "test_categories": sorted(set(test["attack_category"].astype(str).unique())),
        "leaked_categories": leaked,
        "passed": len(leaked) == 0,
    }
