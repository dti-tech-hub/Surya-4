import numpy as np
import pytest

from src.data.loader import (
    ATTACK_CATEGORY_MAP,
    harmonise_labels,
    get_feature_columns,
    NSL_KDD_FEATURES,
)
from src.data.splitting import (
    build_unseen_category_split,
    split_dataframe,
    stratified_split_indices,
    verify_no_category_leakage,
)
from src.utils.config import load_config


def test_harmonise_labels_maps_known_attacks(nsl_frame):
    assert set(nsl_frame["attack_category"].unique()) <= {"Normal", "DoS", "Probe", "R2L", "U2R"}
    row = nsl_frame[nsl_frame["label"] == "neptune"].iloc[0]
    assert row["is_attack"] == 1
    assert row["attack_category"] == "DoS"
    normal = nsl_frame[nsl_frame["label"] == "normal"].iloc[0]
    assert normal["is_attack"] == 0
    assert normal["attack_category"] == "Normal"


def test_attack_taxonomy_covers_standard_categories():
    assert ATTACK_CATEGORY_MAP["neptune"] == "DoS"
    assert ATTACK_CATEGORY_MAP["satan"] == "Probe"
    assert ATTACK_CATEGORY_MAP["guess_passwd"] == "R2L"
    assert ATTACK_CATEGORY_MAP["buffer_overflow"] == "U2R"
    assert len(ATTACK_CATEGORY_MAP) >= 38


def test_feature_columns_exclude_metadata(nsl_frame):
    cols = get_feature_columns(nsl_frame)
    assert "label" not in cols
    assert "is_attack" not in cols
    assert "attack_category" not in cols
    assert len(cols) == 8  # synthetic frame has 5 numeric + 3 categorical features


def test_stratified_split_disjoint_and_sized(nsl_frame):
    idx = stratified_split_indices(nsl_frame["is_attack"].to_numpy(), 0.7, 0.15, 0.15, 42)
    sets = {k: set(v.tolist()) for k, v in idx.items()}
    assert sets["train"].isdisjoint(sets["validation"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["validation"].isdisjoint(sets["test"])
    assert len(sets["train"]) + len(sets["validation"]) + len(sets["test"]) == len(nsl_frame)
    # Stratification preserves both classes in every split.
    for key, indices in idx.items():
        assert nsl_frame.iloc[indices]["is_attack"].nunique() == 2


def test_split_dataframe_reports_no_overlap(nsl_frame):
    cfg = load_config(root=".")
    train, val, test, meta = split_dataframe(nsl_frame, cfg)
    assert len(train) and len(val) and len(test)
    assert all(v == 0 for v in meta["overlap"].values())


def test_unseen_category_split_removes_holdout(nsl_frame):
    split = build_unseen_category_split(nsl_frame, ["R2L", "U2R"], seed=42, normal_ratio=1.0)
    audit = verify_no_category_leakage(split["train"], split["test"], ["R2L", "U2R"])
    assert audit["passed"]
    assert "R2L" not in audit["train_categories"]
    assert "U2R" not in audit["train_categories"]
    assert set(audit["test_categories"]) <= {"Normal", "R2L", "U2R"}
    assert split["test"]["is_attack"].nunique() == 2
