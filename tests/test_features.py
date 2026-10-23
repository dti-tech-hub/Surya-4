import numpy as np
import pytest

from src.data.preprocessing import Preprocessor
from src.features.mutual_information import MIReducer
from src.features.pca import PCAReducer
from src.features.reduction import apply_reduction
from src.features.scaling import FeatureScaler


@pytest.fixture(scope="module")
def prepared(split_frames):
    train, val, test = split_frames
    pre = Preprocessor(min_category_frequency=2).fit(train)
    return pre, pre.transform(train), pre.transform(val), pre.transform(test)


def test_preprocessor_fit_on_train_only(prepared, split_frames):
    pre, Xtr, Xva, Xte = prepared
    assert Xtr.shape == (len(split_frames[0]), len(pre.feature_names))
    assert Xva.shape[1] == Xtr.shape[1] == Xte.shape[1]
    assert np.isfinite(Xtr).all()
    # Unseen categorical levels must not crash (handle_unknown="ignore").
    unseen = split_frames[2].copy()
    unseen.loc[unseen.index[0], "service"] = "never_seen_service"
    out = pre.transform(unseen)
    assert out.shape[1] == Xtr.shape[1]


def test_transform_is_stateless(prepared, split_frames):
    pre, Xtr, _, _ = prepared
    again = pre.transform(split_frames[0])
    np.testing.assert_allclose(again, Xtr, atol=1e-6)


def test_pca_reducer_shapes_and_variance(prepared):
    _, Xtr, Xva, _ = prepared
    reducer = PCAReducer(n_components=4, random_state=42).fit(Xtr)
    Ztr = reducer.transform(Xtr)
    Zva = reducer.transform(Xva)
    assert Ztr.shape == (len(Xtr), 4)
    assert Zva.shape == (len(Xva), 4)
    ev = reducer.explained_variance_ratio_
    assert np.all(ev >= 0)
    assert np.all(np.diff(ev) <= 1e-12)
    assert reducer.n_features_in_ == Xtr.shape[1]


def test_mi_reducer_selects_k(prepared):
    _, Xtr, _, _ = prepared
    y = np.random.default_rng(0).integers(0, 2, len(Xtr))
    reducer = MIReducer(k=3, random_state=42).fit(Xtr, y, feature_names=[str(i) for i in range(Xtr.shape[1])])
    Z = reducer.transform(Xtr)
    assert Z.shape == (len(Xtr), 3)
    assert len(reducer.selected_features()) == 3
    assert len(reducer.ranking()) == Xtr.shape[1]


def test_scaler_bounds(prepared):
    _, Xtr, Xva, _ = prepared
    scaler = FeatureScaler().fit(Xtr)
    Ztr = scaler.transform(Xtr)
    Zva = scaler.transform(Xva)
    assert Ztr.min() >= -1e-6
    assert Ztr.max() <= np.pi + 1e-6
    assert Zva.min() >= -1e-6
    assert Zva.max() <= np.pi + 1e-6
    restored = scaler.inverse_transform(Ztr)
    np.testing.assert_allclose(restored, np.clip(Xtr, scaler.lo_, scaler.hi_), atol=1e-5)


def test_apply_reduction_is_identical_for_all_models(prepared):
    _, Xtr, Xva, Xte = prepared
    ytr = np.random.default_rng(1).integers(0, 2, len(Xtr))
    A = apply_reduction("pca", 4, Xtr, ytr, Xva, Xte, [str(i) for i in range(Xtr.shape[1])])
    B = apply_reduction("pca", 4, Xtr, ytr, Xva, Xte, [str(i) for i in range(Xtr.shape[1])])
    for a, b in zip(A[:3], B[:3]):
        np.testing.assert_allclose(a, b, atol=1e-6)
    assert A[0].shape[1] == 4
