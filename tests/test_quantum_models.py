import numpy as np
import pytest
import torch

from src.classical.cnn_lstm import CNNLSTM, predict_proba
from src.quantum.circuits import fidelity_kernel, kernel_diagnostics
from src.quantum.encoding import EncodingConfig, build_encoder
from src.quantum.qsvm import QuantumSVM
from src.quantum.vqc import VariationalQuantumClassifier


def test_cnn_forward_and_probabilities():
    model = CNNLSTM(n_features=8, conv_filters=8, lstm_units=8, dense_units=8, dropout=0.0)
    model.eval()
    X = torch.randn(4, 8)
    with torch.no_grad():
        logits = model(X)
    assert logits.shape == (4,)
    probs = predict_proba(model, X.numpy())
    assert probs.shape == (4,)
    assert np.all((probs >= 0) & (probs <= 1))


def test_angle_encoder_state_normalisation():
    cfg = EncodingConfig(method="angle", n_features=4, map_name="zz", reps=1)
    encoder = build_encoder(cfg)
    assert cfg.num_qubits == 4
    X = np.linspace(0, np.pi, 20).reshape(5, 4)
    states = encoder.states(X)
    assert states.shape == (5, 16)
    np.testing.assert_allclose(np.linalg.norm(states, axis=1), 1.0, atol=1e-10)


def test_amplitude_encoder_padding():
    cfg = EncodingConfig(method="amplitude", n_features=8)
    encoder = build_encoder(cfg)
    assert cfg.num_qubits == 3
    assert cfg.padded_dimension == 8
    assert cfg.padding == 0

    cfg12 = EncodingConfig(method="amplitude", n_features=12)
    assert cfg12.num_qubits == 4
    assert cfg12.padding == 4
    states = encoder.states(np.random.default_rng(0).random((3, 8)))
    assert states.shape == (3, 8)
    np.testing.assert_allclose(np.linalg.norm(states, axis=1), 1.0, atol=1e-10)


def test_fidelity_kernel_properties():
    states = np.random.default_rng(1).normal(size=(6, 8)) + 1j * np.random.default_rng(2).normal(size=(6, 8))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    K = fidelity_kernel(states)
    assert K.shape == (6, 6)
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)
    assert K.min() >= -1e-12 and K.max() <= 1 + 1e-12
    diag = kernel_diagnostics(K)
    assert diag["diagonal_mean"] == pytest.approx(1.0, abs=1e-10)


def test_qsvm_fit_predict_small():
    rng = np.random.default_rng(3)
    X = rng.random((16, 4)) * np.pi
    y = np.array([0, 1] * 8)
    model = QuantumSVM(
        EncodingConfig(method="angle", n_features=4, reps=1),
        validate_kernel=False, probability=True, seed=42,
    ).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (16,)
    assert np.all((proba >= 0) & (proba <= 1))
    assert model.resources()["num_qubits"] == 4


def test_vqc_fit_converges_and_predicts():
    rng = np.random.default_rng(4)
    X = rng.random((16, 4)) * np.pi
    y = np.array([0, 1] * 8)
    model = VariationalQuantumClassifier(
        EncodingConfig(method="angle", n_features=4, reps=1),
        ansatz_reps=1, maxiter=20, seed=42,
    )
    model.fit(X, y, X, y, verbose=False)
    assert len(model.history_) >= 1
    proba = model.predict_proba(X)
    assert proba.shape == (16,)
    assert np.all((proba >= 0) & (proba <= 1))
    assert model.resources()["num_qubits"] == 4
