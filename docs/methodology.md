# Methodology

This document describes the executed protocol behind QE-IDS. It is written to match the code in
`src/`, not an idealised design. All reported numbers in the project come from this protocol.

---

## 1. Research question and scope

**Question.** Given identical reduced inputs, identical splits, identical common test samples and
an identical evaluation protocol, does a quantum classifier improve binary intrusion detection
over a properly tuned classical CNN+LSTM on NSL-KDD?

**Scope.**

- Binary classification: Normal vs Attack.
- Offline batch evaluation, not streaming deployment.
- Quantum classifiers run on exact statevector simulation, not quantum hardware.
- The comparison is about predictive performance, calibration, robustness and cost under a
  controlled protocol; it is not a claim about asymptotic quantum advantage.

The protocol deliberately avoids three common weaknesses: test-set tuning, mismatched feature
representations, and unpaired comparisons.

---

## 2. Dataset

NSL-KDD consists of two files:

| File | Rows | Role |
|---|---:|---|
| `KDDTrain+.txt` | 125,973 | Only source of training data |
| `KDDTest+.txt` | 22,544 | External, never-trained-on evaluation set |

Each row has 41 features (3 categorical: `protocol_type`, `service`, `flag`; 38 numeric), a
`label`, and a `difficulty` score. The difficulty column is dropped. Labels are lower-cased and
mapped to the five NSL-KDD categories (Normal, DoS, Probe, R2L, U2R) and to a binary `is_attack`
target.

Verified composition:

- 23 distinct labels in KDDTrain+, 38 in KDDTest+.
- 17 attack types appear only in KDDTest+.
- 0 missing values and 0 exact duplicate rows in both files.
- KDDTrain+ attack rate: 46.54% (binary), KDDTest+ attack rate is different and is not used to
  tune anything.

The external file is used only for (a) the novel-attack-type experiment and (b) a simulated
file-order window scan in the drift experiment. It is never used for training, early stopping,
threshold selection, feature selection or calibration.

---

## 3. Splits and leakage control

### 3.1 Quick profile (the executed profile)

Quick mode draws a **stratified 12,000-row subsample** of KDDTrain+ and splits it 70/15/15 on the
binary label:

| Split | Rows | Attack rate |
|---|---:|---:|
| Train | 8,399 | 46.54% |
| Validation | 1,801 | 46.53% |
| Test (common) | 1,800 | 46.56% |

The test split is the **common test set** for all three models. The split is stratified on
`is_attack`; category and attack-type distributions are therefore approximately preserved but
not guaranteed for very rare types (U2R has only six rows in the quick pool).

The full profile is defined in `config/config.yaml` (full KDDTrain+ pool, 3,000 quantum training
rows, 2,500 common test rows, 5 folds) but was **not executed** for the reported results. No
full-profile numbers are claimed.

### 3.2 External set

The official KDDTest+ file is held out entirely. It contains 17 attack types absent from the full
KDDTrain+ label set. Novelty is defined against the **full KDDTrain+ label set**, not against the
quick-mode subsample; this prevents rare training types from being mislabelled as novel.

### 3.3 Leakage policy

Every fitted transformation is estimated on the training split only:

1. median imputation values,
2. one-hot vocabulary and rare-level collapsing,
3. numeric standardisation mean/std,
4. PCA loadings or mutual-information scores,
5. post-reduction min–max rescaling to [0, π].

Validation is used for:

- early stopping and best-epoch selection (CNN+LSTM),
- decision-threshold selection,
- feature-method and feature-count selection,
- encoding-comparison decisions.

The test split is scored once, after all choices are frozen.

The automated audit in `data/metadata/leakage_audit.json` performs eight checks:

| Check | What it verifies | Result |
|---|---|---|
| `scaler_fit_on_train` | Recomputing scaler statistics on train matches the saved scaler | PASSED |
| `onehot_vocabulary_from_train` | No test-only categorical level expanded the vocabulary | PASSED |
| `transform_is_stateless` | Repeated transforms of raw rows are identical | PASSED |
| `reducer_input_dimension` | Reducer receives the expected 111-column design matrix | PASSED |
| `reducer_output_dimension` | Reducer emits exactly 8 components | PASSED |
| `no_exact_train_test_row_overlap` | Zero exact row overlap across splits | PASSED |
| `reduced_shapes` | Train/validation/test shapes are consistent | PASSED |
| `pca_explained_variance_monotone` | PCA explained-variance curve is monotone | PASSED |

The audit also verifies that all design matrices lie in [0, π] and contain no NaN/Inf values.

---

## 4. Preprocessing

The preprocessing chain is implemented in `src/data/preprocessing.py`.

1. **Column handling.** The 43 raw columns are named using the standard NSL-KDD schema;
   `difficulty` is dropped. Constant columns are detected on train and dropped.
2. **Missing values.** Median imputation, with medians learned on train.
3. **Infinite values.** Replaced with NaN before imputation, then imputed.
4. **Categorical features.** One-hot encoding with a vocabulary learned on train. Categories
   occurring fewer than `min_category_frequency = 2` times in train are collapsed to an
   `__rare__` level. Test-only levels are mapped to the unknown/rare bucket rather than expanding
   the feature space.
5. **Numeric features.** Standardised with train mean/std.
6. **Output.** A 111-column design matrix (after one-hot expansion and constant removal).

No resampling (SMOTE, undersampling) is applied. Class imbalance is handled by class weights and
threshold selection instead, which avoids synthesising network-flow records.

---

## 5. Dimensionality reduction and feature selection

### 5.1 Candidates

Two reducers are evaluated:

- **PCA** (`sklearn.decomposition.PCA`) fitted on the 111-column training matrix.
- **Mutual information** (`mutual_info_classif`) ranking fitted on training rows and labels;
  the top-k columns are retained.

Dimensions: **8, 12 and 16**.

### 5.2 Selection rule

The selection rule is fixed before seeing test results:

1. Fit each candidate on train, transform validation.
2. Train a lightweight logistic-regression probe on the reduced training matrix.
3. Score the probe on validation F1.
4. Prefer the highest validation F1 among candidates with `n_features <= max_quantum_qubits`
   (8) for the main comparison, because the quantum statevector simulation is capped at 8 qubits.

Measured ablation (validation probe, seed 42):

| Method | Dimensions | Accuracy | F1 | ROC-AUC | Explained variance |
|---|---:|---:|---:|---:|---:|
| PCA | 8 | 0.9228 | 0.9161 | 0.9735 | 0.6666 |
| PCA | 12 | 0.9311 | 0.9252 | 0.9785 | 0.7809 |
| PCA | 16 | 0.9356 | 0.9295 | 0.9777 | 0.8813 |
| MI | 8 | 0.8851 | 0.8715 | 0.9557 | — |
| MI | 12 | 0.8840 | 0.8734 | 0.9673 | — |
| MI | 16 | 0.9212 | 0.9132 | 0.9776 | — |

PCA-16 is the best overall on validation, but PCA-8 is the best **feasible** configuration under
the 8-qubit cap. PCA-8 is therefore used for the main comparison and for every model, including
the classical baseline. This is a deliberate, disclosed trade-off: the classical model could
likely improve with 16 components, but then the comparison would no longer use identical inputs.

### 5.3 Shared representation

After reduction, a `FeatureScaler` is fitted on the reduced training matrix and maps each
component to **[0, π]** using training min/max. The same transformed matrix is passed to all
models. No model receives a different feature space, different scaling or a different test set.

For amplitude encoding, each 8-dimensional vector is padded to 8 entries (already a power of two),
normalised to unit L2 norm, and prepared as a 3-qubit state. If the input dimension were not a
power of two, zero-padding to the next power of two is applied before normalisation.

---

## 6. Models

### 6.1 Classical CNN+LSTM

Implemented in `src/classical/cnn_lstm.py` with PyTorch.

The reduced 8-vector is reshaped into a single-channel sequence of length 8. This is a
**pseudo-temporal** representation: the order is the PCA component index (variance order), not
packet timing. No claim of genuine temporal modelling is made.

Architecture:

```
Input (B, 1, 8)
  → BatchNorm1d(1)
  → Conv1d(1, 32, kernel=3, padding=1)
  → ReLU
  → MaxPool1d(2)
  → Dropout(0.3)
  → transpose to (B, 4, 32)
  → LSTM(input=32, hidden=32, layers=1, batch_first=True)
  → last hidden state (B, 32)
  → Linear(32, 32) → ReLU → Dropout(0.3)
  → Linear(32, 1) → logit
```

Training:

- Loss: `BCEWithLogitsLoss` with `pos_weight = n_negative / n_positive` (balanced).
- Optimiser: AdamW, learning rate 1e-3, weight decay 1e-4.
- Batch size: 256.
- Max epochs: 25 (quick profile), early-stopping patience 6 on validation ROC-AUC.
- Scheduler: `ReduceLROnPlateau` (factor 0.5, patience 2, min LR 1e-6).
- Gradient clipping: max norm 1.0.
- Best-epoch checkpoint restored before evaluation.
- Device: CPU (no GPU available on the test machine).

Training size: 8,399 rows. This is the full quick-mode training split.

### 6.2 Quantum SVM

Implemented in `src/quantum/qsvm.py` and `src/quantum/circuits.py`.

**Feature map (angle encoding).** A Qiskit `zz_feature_map` with:

- 8 qubits (one per reduced feature),
- 2 repetitions,
- linear entanglement,
- input features scaled to [0, π].

The statevector \(|\phi(x)\rangle\) is computed exactly with
`qiskit.quantum_info.Statevector`.

**Kernel.** The fidelity kernel is

\[
K(x_i, x_j) = \left|\langle \phi(x_i) \mid \phi(x_j) \rangle\right|^2.
\]

In matrix form, with rows of statevectors \(S\), \(K = |S S^\dagger|^2\). The diagonal is 1
(up to \(6.7 \times 10^{-15}\) numerical error).

**Classifier.** `sklearn.svm.SVC` with:

- `kernel="precomputed"`,
- `C = 1.0`,
- `class_weight="balanced"`,
- `probability=True` (Platt scaling for calibrated scores).

Training size: a stratified 600-row subset of the quick training split; validation size 200.
The subset is drawn with the project seed and stratified on the binary label.

**Kernel validation.** The exact kernel is cross-checked against Qiskit
`FidelityQuantumKernel` with `ComputeUncompute(StatevectorSampler, shots=100000)` on 16 samples.
Measured maximum absolute difference: **0.00171**; tolerance derived from binomial shot noise:
**0.00632**; test passed. The production kernel used for all reported results is the exact
statevector overlap, so reported metrics are shot-noise-free.

### 6.3 Variational Quantum Classifier

Implemented in `src/quantum/vqc.py`.

**Circuit.**

```
|0⟩^⊗8 → ZZ feature map (2 reps, linear) → RealAmplitudes (2 reps, linear) → measure qubit 0
```

- 8 qubits for angle encoding; 3 qubits for amplitude encoding.
- 24 trainable ansatz parameters (angle, 8 qubits); 9 parameters (amplitude, 3 qubits).
- Decision score: \(P(q_0 = 1)\).
- Loss: binary cross-entropy with L2 regularisation (1e-4).
- Optimiser: COBYLA, maxiter 150, initial point scale 0.1.
- Training size: 600 rows; validation 200 rows.

**Fast exact simulation.** Feature states are precomputed once. The ansatz unitary is built with
`Operator(ansatz).data` and applied to the batch of feature states. This is mathematically
equivalent to composing and simulating the full circuit for every parameter evaluation. The
equivalence is covered by `tests/test_quantum_models.py`, which compares the fast path with
direct full-circuit simulation.

**Convergence.** COBYLA used all 151 objective evaluations (150 iterations plus the initial
point) and did **not** satisfy its convergence criterion; final training loss was 0.4585. The VQC
result is therefore a budget-limited result, not a converged optimum. This is reported rather than
hidden.

---

## 7. Training budgets and fairness

| Model | Training rows | Validation rows | Test rows | Feature matrix |
|---|---:|---:|---:|---|
| CNN+LSTM | 8,399 | 1,801 | 1,800 | PCA-8, [0, π] |
| QSVM | 600 | 200 | 1,800 | PCA-8, [0, π] |
| VQC | 600 | 200 | 1,800 | PCA-8, [0, π] |

The test set is identical for every model. The training-budget asymmetry is unavoidable at this
scale: exact statevector evaluation of a 600×600 kernel and a 150-iteration VQC is already the
dominant cost. The asymmetry favours the classical model in terms of data volume, so a quantum
win would have been a strong result; a classical win must be interpreted with that asymmetry in
mind. The comparison answers a practical question: what a CPU-scale statevector experiment can
deliver, not what a fault-tolerant quantum computer might deliver.

All models use the same preprocessing chain (fit on the 8,399 training rows) and the same
post-reduction [0, π] scaling. The quantum subset is drawn from the already-transformed training
matrix, so no separate preprocessing is fitted on the subset.

---

## 8. Decision thresholds

Thresholds are selected on validation data only, maximising F1 over the grid

```
0.05, 0.10, 0.15, ..., 0.95
```

The selected thresholds are then frozen for the common test set:

| Model | Validation-selected threshold | Validation F1 | Test F1 |
|---|---:|---:|---:|
| CNN+LSTM | 0.35 | 0.9572 | 0.9533 |
| QSVM | 0.35 | 0.9485 | 0.9231 |
| VQC | 0.50 | 0.8218 | 0.8028 |

The cost-sensitive objective (missed attack = 10× false alarm) is implemented and can be selected
with `evaluation.threshold_objective=cost`; the reported run uses F1.

---

## 9. Metrics

For each model on the common test set:

- Accuracy, precision, recall/sensitivity, F1, specificity.
- ROC-AUC and average precision (PR-AUC).
- False-positive rate (FPR) and false-negative rate (FNR).
- Confusion matrix (TN, FP, FN, TP).
- MCC, balanced accuracy, NPV, PPV, Brier score, log loss.
- Per-sample scores saved to `results/predictions/*.csv`.

Threshold analysis is produced separately on validation and test grids. The test threshold curve
is diagnostic only; it is not used to choose the operating point.

---

## 10. Statistical analysis

### 10.1 McNemar

For each model pair, the 2×2 table of correct/incorrect decisions on the same test rows is built.
If discordant pairs < 25, the exact binomial test is used; otherwise the chi-square test with
continuity correction. Cohen's h is reported as an effect size. McNemar tests whether the
**pattern of errors** differs, not whether one model is universally better.

### 10.2 Paired bootstrap

1,000 resamples of the test rows, with the same resampled indices applied to both models. The
reported difference is metric(A) − metric(B) at each model's validation-selected threshold for
F1, and threshold-free for ROC-AUC. Percentile 95% CIs are reported. A CI crossing zero means the
difference is not resolved at this sample size.

### 10.3 Wilcoxon across folds

The signed-rank test is applied to per-fold F1 values. With 3 folds the smallest attainable
two-sided p-value is 0.25, so the test is reported as supporting evidence only. It is not used to
claim significance.

### 10.4 Multiplicity

The project reports a small, pre-specified set of pairwise comparisons. No formal family-wise
correction is applied; p-values should be read as descriptive evidence on this benchmark, not as
confirmatory inference for a population of networks.

---

## 11. Experiment registry

| ID | Experiment | Protocol | Primary artefact |
|---|---|---|---|
| E1 | Main comparison | Train all three on identical PCA-8 data; score once on common test | `results/tables/master_comparison.csv` |
| E2 | Cross-validation | 3 stratified folds on train+validation pool; preprocessing refit per fold | `results/tables/cross_validation_summary.csv` |
| E3 | Significance | McNemar, paired bootstrap, Wilcoxon | `results/metrics/statistical_tests.json` |
| E4 | Unseen categories | Remove R2L+U2R entirely; retrain; balanced holdout | `results/tables/unseen_category_results.csv` |
| E5 | External novel types | Score KDDTest+ novel types + normal | `results/tables/external_novel_attack_results.csv` |
| E6 | Encoding comparison | Retrain QSVM/VQC with angle and amplitude, equal optimizer budget | `results/tables/encoding_comparison.csv` |
| E7 | Feature ablation | PCA/MI at 8/12/16; validation logistic probe | `results/tables/feature_reduction_ablation.csv` |
| E8 | Latency/resources | Single-sample and batch timing; circuit depth/size/gates | `results/tables/latency_comparison.csv`, `quantum_resources.csv` |
| E9 | Explainability | Permutation, PCA attribution, SHAP, quantum input sensitivity | `results/explainability/` |
| E10 | Distribution shift | Gaussian noise and feature permutation; file-order windows | `results/tables/drift_summary.csv` |
| E11 | Alert triage | Five-level policy over predicted probabilities | `results/metrics/alert_summary.json` |

Detailed protocols and measured outcomes are in [experiments.md](experiments.md).

---

## 12. Reproducibility

- Seed: 42 (`project.seed`); passed to NumPy, scikit-learn, PyTorch and Qiskit where supported.
- One configuration file: `config/config.yaml`.
- CLI overrides: `--set key.path=value`.
- Saved preprocessing objects: `preprocessor.joblib`, `reducer.joblib`, `scaler.joblib`.
- Saved model objects and metadata: `models/`.
- Environment capture: `results/environment.json`.
- Run manifest: `results/run_manifest.json`.
- Per-run logs: `results/logs/`.
- `python -m pytest -q` covers data contracts, leakage checks, feature reduction, quantum
  encodings/kernels, VQC equivalence, metrics, statistics, alert tiers and pipeline outputs.

Exact timing and some library-level nondeterminism can change the last decimal place on a
different machine. Predictive metrics are deterministic for the saved seed and CPU setup used
here; the VQC optimiser is deterministic given the same Qiskit/SciPy versions.

---

## 13. Terminology and ethics

- **Unseen category** means an entire attack category was removed from training. It is not a
  zero-day claim.
- **Novel type** means an attack label absent from the full KDDTrain+ label set. KDDTest+ also
  differs in feature distribution, so this is a joint novelty + covariate-shift test.
- **Simulation** means exact statevector arithmetic; no quantum hardware, noise model or
  error mitigation is involved.
- The alert engine is an assistive triage demonstration. It is not an autonomous blocking
  system and has not been validated on live traffic.
