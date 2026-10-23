# QE-IDS — Conference Paper Sourcebook

**Verified measurements, manuscript blueprint, and claim guardrails**

Version 1.1 · 22 October 2026 · seed 42 · quick profile only

> **Purpose.** This document is the single source of truth for writing the QE-IDS conference paper.
> Every numeric value below was read from executed artifacts under `results/`, `data/metadata/`, or
> `models/`, or computed directly from the raw NSL-KDD files while preparing this sourcebook.
> The only placeholders are the author names, affiliations, and contact details.
>
> **Scope.** The reported experiments use the **quick profile** only. The full profile defined in
> `config/config.yaml` was **not executed**. All quantum results are exact Qiskit statevector
> simulations. No quantum hardware was used.
>
> **Revision 1.1 — 22 October 2026.** Supersedes the 21 October draft. The external novel-type
> experiment (E5) now scores the already-trained main models with the main preprocessing and their
> validation-selected thresholds. Earlier E5 values produced by a mixed unseen-category/main-
> preprocessing path are invalid and must not be used.

**Companion artifacts**

| Artifact | Role |
|---|---|
| `results/final_summary.json` | Canonical machine-readable results |
| `results/final_summary.md` | Human-readable result summary |
| `results/tables/master_comparison.csv` | Main comparison table |
| `results/metrics/statistical_tests.json` | McNemar, bootstrap, Wilcoxon |
| `results/logs/run_quick_20261021_132234.log` | Clean 14-stage run log |
| `docs/conference_paper.md` | Existing full paper draft using the same values |
| `docs/methodology.md` | Protocol details |
| `docs/limitations.md` | Scope and validity caveats |

**Contents**

- [0. At a glance](#0-at-a-glance)
- [1. Manuscript blueprint](#1-manuscript-blueprint)
- [2. Research question and scope](#2-research-question-and-scope)
- [3. Dataset](#3-dataset)
- [4. Protocol and leakage control](#4-protocol-and-leakage-control)
- [5. Feature reduction](#5-feature-reduction)
- [6. Models and training budgets](#6-models-and-training-budgets)
- [7. Results](#7-results)
- [8. Discussion points](#8-discussion-points)
- [9. Limitations and threats to validity](#9-limitations-and-threats-to-validity)
- [10. Reproducibility](#10-reproducibility)
- [11. Claim guardrails](#11-claim-guardrails)
- [12. Figure inventory with suggested captions](#12-figure-inventory-with-suggested-captions)
- [13. Table and data provenance](#13-table-and-data-provenance)
- [14. References](#14-references)
- [Appendix A. Exact values by result file](#appendix-a-exact-values-by-result-file)
- [Appendix B. LaTeX-ready main table](#appendix-b-latex-ready-main-table)
- [Appendix C. One-page number sheet](#appendix-c-one-page-number-sheet)

---

## 0. At a glance

| Item | Measured value |
|---|---|
| Dataset | NSL-KDD (`KDDTrain+` / `KDDTest+`) |
| Full training file | 125,973 rows; 23 labels; 46.5417% attacks |
| Full external file | 22,544 rows; 38 labels; 56.9242% attacks |
| Quick pool | 12,000 stratified rows from `KDDTrain+` |
| Split sizes | 8,399 train / 1,801 validation / 1,800 common test |
| Common test composition | 962 Normal / 838 Attack |
| Shared representation | PCA-8, 66.6556% explained variance, rescaled to `[0, π]` |
| CNN+LSTM | F1 **0.9533**, ROC-AUC **0.9891**, FPR **0.0374** |
| QSVM | F1 **0.9231**, ROC-AUC **0.9903**, FPR **0.1185** |
| VQC | F1 **0.8028**, ROC-AUC **0.8982**, FPR **0.3025** |
| CNN − QSVM F1 | **0.0302** (95% CI 0.0166–0.0434) |
| CNN − QSVM ROC-AUC | **−0.0012** (95% CI −0.0061 to 0.0030; crosses zero) |
| McNemar, CNN vs QSVM | **p = 2.389e-06**, Cohen's h = 0.138 (negligible) |
| 3-fold CV F1 | CNN 0.9346 ± 0.0135; QSVM 0.9180 ± 0.0161; VQC 0.7452 ± 0.0527 |
| Unseen-category recall (R2L+U2R holdout) | VQC 0.7619; CNN 0.0476; QSVM 0.0119 |
| External novel-type F1 | QSVM 0.6572; VQC 0.5189; CNN 0.4451 |
| External novel-type ROC-AUC | QSVM 0.8917; VQC 0.7801; CNN 0.6665 |
| Single-sample latency | CNN 0.9323 ms; QSVM 3.9535 ms (4.28×); VQC 32.7402 ms (35.51×) |
| Verdict | **No quantum advantage on the primary common test; QSVM leads the secondary external novel-type subset** |

The classical CNN+LSTM is the strongest model on the primary common test for F1, accuracy, precision,
MCC, Brier score, log loss and false-positive rate. The QSVM is the best quantum model by F1 and has
the highest recall and the highest ROC-AUC on that test, but its AUC advantage is not resolved and it
raises substantially more false alarms. The VQC is the weakest model on the main task and did not
converge within its optimisation budget.

On the secondary external novel-type subset the ordering changes: the QSVM leads by F1 (0.6572) and
ROC-AUC (0.8917), while the CNN retains the lowest FPR (0.0807) and the highest precision (0.6234).
This is a scope-specific result under a single external file and strong covariate shift, not a
general quantum advantage.

### 0.1 Reporting conventions

- Rates and metrics are reported to four decimal places in the main text. Appendix A retains more
  precision for audit.
- Percentages are used only for readability; the underlying fraction is the measured value.
- p-values are reported in scientific notation. No p-value is rounded to zero.
- Wall-clock times are reported in seconds to two decimal places; latency is reported in
  milliseconds to four decimal places; throughput is reported to one decimal place.
- All quantum metrics are exact statevector results, not hardware results. Do not add a noise
  model or a hardware qualifier that is not in the artifacts.
- Do not substitute full-profile numbers. The full profile was not executed.
- The draft mixes British and American spellings in a few places. Harmonise to the venue's style
  during copy-editing; do not change code identifiers or config keys.

---

## 1. Manuscript blueprint

### 1.1 Title options

1. **QE-IDS: A Leakage-Audited Comparison of Classical and Quantum Classifiers for Network
   Intrusion Detection** — preferred; balanced and descriptive.
2. **No Quantum Advantage on the Primary Task Under an Eight-Qubit Budget: A Controlled NSL-KDD
   Study of CNN+LSTM, QSVM, and VQC** — precise about the scope of the negative result.
3. **Fair Comparison of Classical and Quantum Intrusion Detection on NSL-KDD: A Negative Primary
   Result** — direct, but less specific about the protocol.

### 1.2 Author block

Use real names and affiliations only. Suggested structure:

```text
[Author 1], [Author 2], [Author 3]
[Department, Institution, City, Country]
[email addresses]
```

These are the only placeholders in this document. Do not invent author details.

### 1.3 Paste-ready abstract

Quantum machine learning is often proposed as a route to stronger network intrusion detection, but
reported gains are difficult to interpret when classical baselines, feature representations, splits
and evaluation protocols differ. We present a controlled, leakage-audited comparison on NSL-KDD
between a classical CNN+LSTM, a Quantum Support Vector Machine (QSVM) using an exact fidelity
quantum kernel, and a Variational Quantum Classifier (VQC). All three models consume the same
eight-component PCA representation, rescaled to `[0, π]` using training statistics, and are scored
on the same 1,800-row common test set; decision thresholds are selected on validation data only.
The CNN+LSTM achieves F1 = 0.9533 and ROC-AUC = 0.9891; the QSVM achieves F1 = 0.9231 and
ROC-AUC = 0.9903; the VQC achieves F1 = 0.8028 and ROC-AUC = 0.8982. The CNN+LSTM's F1 advantage
over the QSVM is 0.0302 (paired-bootstrap 95% CI 0.0166–0.0434), while the ROC-AUC difference of
−0.0012 is unresolved (95% CI −0.0061 to 0.0030). McNemar's test rejects equal error patterns
(p = 2.389 × 10⁻⁶), but the effect size is negligible (Cohen's h = 0.138). We also report
cross-validation, unseen attack-category generalisation, external novel attack types, angle versus
amplitude encoding, latency, circuit resources, explainability, controlled distribution shift and a
tiered alert policy. No quantum advantage is observed on the primary common test. On the secondary
external novel-type subset, the QSVM leads by F1 (0.6572 vs 0.4451 for the CNN) and ROC-AUC (0.8917
vs 0.6665), a scope-specific nuance rather than a general advantage. All quantum results come from
exact statevector simulation and represent a shot-noise-free upper bound relative to noisy hardware,
not a hardware claim.

**Keywords:** intrusion detection, NSL-KDD, quantum machine learning, quantum kernel, variational
quantum classifier, CNN-LSTM, reproducibility, negative results.

### 1.4 Contributions

1. A leakage-audited protocol in which imputation, one-hot encoding, standardisation, PCA or
   mutual-information reduction, and quantum angle rescaling are fitted exclusively on training
   data. Eight automated audit checks pass.
2. A three-way comparison on the same PCA-8 matrix and the same 1,800 test rows, with
   validation-only threshold and feature-configuration selection.
3. An unseen attack-category experiment that removes R2L and U2R from training entirely, plus an
   external novel-type test on `KDDTest+` that reveals a scope-specific QSVM advantage.
4. A systems analysis of cross-validation, paired statistics, latency, circuit resources,
   explainability, controlled distribution shift and tiered alert triage.
5. Explicit reporting of the negative primary result, the scope-specific external exception, and the
   simulation-only limits.

### 1.5 Suggested section and word budget

| Section | Suggested words | Core content |
|---|---:|---|
| Abstract | 220–250 | Question, protocol, three headline results, statistical nuance, primary negative verdict, external exception |
| 1. Introduction | 700–900 | IDS problem, QML promise, reproducibility gap, research question, contributions |
| 2. Related work | 600–800 | Quantum kernels, VQCs, prior IDS comparisons, NSL-KDD limitations |
| 3. Dataset and protocol | 1,200–1,500 | Dataset facts, splits, leakage audit, preprocessing, PCA-8, thresholds |
| 4. Models | 1,200–1,500 | CNN+LSTM, QSVM, VQC, budgets, hyperparameters, simulation details |
| 5. Results | 2,000–2,500 | Main table, CV, paired tests, unseen category, external novel, encoding, latency, resources, explainability, drift |
| 6. Discussion | 800–1,000 | Why the classical model wins here, AUC nuance, unseen-category nuance, external novel-type nuance, resource caveats |
| 7. Limitations | 400–600 | Quick profile, budget asymmetry, 8-qubit cap, simulation, single seed, benchmark age |
| 8. Conclusion | 200–300 | Primary negative result, scope-specific external exception, methodological contribution, next steps |
| References | 16 entries | See Section 14 |

### 1.6 Figure and table plan

**Suggested figures**

| No. | Figure | File |
|---:|---|---|
| 1 | System and evaluation architecture | `docs/architecture.png` |
| 2 | Class distribution in the quick pool | `results/figures/class_distribution.png` |
| 3 | PCA explained variance | `results/figures/pca_explained_variance.png` |
| 4 | Reduced feature space | `results/figures/reduced_feature_space.png` |
| 5 | CNN+LSTM training curves | `results/figures/cnn_lstm_training_curves.png` |
| 6 | ROC comparison | `results/figures/roc_comparison.png` |
| 7 | Precision–recall comparison | `results/figures/pr_comparison.png` |
| 8 | Confusion matrices | `results/figures/confusion_matrices.png` |
| 9 | 3-fold cross-validation F1 | `results/figures/cross_validation_f1.png` |
| 10 | Unseen-category performance | `results/figures/unseen_attack_performance.png` |
| 11 | Angle versus amplitude encoding | `results/figures/encoding_comparison.png` |
| 12 | Latency comparison | `results/figures/latency_comparison.png` |
| 13 | Quantum circuit resources | `results/figures/quantum_resources.png` |
| 14 | VQC convergence | `results/figures/vqc_convergence.png` |
| 15 | Permutation importance | `results/figures/permutation_importance.png` |
| 16 | SHAP summary | `results/explainability/shap_summary.png` |
| 17 | Drift under Gaussian noise | `results/figures/drift_gaussian_noise.png` |
| 18 | Drift under feature permutation | `results/figures/drift_feature_permutation.png` |
| 19 | Simulated file-order windows | `results/figures/drift_simulated_windows.png` |

**Suggested tables**

| No. | Table | Source |
|---:|---|---|
| 1 | Dataset composition and splits | Section 3 of this document |
| 2 | Feature-reduction ablation | `results/tables/feature_reduction_ablation.csv` |
| 3 | Model and training configuration | Section 6 |
| 4 | Main common-test comparison | `results/tables/master_comparison.csv` |
| 5 | Confusion matrices and per-category detection | `results/metrics/*.json` |
| 6 | Cross-validation | `results/tables/cross_validation_summary.csv` |
| 7 | McNemar and paired bootstrap | `results/metrics/statistical_tests.json` |
| 8 | Unseen-category holdout | `results/tables/unseen_category_results.csv` |
| 9 | External novel attack types | `results/tables/external_novel_attack_results.csv` |
| 10 | Encoding comparison | `results/tables/encoding_comparison.csv` |
| 11 | Latency and circuit resources | `results/tables/latency_comparison.csv`, `results/tables/quantum_resources.csv` |
| 12 | Explainability summary | `results/explainability/explainability_summary.json` |
| 13 | Controlled distribution shift | `results/tables/drift_controlled.csv` |
| 14 | Tiered alert triage | `results/metrics/alert_summary.json` |

---

## 2. Research question and scope

**Research question.** Given identical reduced inputs, identical splits, identical common test
samples and an identical evaluation protocol, does a quantum classifier improve binary intrusion
detection over a reasonably configured classical CNN+LSTM on NSL-KDD?

**Operational questions.**

1. Which model has the best F1, accuracy, MCC and false-positive rate on the common test set?
2. Are pairwise differences resolved by paired tests and bootstrap confidence intervals?
3. Do the models generalise to attack categories removed from training?
4. Do they generalise to novel attack types in the external `KDDTest+` file?
5. What are the latency and circuit-resource costs under exact statevector simulation?

**In scope**

- Binary classification: Normal vs Attack.
- Offline batch evaluation.
- Exact statevector simulation for all quantum models.
- Leakage-controlled preprocessing and validation-only tuning.

**Out of scope**

- Quantum-hardware performance, noise, decoherence, readout error or transpilation costs.
- Zero-day detection claims.
- Real-time inline deployment.
- Modern enterprise traffic.
- Exhaustive classical or quantum hyperparameter search.
- Full-profile NSL-KDD results.

---

## 3. Dataset

### 3.1 Files and verified properties

| File | Rows | Attack rate | Distinct labels | Role |
|---|---:|---:|---:|---|
| `KDDTrain+.txt` | 125,973 | 46.5417% | 23 | Training pool only |
| `KDDTest+.txt` | 22,544 | 56.9242% | 38 | External, never-trained-on evaluation |

Each row has 41 features (3 categorical: `protocol_type`, `service`, `flag`; 38 numeric), a label,
and a `difficulty` score. The difficulty column is dropped. Labels are lower-cased and mapped to the
five NSL-KDD categories (Normal, DoS, Probe, R2L, U2R) and to a binary `is_attack` target.

Verified properties:

- 23 distinct labels in `KDDTrain+`, 38 in `KDDTest+`.
- 17 attack types appear only in `KDDTest+`.
- Zero missing values and zero exact duplicate rows in both files.
- 43 raw columns: 41 features, label, difficulty.
- The external file is never used for training, early stopping, threshold selection, feature
  selection or probability calibration.

### 3.2 Full dataset composition

Category counts computed directly from the raw files:

| Category | `KDDTrain+` | `KDDTest+` |
|---|---:|---:|
| Normal | 67,343 | 9,711 |
| DoS | 45,927 | 7,460 |
| Probe | 11,656 | 2,421 |
| R2L | 995 | 2,885 |
| U2R | 52 | 67 |
| **Total** | **125,973** | **22,544** |

### 3.3 Quick pool and splits

Quick mode draws a stratified 12,000-row subsample of `KDDTrain+` and splits it 70/15/15 on the
binary label:

| Split | Rows | Normal | Attack | Attack rate | DoS | Probe | R2L | U2R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Quick pool | 12,000 | 6,415 | 5,585 | 46.5417% | 4,465 | 1,017 | 97 | 6 |
| Train (70%) | 8,399 | 4,490 | 3,909 | 46.5413% | 3,125 | 717 | 62 | 5 |
| Validation (15%) | 1,801 | 963 | 838 | 46.5297% | 662 | 159 | 16 | 1 |
| Common test (15%) | 1,800 | 962 | 838 | 46.5556% | 678 | 141 | 19 | 0 |

The quick pool contains 18 attack types. The test split has no U2R rows, so per-category U2R
detection is not measurable on the common test set. The test split is the common test set for all
three models.

Profile settings from `config/config.yaml`:

| Setting | Quick (executed) | Full (not executed) |
|---|---:|---:|
| Pool size | 12,000 | all 125,973 |
| Quantum training rows | 600 | 3,000 |
| Quantum validation rows | 200 | 600 |
| Common test rows | 1,800 | 2,500 |
| CV folds | 3 | 5 |
| CV pool size | 3,000 | 15,000 |
| CV quantum training rows | 300 | 800 |
| CNN maximum epochs | 25 | 40 |

The full profile was **not executed**, and no full-profile metrics are reported.

### 3.4 External set and novelty definition

The external experiment scores `KDDTest+` rows whose attack type is absent from the **full**
`KDDTrain+` label set. Novelty is deliberately defined against the full training file, not the
quick subsample, so rare training types are not mislabelled as novel. The external rows are
transformed with the main experiment's fitted preprocessing, PCA and scaling, then scored by the
already-trained main models at their original validation-selected thresholds. No external row is
used for fitting, calibration or threshold selection.

| Item | Value |
|---|---:|
| Novel attack types | 17 |
| Novel attack rows | 3,750 |
| Normal rows | 9,711 |
| Total evaluated rows | 13,461 |

The 17 novel types are: `apache2`, `httptunnel`, `mailbomb`, `mscan`, `named`, `processtable`,
`ps`, `saint`, `sendmail`, `snmpgetattack`, `snmpguess`, `sqlattack`, `udpstorm`, `worm`, `xlock`,
`xsnoop`, `xterm`.

`KDDTest+` also differs in feature distribution from `KDDTrain+`. The external result therefore
measures novel-type generalisation **under covariate shift**, not an isolated zero-day effect.

---

## 4. Protocol and leakage control

### 4.1 Executed pipeline order

1. Load raw NSL-KDD, assign the standard 43-column schema, drop `difficulty`.
2. Detect and drop constant columns on the training split.
3. Replace infinite values with NaN; median-impute using training medians.
4. One-hot encode categorical features with a training-fit vocabulary; collapse training levels
   occurring fewer than 2 times into a rare bucket; map unseen test levels to the unknown/rare
   bucket.
5. Standardise numeric features with training mean and standard deviation.
6. Fit PCA (or mutual-information ranking) on the training matrix only.
7. Rescale the reduced components to `[0, π]` using training min/max.
8. Train each model; use validation data for early stopping, threshold selection and
   feature-configuration selection.
9. Score the common test set once, after all choices are frozen.

The resulting design matrix has **111 columns** after one-hot expansion and constant removal.

### 4.2 Eight-check leakage audit

Source: `data/metadata/leakage_audit.json`.

| Check | What it verifies | Result |
|---|---|---|
| `scaler_fit_on_train` | Recomputed scaler statistics on train match the saved scaler | PASSED |
| `onehot_vocabulary_from_train` | No test-only categorical level expanded the vocabulary | PASSED |
| `transform_is_stateless` | Repeated transforms of raw rows are identical | PASSED |
| `reducer_input_dimension` | Reducer receives the expected 111-column matrix | PASSED |
| `reducer_output_dimension` | Reducer emits exactly 8 components | PASSED |
| `no_exact_train_test_row_overlap` | Zero exact row overlap across splits | PASSED |
| `reduced_shapes` | Train/validation/test shapes are consistent | PASSED |
| `pca_explained_variance_monotone` | PCA explained-variance curve is monotone | PASSED |

Additional audit results:

- Split overlaps: train–validation 0, train–test 0, validation–test 0.
- Union size: 12,000, equal to the expected total.
- All design matrices lie in `[0, π]` and contain no NaN or Inf.
- Scaler recomputation maximum absolute difference: 0.0.
- No unseen test levels in the one-hot vocabulary.

### 4.3 Design-matrix statistics

| Matrix | Shape | Min | Max | Mean | Std |
|---|---|---:|---:|---:|---:|
| `X_train` | 8,399 × 8 | 0.0 | 3.1415927 | 1.2150776 | 0.8013405 |
| `X_validation` | 1,801 × 8 | 0.0 | 3.1415927 | 1.2241240 | 0.8001612 |
| `X_test` | 1,800 × 8 | 0.0 | 3.1415927 | 1.2260147 | 0.8010823 |

### 4.4 Validation-only decisions

Validation data selects:

- the CNN+LSTM early-stopping epoch and best checkpoint,
- the decision threshold for every model,
- the feature method and dimension,
- the encoding-comparison configuration.

The test set is not used for any of these choices. Threshold curves on the test set are generated
for diagnostic plots only and are not used to select operating points.

---

## 5. Feature reduction

### 5.1 Candidates and selection rule

Two reducers are compared at 8, 12 and 16 dimensions:

- **PCA**, fitted on the 111-column training matrix.
- **Mutual-information ranking**, fitted on training rows and labels; the top-k columns are kept.

A lightweight logistic-regression probe is trained on each reduced training matrix and scored on
validation F1. The selection rule is fixed before test results:

1. Prefer the highest validation F1 among configurations with `n_features <= 8`, because the
   reported experiment is capped at 8 qubits (one qubit per reduced feature) to keep exact
   statevector simulation tractable.
2. Use the same selected representation for **every** model, including the classical baseline.

PCA-16 has the best overall validation F1, but PCA-8 is the best feasible configuration under the
8-qubit cap. PCA-8 is therefore used for the main comparison.

### 5.2 Measured ablation

| Method | Dimensions | Accuracy | F1 | ROC-AUC | Explained variance |
|---|---:|---:|---:|---:|---:|
| PCA | 8 | 0.9228 | 0.9161 | 0.9735 | 0.6666 |
| PCA | 12 | 0.9311 | 0.9252 | 0.9785 | 0.7809 |
| PCA | 16 | **0.9356** | **0.9295** | 0.9777 | 0.8813 |
| Mutual information | 8 | 0.8851 | 0.8715 | 0.9557 | — |
| Mutual information | 12 | 0.8840 | 0.8734 | 0.9673 | — |
| Mutual information | 16 | 0.9212 | 0.9132 | 0.9776 | — |

These are validation-probe results used for feature selection, not the main model test results.
The PCA-8 validation F1 is 0.0134 lower than PCA-16 (1.34 F1 points). This is the measured cost of
the simulator cap and is disclosed as a design constraint, not a post-hoc choice.

### 5.3 PCA explained variance

| Component | Explained variance ratio | Cumulative |
|---|---:|---:|
| PC1 | 0.208676 | 0.208676 |
| PC2 | 0.139389 | 0.348065 |
| PC3 | 0.086447 | 0.434512 |
| PC4 | 0.055910 | 0.490423 |
| PC5 | 0.053996 | 0.544418 |
| PC6 | 0.049726 | 0.594145 |
| PC7 | 0.039123 | 0.633268 |
| PC8 | 0.033289 | **0.666556** |

### 5.4 Shared representation

After reduction, a `FeatureScaler` fitted on the reduced training matrix maps each component to
`[0, π]` using training min/max. The same transformed matrix is passed to all models. No model
receives a different feature space, different scaling or a different test set.

For amplitude encoding, each 8-dimensional vector is already a power of two, normalised to unit
L2 norm and prepared as a 3-qubit state. If the input dimension were not a power of two,
zero-padding to the next power of two would be applied before normalisation.

---

## 6. Models and training budgets

### 6.1 CNN+LSTM

The reduced 8-vector is reshaped into a single-channel sequence of length 8. This is a
**pseudo-temporal** representation: the order is the PCA component index, not packet timing. No
claim of genuine temporal modelling is made.

Architecture:

```text
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

| Setting | Value |
|---|---|
| Trainable parameters | 9,667 |
| Training rows | 8,399 |
| Validation rows | 1,801 |
| Loss | `BCEWithLogitsLoss` with `pos_weight = n_negative / n_positive` |
| Optimiser | AdamW, learning rate 1e-3, weight decay 1e-4 |
| Batch size | 256 |
| Maximum epochs | 25 |
| Epochs completed | 25 (no early stop) |
| Best epoch | 24 |
| Best validation ROC-AUC | 0.9902 |
| Positive-class weight | 1.1486 |
| Early stopping | Patience 6 on validation ROC-AUC |
| Scheduler | `ReduceLROnPlateau` (factor 0.5, patience 2, min LR 1e-6) |
| Gradient clipping | Max norm 1.0 |
| Device | CPU |
| Reported training time | 9.56 s |
| Best-epoch checkpoint | Restored before evaluation |

The clean run completed all 25 requested epochs (`stopped_early = false`) and restored the epoch-24
checkpoint before test scoring.

### 6.2 QSVM

The QSVM uses an angle-encoded ZZ feature map and an exact statevector fidelity kernel.

| Setting | Value |
|---|---|
| Feature map | `zz_feature_map`, 8 qubits, 2 repetitions, linear entanglement |
| Input scaling | `[0, π]`, training-fit |
| Kernel | \(K(x_i,x_j) = \vert\langle\phi(x_i)\vert\phi(x_j)\rangle\vert^2\) |
| Matrix form | \(K = \vert S S^\dagger\vert^2\) for statevector rows \(S\) |
| Kernel diagonal error | Up to \(6.7 \times 10^{-15}\) |
| Classifier | `sklearn.svm.SVC`, `kernel="precomputed"`, `C=1.0` |
| Class weighting | Balanced |
| Probability | Platt scaling via `probability=True` |
| Training rows | 600 (stratified subset of the 8,399 training rows) |
| Validation rows | 200 |
| Circuit depth | 31 |
| Circuit size | 74 |
| Two-qubit gates | 28 |
| Circuit parameters | 8 feature-map inputs; no trainable circuit parameters |
| Reported training time | 1.72 s |
| Reported batch inference | 2.7440 ms/sample |
| Backend | `qiskit.quantum_info.Statevector` (exact fidelity kernel) |

**Kernel validation.** The exact kernel is cross-checked against Qiskit
`FidelityQuantumKernel` with `ComputeUncompute(StatevectorSampler, shots=100000)` on 16 samples.
Measured maximum absolute difference: **0.00171058**; tolerance derived from binomial shot noise:
**0.00632456**; test passed. The production kernel used for all reported metrics is the exact
statevector overlap, so reported results are shot-noise-free.

### 6.3 VQC

The VQC uses the same ZZ feature map followed by a RealAmplitudes ansatz.

| Setting | Value |
|---|---|
| Feature map | `zz_feature_map`, 8 qubits, 2 repetitions, linear entanglement |
| Ansatz | `real_amplitudes`, 2 repetitions, linear entanglement |
| Trainable parameters | 24 |
| Decision score | \(P(q_0 = 1)\) |
| Loss | Binary cross-entropy with L2 regularisation 1e-4 |
| Optimiser | COBYLA, `maxiter=150` |
| Initial point scale | 0.1 |
| Objective evaluations | 151 |
| Final training loss | 0.4585 |
| Converged | **false** |
| Training rows | 600 |
| Validation rows | 200 |
| Circuit depth | 38 |
| Circuit size | 112 |
| Two-qubit gates | 42 |
| Reported training time | 8.61 s |
| Reported batch inference | 2.8822 ms/sample |
| Backend | `qiskit.quantum_info.Statevector` (analytic probabilities) |

**Fast exact simulation.** Feature states are precomputed once. The ansatz unitary is built with
`Operator(ansatz).data` and applied to the batch of feature states. This is mathematically
equivalent to composing and simulating the full circuit for every parameter evaluation. The
equivalence is covered by `tests/test_quantum_models.py`, which compares the fast path with direct
full-circuit simulation.

**Convergence.** COBYLA used all 151 objective evaluations (`maxiter=150`) and did not satisfy its
convergence criterion. The VQC result is a budget-limited result, not a converged optimum.

### 6.4 Training-budget asymmetry

| Model | Training rows | Validation rows | Common test rows | Feature matrix |
|---|---:|---:|---:|---|
| CNN+LSTM | 8,399 | 1,801 | 1,800 | PCA-8, `[0, π]` |
| QSVM | 600 | 200 | 1,800 | PCA-8, `[0, π]` |
| VQC | 600 | 200 | 1,800 | PCA-8, `[0, π]` |

The test set is identical for every model. The training-budget asymmetry is forced by exact
statevector cost: a 600×600 kernel and a 150-iteration VQC are large enough to motivate a bounded
budget. The asymmetry favours the classical model in data volume, so a quantum win would have been a
strong result; a classical win must be interpreted with this constraint in mind. The comparison is
fair in evaluation, not in training data volume.

### 6.5 Thresholds

Thresholds are selected on validation F1 over the grid
`0.05, 0.10, 0.15, ..., 0.95` and then frozen for the common test set.

| Model | Selected threshold | Validation F1 | Test F1 | Test FPR |
|---|---:|---:|---:|---:|
| CNN+LSTM | 0.35 | 0.9572 | 0.9533 | 0.0374 |
| QSVM | 0.35 | 0.9485 | 0.9231 | 0.1185 |
| VQC | 0.50 | 0.8218 | 0.8028 | 0.3025 |

A cost-sensitive objective (missed attack = 10× false alarm) is implemented and can be selected
with `evaluation.threshold_objective=cost`; the reported run uses F1.

---

## 7. Results

### 7.1 Main comparison on the common test set

All values are at each model's validation-selected threshold. Common test set: 1,800 rows
(962 Normal, 838 Attack).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR | MCC | Brier | Log loss | Avg. precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **CNN+LSTM** | **0.9567** | **0.9567** | 0.9499 | **0.9533** | 0.9891 | **0.0374** | 0.0501 | **0.9129** | **0.0308** | **0.1124** | **0.9907** |
| QSVM | 0.9244 | 0.8774 | **0.9737** | 0.9231 | **0.9903** | 0.1185 | **0.0263** | 0.8537 | 0.0419 | 0.1398 | 0.9898 |
| VQC | 0.7933 | 0.7223 | 0.9033 | 0.8028 | 0.8982 | 0.3025 | 0.0967 | 0.6077 | 0.1527 | 0.4784 | 0.8807 |

**Confusion matrices**

| Model | Threshold | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|
| CNN+LSTM | 0.35 | 926 | 36 | 42 | 796 |
| QSVM | 0.35 | 848 | 114 | 22 | 816 |
| VQC | 0.50 | 671 | 291 | 81 | 757 |

**Per-category detection on the common test set**

| Model | Normal FPR | DoS detection | Probe detection | R2L detection |
|---|---:|---:|---:|---:|
| CNN+LSTM | 0.0374 | 0.9690 (657/678) | 0.9645 (136/141) | 0.1579 (3/19) |
| QSVM | 0.1185 | 0.9823 (666/678) | 0.9716 (137/141) | 0.6842 (13/19) |
| VQC | 0.3025 | 0.9454 (641/678) | 0.7021 (99/141) | 0.8947 (17/19) |

The test split contains no U2R rows, so U2R detection is not measurable on the common test set.
The R2L column is based on only 19 rows and should not be over-interpreted.

### 7.2 Interpretation of the main result

- The CNN+LSTM has the best F1 (0.9533), accuracy (0.9567), precision (0.9567), MCC (0.9129),
  Brier score (0.0308), log loss (0.1124), average precision (0.9907) and FPR (0.0374).
- The QSVM has the best recall (0.9737) and the lowest FNR (0.0263), missing 22 attacks versus 42
  for the CNN. It also has the highest ROC-AUC (0.9903), but by only 0.0012.
- The QSVM raises 114 false positives versus 36 for the CNN, a 3.17× higher false-positive count.
- The VQC has the weakest F1 (0.8028) and the highest FPR (0.3025), despite respectable recall
  (0.9033).
- The CNN's F1 advantage over the QSVM is 0.0302 (3.02 points). The accuracy advantage is 0.0322
  (3.22 points).
- The QSVM's AUC advantage is 0.0012, but its average precision is lower than the CNN's
  (0.9898 vs 0.9907), so it is not a clean ranking win.

### 7.3 Cross-validation

Protocol: 3 stratified folds on a 3,000-row pool from the training+validation data. Each fold uses
1,760 training rows, 240 internal validation rows and 1,000 test rows. The CNN uses all 1,760 fold
training rows; the quantum models use a stratified 300-row subset of those rows. Preprocessing,
feature reduction and quantum rescaling are refit inside every fold on fold-training data only.
Early stopping uses an internal validation split carved out of the fold-training data. The held-out
fold is used only for scoring.

CV classification metrics use the fixed 0.5 decision threshold, not a per-fold
validation-selected threshold. This differs from the main common-test protocol and is a limitation
of the CV comparison; ROC-AUC is threshold-free.

| Model | F1 | Accuracy | ROC-AUC | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| CNN+LSTM | **0.9346 ± 0.0135**<br>[0.9010, 0.9682] | **0.9400 ± 0.0123**<br>[0.9095, 0.9705] | **0.9909 ± 0.0025**<br>[0.9846, 0.9971] | **0.9219 ± 0.0144**<br>[0.8862, 0.9577] | 0.9477 ± 0.0129<br>[0.9157, 0.9797] |
| QSVM | 0.9180 ± 0.0161<br>[0.8779, 0.9580] | 0.9283 ± 0.0120<br>[0.8985, 0.9582] | 0.9869 ± 0.0013<br>[0.9837, 0.9901] | 0.8654 ± 0.0410<br>[0.7636, 0.9671] | **0.9788 ± 0.0166**<br>[0.9376, 1.0200] |
| VQC | 0.7452 ± 0.0527<br>[0.6144, 0.8761] | 0.7613 ± 0.0580<br>[0.6171, 0.9055] | 0.8575 ± 0.0539<br>[0.7235, 0.9914] | 0.7457 ± 0.0359<br>[0.6566, 0.8349] | 0.7462 ± 0.0756<br>[0.5585, 0.9340] |

Per-fold F1 values:

| Model | Fold 1 | Fold 2 | Fold 3 |
|---|---:|---:|---:|
| CNN+LSTM | 0.9500 | 0.9293 | 0.9246 |
| QSVM | 0.9012 | **0.9333** | 0.9194 |
| VQC | 0.6906 | 0.7494 | 0.7957 |

The mean ordering matches the main test set: CNN > QSVM > VQC. The CNN–QSVM gap is small, and the
QSVM is slightly higher in fold 2 (0.9333 vs 0.9293). The CV AUC ordering also reverses relative to
the single common test set: CNN 0.9909 > QSVM 0.9869. The QSVM's 0.0012 test AUC advantage
therefore does not replicate in the 3-fold CV.

The QSVM precision confidence interval upper bound exceeds 1.0 because a symmetric Wald-style
interval is used; precision is bounded at 1.0. Report the point estimate and note the interval is
approximate.

### 7.4 McNemar tests

Method: chi-square with continuity correction when discordant pairs are at least 25; exact binomial
otherwise. All three comparisons here use the chi-square form.

| Comparison | CNN-only correct (n10) | Other-only correct (n01) | Both wrong (n00) | Both correct (n11) | Discordant | McNemar χ² | p-value | Odds ratio | Cohen's h | Effect |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CNN vs QSVM | 102 | 44 | 34 | 1,620 | 146 | 22.2534 | **2.389e-06** | 2.3182 | 0.1375 | negligible |
| CNN vs VQC | 347 | 53 | 25 | 1,375 | 400 | 214.6225 | **1.347e-48** | 6.5472 | 0.5245 | medium |
| QSVM vs VQC | 299 | 63 | 73 | 1,365 | 362 | 152.5552 | **4.792e-35** | 4.7460 | 0.3869 | small |

McNemar tests whether the **pattern of errors** differs. A significant result does not by itself
establish a practically meaningful advantage; the CNN–QSVM effect size is negligible.

### 7.5 Paired bootstrap

1,000 resamples of the test rows; the same resampled indices are applied to both models.
Percentile 95% confidence intervals. F1 is evaluated at each model's validation-selected threshold;
ROC-AUC is threshold-free.

| Comparison | Metric | Difference (A − B) | 95% CI | Crosses zero |
|---|---|---:|---|---|
| CNN vs QSVM | F1 | **0.0302** | [0.0166, 0.0434] | no |
| CNN vs QSVM | ROC-AUC | −0.0012 | [−0.0061, 0.0030] | **yes** |
| CNN vs VQC | F1 | 0.1505 | [0.1308, 0.1712] | no |
| CNN vs VQC | ROC-AUC | 0.0909 | [0.0768, 0.1055] | no |
| QSVM vs VQC | F1 | 0.1203 | [0.1026, 0.1384] | no |
| QSVM vs VQC | ROC-AUC | 0.0921 | [0.0792, 0.1074] | no |

### 7.6 Wilcoxon signed-rank across folds

| Comparison | Metric | Folds | Statistic | p-value | Significant at 0.05 | Mean difference |
|---|---|---:|---:|---:|---|---:|
| CNN vs QSVM | F1 | 3 | 1.0 | 0.5000 | no | 0.0166 |
| CNN vs VQC | F1 | 3 | 0.0 | 0.2500 | no | 0.1894 |
| QSVM vs VQC | F1 | 3 | 0.0 | 0.2500 | no | 0.1727 |

With 3 folds, the minimum attainable two-sided p-value is 0.25. The Wilcoxon results are
underpowered by construction and are reported as supporting evidence only, not as significance
claims.

### 7.7 Unseen attack-category generalisation

R2L and U2R are removed from training entirely. Models are retrained on Normal + DoS + Probe and
scored on a balanced holdout of 84 Normal and 84 held-out-category attacks (168 rows: 78 R2L and
6 U2R). The retrain split contains 8,598 training rows and 1,518 validation rows.

| Model | Train rows | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | MCC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN+LSTM | 8,598 | 0.30 | 0.5119 | 0.6667 | 0.0476 | 0.0889 | 0.7484 | 0.0238 | 0.0642 | 82 | 2 | 80 | 4 |
| QSVM | 600 | 0.50 | 0.4940 | 0.3333 | 0.0119 | 0.0230 | 0.6559 | 0.0238 | −0.0449 | 82 | 2 | 83 | 1 |
| VQC | 600 | 0.45 | **0.7976** | **0.8205** | **0.7619** | **0.7901** | **0.8707** | 0.1667 | **0.5968** | 70 | 14 | 20 | 64 |

Per-category recall:

| Model | R2L (n = 78) | U2R (n = 6) |
|---|---:|---:|
| CNN+LSTM | 0.0385 (3/78) | 0.1667 (1/6) |
| QSVM | 0.0000 (0/78) | 0.1667 (1/6) |
| VQC | 0.7821 (61/78) | 0.5000 (3/6) |

The VQC's advantage is large on this holdout but fragile: 168 rows, one seed, a 16.7% false-positive
rate, and a threshold selected on a small held-out-category validation split. It is **unseen
attack-category generalisation**, not a zero-day detection claim.

### 7.8 External novel attack types

Novelty is defined against the full `KDDTrain+` label set. The evaluation subset contains 17 novel
types, 3,750 novel attacks and 9,711 Normal rows (13,461 total). Known attack types are excluded.

The external rows use the main models' validation-selected thresholds: 0.35 for the CNN and QSVM,
0.50 for the VQC.

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | MCC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN+LSTM | 0.35 | 0.7596 | **0.6234** | 0.3461 | 0.4451 | 0.6665 | **0.0807** | 0.3290 | 8,927 | 784 | 2,452 | 1,298 |
| QSVM | 0.35 | **0.8199** | 0.6996 | **0.6197** | **0.6572** | **0.8917** | 0.1028 | **0.5375** | 8,713 | 998 | 1,426 | 2,324 |
| VQC | 0.50 | 0.7040 | 0.4741 | 0.5731 | 0.5189 | 0.7801 | 0.2455 | 0.3107 | 7,327 | 2,384 | 1,601 | 2,149 |

The ordering on this subset differs from the primary common test. The QSVM is strongest by F1
(0.6572), accuracy (0.8199), recall (0.6197), ROC-AUC (0.8917) and MCC (0.5375). The CNN is the most
conservative: it has the highest precision (0.6234) and the lowest FPR (0.0807), but its recall is
only 0.3461. The VQC is intermediate on F1 (0.5189) and ROC-AUC (0.7801) but has the highest FPR
(0.2455).

The QSVM's F1 advantage over the CNN is 0.2121 (21.2 points), and its ROC-AUC advantage is 0.2252.
This is a clear positive result for a quantum model in the study, but it is scope-specific. The
external subset is normal-heavy (72.14% Normal), comes from a single file, and inherits the main
validation-selected thresholds; no external-set confidence intervals or multi-seed repeats were
computed. The result therefore does not overturn the primary common-test ranking and should not be
presented as a general quantum advantage or deployment evidence. It does suggest that the QSVM's
kernel ranking transfers better to novel attack types under this covariate shift, which is a
worthwhile follow-up hypothesis.

### 7.9 Angle versus amplitude encoding

Both encodings are retrained under the same protocol; VQC runs receive an equal COBYLA budget of
150 iterations.

| Model | Encoding | Qubits | Depth | Size | Parameters | F1 | ROC-AUC | FPR | Train (s) | Inference (ms/sample) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QSVM | angle | 8 | 31 | 74 | 8 | 0.9231 | **0.9903** | 0.1185 | 1.97 | 3.8014 |
| QSVM | amplitude | 3 | 1 | 1 | 0 | **0.9364** | 0.9783 | **0.0156** | 0.31 | 0.8423 |
| VQC | angle | 8 | 38 | 112 | 24 | 0.8028 | 0.8982 | 0.3025 | 9.58 | 3.4843 |
| VQC | amplitude | 3 | 8 | 14 | 9 | **0.8958** | **0.9609** | **0.0374** | 0.77 | 0.4877 |

Amplitude encoding uses three qubits instead of eight, trains faster and improves F1 for both
models. It costs the QSVM 1.2 ROC-AUC points and gains the VQC 6.3. However, basis-state amplitude
encoding with real non-negative amplitudes produces a squared-cosine fidelity kernel, a classical
polynomial kernel. The comparison is a resource/performance check, not evidence of quantum
advantage.

The angle-encoded rows reproduce the main experiment's metrics exactly; their training and inference
times differ because they come from a separate encoding run. Use the main comparison table for the
primary timing values and this table for the angle-versus-amplitude contrast.

### 7.10 Latency

Single-sample end-to-end CPU latency, median of five repeats, 1,800 samples, batch size 256:

| Model | Preprocess (ms) | Reduction (ms) | Single-sample inference (ms) | Batch inference (ms/sample) | Throughput (samples/s) | End-to-end (ms) | Slowdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN+LSTM | 0.00986 | 0.00084 | **0.9216** | **0.00544** | **183,947.7** | **0.9323** | 1.00× |
| QSVM | 0.00986 | 0.00084 | 3.9428 | 2.5958 | 385.2 | 3.9535 | 4.28× |
| VQC | 0.00986 | 0.00084 | 32.7295 | 3.0120 | 332.0 | 32.7402 | 35.51× |

Quantum stage breakdown:

| Model | State preparation (ms/sample) | Kernel evaluation (ms/sample) | Quantum inference (ms/sample) |
|---|---:|---:|---:|
| QSVM | 0.3833 | 0.0211 | — |
| VQC | — | — | 3.3711 |

**Measurement note.** The main comparison table's `inference_ms_per_sample` values are measured on
the full 1,800-row batch evaluation: CNN 0.00726 ms, QSVM 2.7440 ms, VQC 2.8822 ms. The latency
table above comes from a separate benchmark with its own single-sample and batch measurements. The
two contexts are not interchangeable. Absolute timings are CPU-only, load-dependent and indicative,
not production capacity numbers.

### 7.11 Quantum circuit resources

| Model | Encoding | Qubits | Depth | Size | Parameters | Two-qubit gates | Optimizer | Shots recorded | Backend |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| QSVM | angle | 8 | 31 | 74 | 8 | 28 | — | — | `qiskit.quantum_info.Statevector` (exact fidelity kernel) |
| VQC | angle | 8 | 38 | 112 | 24 | 42 | COBYLA | 1,024 | `qiskit.quantum_info.Statevector` (analytic probabilities) |

The VQC's 1,024 shots are recorded metadata; the reported metrics use analytic statevector
probabilities and are shot-noise-free.

### 7.12 Explainability

**Permutation importance (CNN+LSTM, ROC-AUC drop, 500 samples, 10 repeats, baseline AUC 0.9905)**

| Feature | Mean importance | Std |
|---|---:|---:|
| PC1 | 0.3503 | 0.0280 |
| PC6 | 0.0400 | 0.0035 |
| PC2 | 0.0241 | 0.0029 |
| PC4 | 0.0136 | 0.0022 |
| PC5 | 0.0115 | 0.0019 |
| PC3 | 0.0068 | 0.0019 |
| PC7 | 0.0026 | 0.0019 |
| PC8 | 0.0003 | 0.0005 |

**SHAP (CNN+LSTM, mean absolute SHAP, 100 explained rows, 50 background rows)**

| Feature | Mean absolute SHAP |
|---|---:|
| PC1 | 0.3556 |
| PC2 | 0.0677 |
| PC4 | 0.0640 |
| PC5 | 0.0344 |
| PC6 | 0.0340 |
| PC7 | 0.0195 |
| PC3 | 0.0165 |
| PC8 | 0.0042 |

**PCA attribution (top original features)**

| Original feature | Attributed importance | Normalised |
|---|---:|---:|
| `dst_host_srv_serror_rate` | 0.1222 | 0.0636 |
| `dst_host_serror_rate` | 0.1221 | 0.0636 |
| `serror_rate` | 0.1220 | 0.0635 |
| `srv_serror_rate` | 0.1217 | 0.0634 |
| `same_srv_rate` | 0.1215 | 0.0632 |

**Quantum input sensitivity (mean absolute probability change, δ = 0.25, 100 samples)**

| Feature | QSVM | VQC |
|---|---:|---:|
| PC1 | 0.0651 | 0.1386 |
| PC2 | 0.1225 | 0.1610 |
| PC3 | **0.1778** | **0.2033** |
| PC4 | 0.0653 | 0.0250 |
| PC5 | 0.1690 | 0.0022 |
| PC6 | 0.1173 | ~1.14e-16 |
| PC7 | 0.0781 | ~9.99e-17 |
| PC8 | 0.0615 | ~8.27e-17 |

The CNN concentrates on PC1 plus a few secondary components. PCA attribution maps these to
connection-error-rate features, plausible for DoS/Probe detection. Quantum sensitivity is a local
input-perturbation analysis, not a Shapley attribution. The VQC is numerically insensitive to
PC6–PC8 (~1e-16), which is a circuit/measurement property, not evidence that those components are
irrelevant.

### 7.13 Controlled distribution shift

CNN+LSTM only, on the common test set. Baseline: F1 0.9533, ROC-AUC 0.9891, KS mean 0.0197.

| Shift | Severity | F1 | ROC-AUC | FPR | KS mean | Permuted features |
|---|---:|---:|---:|---:|---:|---:|
| None | 0.0 | 0.9533 | 0.9891 | 0.0374 | 0.0197 | 0 |
| Gaussian noise | 0.1 | 0.9576 | 0.9888 | 0.0353 | 0.0528 | — |
| Gaussian noise | 0.25 | 0.9459 | 0.9869 | 0.0509 | 0.0936 | — |
| Gaussian noise | 0.5 | 0.8984 | 0.9629 | 0.1091 | 0.1371 | — |
| Gaussian noise | 1.0 | 0.7882 | 0.8755 | 0.2588 | 0.2160 | — |
| Feature permutation | 0.1 | 0.9167 | 0.9778 | 0.1019 | 0.0197 | 1 |
| Feature permutation | 0.25 | 0.4546 | 0.5613 | 0.4116 | 0.0197 | 2 |
| Feature permutation | 0.5 | 0.4685 | 0.5803 | 0.4064 | 0.0197 | 4 |

The model tolerates moderate Gaussian noise but collapses when 25% or more of features are
permuted. Feature permutation does not change marginal distributions, so the KS statistic remains
at its baseline value while performance collapses; this is a useful distinction between
distribution shift and feature-relationship corruption.

**Simulated file-order windows on `KDDTest+`**

| Window | Rows | Attack rate | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,508 | 0.5708 | 0.7502 | 0.9099 | 0.6242 | 0.7404 | 0.8416 | 0.0822 |
| 2 | 4,509 | 0.5828 | 0.7376 | 0.9089 | 0.6111 | 0.7308 | 0.8399 | 0.0856 |
| 3 | 4,509 | 0.5567 | 0.7554 | 0.9170 | 0.6163 | 0.7372 | 0.8351 | 0.0700 |
| 4 | 4,509 | 0.5706 | 0.7549 | 0.9091 | 0.6339 | 0.7470 | 0.8404 | 0.0842 |
| 5 | 4,509 | 0.5653 | 0.7507 | 0.9078 | 0.6222 | 0.7384 | 0.8457 | 0.0821 |

`KDDTest+` file order is not chronological. This is a simulated stress test, not concept-drift
detection. Only the classical model is evaluated under drift; no comparison under drift is claimed.

### 7.14 Tiered alert triage

CNN+LSTM probabilities mapped to a five-level policy on the 1,800-row common test set:

| Level | Count | Fraction | Action |
|---|---:|---:|---|
| NORMAL | 920 | 51.11% | ALLOW |
| MONITOR | 67 | 3.72% | LOG_AND_MONITOR |
| ALERT | 33 | 1.83% | NOTIFY_ANALYST |
| QUARANTINE | 36 | 2.00% | QUARANTINE_HOST |
| ESCALATE | 744 | 41.33% | BLOCK_AND_ESCALATE |

Policy bands: `p_attack < 0.20` NORMAL; `0.20–0.45` MONITOR; `0.45–0.70` ALERT; `0.70–0.90`
QUARANTINE; `>= 0.90` ESCALATE. U2R and R2L predictions receive a one-level severity bump.

The high ESCALATE fraction reflects the benchmark's 46.56% attack rate and fixed conservative
bands. The engine is an assistive demonstration, not a calibrated production policy.

---

## 8. Discussion points

### 8.1 Why the classical model wins here

The CNN+LSTM sees 8,399 training rows versus 600 for each quantum model and can fit a flexible
decision boundary in the reduced space. The QSVM, despite seeing only 600 rows, is within 3.02 F1
points and has a marginally higher test ROC-AUC. That is a credible result for a kernel method on a
small sample, but it is not an advantage. The VQC is the weakest model: its optimiser did not
converge, its ansatz is shallow, and its 30.25% FPR would be difficult to justify in most
operational settings.

The result should not be read as "quantum methods cannot work for IDS." It says that, on the
primary common test under an 8-qubit, 600-training-row, exact-simulation budget with a PCA-8
representation, the quantum models do not beat a reasonably configured classical baseline. Larger
circuits, larger training sets, better ansätze and hardware-aware training could change the picture;
none of those was tested. The external novel-type subset is a separate, scope-specific exception
discussed in Section 8.4.

### 8.2 The ROC-AUC nuance

The QSVM's test ROC-AUC is 0.9903 versus 0.9891 for the CNN, a difference of 0.0012 with a paired
bootstrap CI that crosses zero. Reporting this as a quantum win would be unjustified. It is better
read as a tie on ranking quality, with the CNN better on average precision (0.9907 vs 0.9898),
Brier score (0.0308 vs 0.0419) and false-positive rate (0.0374 vs 0.1185). Cross-validation also
reverses the AUC ordering (CNN 0.9909 vs QSVM 0.9869), so the single-test AUC advantage is not
stable.

### 8.3 The unseen-category result

The VQC's 76.19% recall on held-out R2L/U2R is a striking positive result for a quantum model on a
small, different holdout. It is also fragile: 168 rows, one seed, a 16.7% FPR, and a threshold tuned
on a small held-out-category validation split. The CNN and QSVM show near-zero recall on this
holdout, consistent with decision boundaries that do not transfer to the held-out categories. This
is a real generalisation phenomenon worth follow-up, not a deployment recommendation.

### 8.4 A scope-specific external novel-type result

On external novel types, the QSVM leads by F1 (0.6572 vs 0.5189 for the VQC and 0.4451 for the
CNN) and ROC-AUC (0.8917 vs 0.7801 and 0.6665). The CNN remains the most conservative detector, with
the lowest FPR (0.0807) and highest precision (0.6234). The QSVM's external lead is threshold-free
for AUC but threshold-dependent for F1, and the external subset is normal-heavy (72.14% Normal) and
drawn from one file. It is a genuine, scope-specific result, not a general quantum advantage. The
primary common-test result still favours the CNN on F1, and the QSVM's main AUC advantage is
unresolved. The external ordering is therefore best reported as a hypothesis about kernel ranking
under novel-type shift, not as a deployment recommendation.

### 8.5 Resource and simulation caveats

The quantum models are slower per sample in simulation, and hardware would add sampling and error
costs. Batching does amortise state preparation: QSVM batch inference is 2.60 ms/sample versus
3.94 ms single-sample, and VQC is 3.01 ms/sample versus 32.73 ms single-sample. Both remain far
slower than the CNN, whose batch throughput is 183,947.7 samples/s. Single-sample latency is 4.28×
the CNN for the QSVM and 35.51× for the VQC. These numbers are indicative, not a capacity plan.

Amplitude encoding is cheaper and produces higher F1 here, but it reduces to a classical polynomial
kernel for real non-negative amplitudes. It cannot be used as evidence of quantum advantage.

### 8.6 What would change the conclusion

- Larger quantum training sets and circuits beyond 8 qubits.
- A converged VQC with a deeper ansatz and a larger optimisation budget.
- Hardware-aware noise models and finite-shot evaluation.
- Multi-seed runs and a larger common test set.
- Modern datasets with timestamps and realistic traffic mixtures.
- A broader classical hyperparameter search, including non-PCA representations.

None of these was tested in the reported run.

---

## 9. Limitations and threats to validity

### 9.1 Numbered limitations

1. **Quick profile only.** The full 125,973-row profile was not executed. Absolute metrics are not
   the best achievable on NSL-KDD.
2. **Training-budget asymmetry.** Quantum models see 600 rows versus 8,399 for the CNN. The
   comparison is fair in evaluation, not in data volume.
3. **Eight-qubit cap.** PCA-8 costs 0.0134 validation F1 (1.34 points) relative to PCA-16 and limits
   the quantum feature space.
4. **Simulation only.** No hardware noise, decoherence, readout error, transpilation or finite-shot
   sampling in the reported metrics. Results are a shot-noise-free upper bound relative to noisy
   hardware.
5. **VQC did not converge.** COBYLA used all 151 evaluations; final loss 0.4585. The VQC result is
   budget-limited.
6. **Single seed.** Cross-validation provides stability evidence, but no multi-seed study was
   performed.
7. **Small holdouts and external uncertainty.** The unseen-category test has 168 rows and 84
   attacks. The external novel test has 3,750 attacks and 9,711 Normal rows, but comes from one file
   with strong covariate shift and no external-set confidence intervals or multi-seed repeats.
8. **NSL-KDD is old.** No timestamps, synthetic attack mix, 1998-era traffic. Results do not transfer
   automatically to modern networks.
9. **"Unseen category" is not "zero-day."** The held-out categories are known and present in the
   test set.
10. **Amplitude encoding is classically simulable.** Its empirical gains are not quantum advantage.
11. **Latency is load-dependent.** CPU-only medians of five repeats; absolute timings vary.
12. **Explainability is post hoc.** Permutation, SHAP and PCA attribution are correlational; quantum
    sensitivity is not a Shapley value.
13. **Drift is simulated.** File order is not chronological; Gaussian noise and feature permutation
    are stress tests.
14. **No production validation.** No adversarial, streaming, encrypted-traffic or live deployment
    evaluation was performed.
15. **No exhaustive hyperparameter search.** The reported run uses fixed hyperparameters from
    `config/config.yaml`; the classical baseline is reasonably configured but not exhaustively
    tuned.
16. **CV threshold.** Cross-validation classification metrics use the fixed 0.5 threshold, so CV F1
    is not directly comparable to the validation-thresholded common-test F1; ROC-AUC is unaffected.
17. **External CIs not computed.** The external novel-type metrics are point estimates. The QSVM's
    external F1/AUC lead was not accompanied by a paired bootstrap or McNemar test on the external
    subset.

### 9.2 Threats to validity

| Type | Threat | Mitigation / disclosure |
|---|---|---|
| Internal | Preprocessing leakage | Eight-check audit; all fitted transforms train-only; split overlap zero |
| Internal | Test-set tuning | Thresholds and feature configuration selected on validation only; test scored once |
| Internal | Single seed | 3-fold CV reported; no multi-seed claim |
| Construct | PCA-8 proxy for traffic | Same representation for all models; disclosed 8-qubit constraint |
| Construct | Pseudo-temporal CNN | Explicitly stated as PCA-index order, not packet timing |
| External | NSL-KDD age and synthetic mix | Framed as a controlled benchmark, not modern-network evidence |
| External | Quick subsample | Full profile not executed; no full-profile numbers claimed |
| Statistical | Multiple comparisons | Small pre-specified set; no family-wise correction; p-values descriptive |
| Statistical | CV threshold fixed at 0.5 | Disclosed; ROC-AUC is threshold-free |
| Statistical | Underpowered Wilcoxon | 3 folds; reported as supporting evidence only |
| Statistical | Small unseen holdout | 168 rows; VQC result labelled suggestive, not conclusive |
| External | External novel subset normal-heavy | 72.14% Normal; F1/MCC reported alongside accuracy |
| External | One external file; no external CIs | QSVM external lead reported as scope-specific; no general-advantage claim |
| Conclusion | Primary negative-result scope | Limited to 8 qubits, 600 quantum rows, PCA-8, simulation; external exception reported separately |

---

## 10. Reproducibility

### 10.1 Executed command and runtime

```bash
python run_all.py --mode quick --force
```

The clean 14-stage run completed in **603.3 s** (10 m 3.3 s) on 21 October 2026. Stages:
`prepare`, `features`, `train`, `evaluate`, `cv`, `stats`, `zero_day`, `encoding`, `latency`,
`drift`, `explain`, `tables`, `dashboard`, `report`.

A reporting-only refresh uses:

```bash
python run_pipeline.py --mode dashboard
```

This regenerates `results/final_summary.json`, `results/final_summary.md`,
`docs/paper_draft.md` and `results/dashboard.html` without retraining.

The full profile command is available but **was not executed**:

```bash
python run_all.py --mode full --force
```

### 10.2 Environment

| Item | Value |
|---|---|
| Operating system | Windows 11 (AMD64) |
| CPU | Intel64 Family 6 Model 186 Stepping 3, GenuineIntel |
| Logical cores | 12 |
| RAM | 15.72 GB |
| GPU | None (CPU only) |
| Python | 3.14.5 |
| NumPy | 2.4.4 |
| SciPy | 1.17.1 |
| pandas | 3.0.1 |
| scikit-learn | 1.8.0 |
| PyTorch | 2.14.1 |
| Qiskit | 2.6.0 |
| qiskit-aer | 0.18.0 |
| qiskit-machine-learning | 0.9.1 |
| SHAP | 0.53.0 |
| Streamlit | 1.65.0 |
| Seed | 42 |
| `PYTHONHASHSEED` | 42 |
| `OMP_NUM_THREADS` | 1 |

### 10.3 Artifacts

- Preprocessing objects: `data/processed/preprocessor.joblib`, `reducer.joblib`, `scaler.joblib`.
- Models: `models/cnn_lstm.pt`, `models/qsvm.joblib`, `models/vqc.joblib`, plus `*_meta.json`.
- Splits: `data/splits/train.parquet`, `validation.parquet`, `test.parquet`.
- Metrics: `results/metrics/*.json`.
- Predictions: `results/predictions/*.csv`.
- Figures: `results/figures/`.
- Tables: `results/tables/`.
- Environment: `results/environment.json`.
- Manifest: `results/run_manifest.json`.
- Logs: `results/logs/`.

### 10.4 Test suite

```bash
python -m pytest -q
```

Result on 21 October 2026: **31 passed** (8.43 s). Tests cover data validation, leakage checks,
feature reduction, quantum encodings and kernel properties, VQC equivalence to direct simulation,
metrics, statistical tests, alert tiers and pipeline output contracts.

### 10.5 Artifact caveat

The latest `run_manifest.json` `run_complete` entry was overwritten by a report-only refresh at
15:47:26 (0.14 s; stages `report`, `dashboard`), after the targeted external-novel correction at
15:42:15 (172.9 s; stages `zero_day`, `tables`, `report`). The clean full quick run is recorded in
`results/logs/run_quick_20261021_132234.log` and in
`results/logs/clean_quick_stdout.log` (603.3 s). When reporting runtime, cite the clean run log, not
the targeted refresh. The external-correction run log is
`results/logs/run_quick_20261021_153917.log`.

### 10.6 Data and code availability

The repository contains the code, configuration, split files, trained model payloads, predictions,
metrics, figures and tables needed to reproduce the reported quick-profile experiment. NSL-KDD is a
public benchmark; check the original dataset's redistribution terms before publishing raw rows. No
new traffic was captured, no human subjects were involved and no live network was monitored.

---

## 11. Claim guardrails

### 11.1 Safe claims

| Claim | Evidence |
|---|---|
| The CNN+LSTM achieved higher F1 than the QSVM on the common test set. | 0.9533 vs 0.9231; paired-bootstrap difference 0.0302 [0.0166, 0.0434] |
| The QSVM's test ROC-AUC was 0.0012 higher, but the difference is unresolved. | 0.9903 vs 0.9891; 95% CI [−0.0061, 0.0030] crosses zero |
| The CNN+LSTM had a much lower false-positive rate. | 0.0374 vs 0.1185 vs 0.3025 |
| The QSVM had the highest recall and lowest FNR. | Recall 0.9737, FNR 0.0263 |
| The VQC was the weakest model on the main task. | F1 0.8028, FPR 0.3025 |
| The VQC detected more held-out R2L/U2R attacks in the unseen-category holdout. | Recall 0.7619 vs 0.0476 (CNN) and 0.0119 (QSVM), n = 168 |
| The QSVM had the best external novel-type F1 and ROC-AUC. | F1 0.6572 vs 0.5189 (VQC) and 0.4451 (CNN); AUC 0.8917 vs 0.7801 and 0.6665 |
| The CNN+LSTM had the lowest external novel-type FPR and highest precision. | FPR 0.0807 vs 0.1028 (QSVM) and 0.2455 (VQC); precision 0.6234 |
| All quantum results are exact statevector simulations. | `results/environment.json`; no hardware backend |
| The pipeline passed its leakage audit. | 8/8 checks passed in `data/metadata/leakage_audit.json` |
| The VQC did not converge within its budget. | 151 COBYLA evaluations; loss 0.4585; `converged = false` |

### 11.2 Unsafe claims

| Do not claim | Why |
|---|---|
| "Quantum machine learning is worse than classical for IDS." | The result is limited to one benchmark, 8 qubits, 600 quantum training rows and one seed. |
| "The QSVM has better ranking quality on the primary common test." | The AUC difference is 0.0012 with a CI crossing zero; CV reverses the ordering. |
| "The QSVM is generally better under distribution shift." | The external lead is one normal-heavy file with no external-set CIs; the main common test still favours the CNN on F1. |
| "The QSVM's external advantage is deployment-ready." | One external file, inherited thresholds, 10.28% FPR, simulation only; no live-traffic validation. |
| "The VQC detects zero-day attacks." | The holdout uses known R2L/U2R categories; this is unseen-category generalisation. |
| "The system is ready for production." | No streaming, adversarial, encrypted-traffic or live deployment evaluation. |
| "The results reflect quantum hardware." | Exact statevector simulation only; no noise, decoherence or readout error. |
| "The CNN+LSTM models real packet timing." | The input is a PCA-index sequence, not a temporal packet sequence. |
| "The model is state-of-the-art on NSL-KDD." | Quick profile only; no exhaustive tuning; no comparison to published SOTA. |
| "The classical baseline was exhaustively tuned." | Hyperparameters are fixed in config; no broad search. |
| "Amplitude encoding proves quantum advantage." | Real non-negative amplitude encoding reduces to a classical polynomial kernel. |
| "The drift experiment detects real concept drift." | No timestamps; file-order windows and synthetic shifts only. |
| "McNemar proves a practically important difference." | The CNN–QSVM effect size is negligible (Cohen's h = 0.138). |
| "The VQC is generally better on unseen attacks." | The holdout has 168 rows, one seed and a 16.7% FPR. |

### 11.3 Reviewer questions and suggested answers

1. **Why is the QSVM trained on only 600 rows?**
   Exact statevector kernel evaluation grows quadratically with training size. A 600×600 kernel and
   a 150-iteration VQC are large enough to motivate a bounded budget. The test set is identical for
   all models, and the asymmetry is disclosed. It favours the classical model in data volume.

2. **Why PCA-8 instead of the best PCA-16?**
   The reported experiment is capped at 8 qubits (one qubit per reduced feature). PCA-8 is the best
   feasible configuration on validation F1. Using PCA-16 for the classical model only would break
   the identical-input comparison.

3. **Is the comparison fair?**
   It is fair in evaluation: identical splits, identical reduced matrix, identical test rows,
   identical metrics and validation-only thresholds. It is not fair in training data volume; that
   is a stated limitation.

4. **Why does the CNN+LSTM beat the quantum models on the primary common test?**
   It sees 14× more training data and can fit a flexible boundary in the reduced space. The QSVM is
   still within 3.02 F1 points, which is credible for a 600-row kernel method. The VQC is
   budget-limited and did not converge.

5. **Why report a negative result?**
   The primary protocol is designed to make the comparison auditable. Reporting only positive
   quantum results would distort the evidence. The methodological contribution is the
   leakage-audited comparison itself, including the scope-specific external exception.

6. **Are the statistical tests appropriate?**
   McNemar uses paired predictions on the same test rows. The paired bootstrap resamples rows and
   applies the same indices to both models. Wilcoxon is underpowered with 3 folds and is reported as
   supporting evidence only. No family-wise correction is applied; p-values are descriptive.

7. **Why is the QSVM's AUC higher but its F1 lower on the primary common test?**
   AUC is threshold-free; F1 is evaluated at the validation-selected threshold. The QSVM's
   probability ranking is slightly better on this test set, but its operating point trades
   precision for recall: 114 false positives versus 36 for the CNN. Average precision is actually
   lower for the QSVM.

8. **Why does the VQC win the unseen-category holdout?**
   The VQC's decision boundary happens to generalise better to the held-out R2L/U2R rows in this
   small, balanced holdout. The result is fragile: 168 rows, one seed, a 16.7% FPR and a
   validation-selected threshold on a small split. It is a follow-up hypothesis, not a conclusion.

9. **Why does the QSVM lead the external novel-type subset but lose the common test?**
   The external subset has a different class mix (72.14% Normal) and contains 17 attack types absent
   from training, so it stresses ranking under novel-type covariate shift rather than the primary
   benchmark. The QSVM's AUC is threshold-free and higher on that subset, while its F1 depends on the
   main validation-selected threshold. The CNN remains better on the primary common test. With one
   external file and no external-set CIs, this is a scope-specific result, not a general advantage.

---

## 12. Figure inventory with suggested captions

| File | Suggested caption |
|---|---|
| `docs/architecture.png` | End-to-end QE-IDS pipeline: leakage-safe preprocessing, shared PCA-8 representation, three classifiers, common-test evaluation and paired statistics. |
| `results/figures/class_distribution.png` | Class and attack-category distribution in the 12,000-row quick pool. |
| `results/figures/pca_explained_variance.png` | PCA explained variance; the first eight components retain 66.66% of total variance. |
| `results/figures/reduced_feature_space.png` | Two-dimensional projection of the shared reduced feature space. |
| `results/figures/pca_component_loadings.png` | Loadings of the original features on the retained principal components. |
| `results/figures/cnn_lstm_training_curves.png` | CNN+LSTM training and validation loss/F1/ROC-AUC over 25 epochs. |
| `results/figures/roc_comparison.png` | ROC curves for all three models on the 1,800-row common test set. |
| `results/figures/pr_comparison.png` | Precision–recall curves on the common test set. |
| `results/figures/confusion_matrices.png` | Confusion matrices at each model's validation-selected threshold. |
| `results/figures/cross_validation_f1.png` | Three-fold cross-validation F1 with mean and 95% confidence interval. |
| `results/figures/unseen_attack_performance.png` | Unseen-category holdout performance after removing R2L and U2R from training. |
| `results/figures/encoding_comparison.png` | Angle versus amplitude encoding for QSVM and VQC under an equal optimiser budget. |
| `results/figures/latency_comparison.png` | Single-sample and batch inference latency on CPU. |
| `results/figures/quantum_resources.png` | Circuit depth, size, parameter count and two-qubit gate count. |
| `results/figures/vqc_convergence.png` | VQC COBYLA convergence history; the optimiser used all 151 evaluations. |
| `results/figures/permutation_importance.png` | Permutation importance of the eight PCA components for the CNN+LSTM. |
| `results/explainability/shap_summary.png` | SHAP summary for the CNN+LSTM on 100 explained rows. |
| `results/figures/pca_feature_attribution.png` | Attribution of CNN importance back to original NSL-KDD features through PCA loadings. |
| `results/figures/qsvm_feature_sensitivity.png` | QSVM local input-sensitivity analysis at δ = 0.25. |
| `results/figures/vqc_feature_sensitivity.png` | VQC local input-sensitivity analysis at δ = 0.25. |
| `results/figures/drift_gaussian_noise.png` | CNN+LSTM performance under controlled Gaussian noise. |
| `results/figures/drift_feature_permutation.png` | CNN+LSTM performance under feature permutation. |
| `results/figures/drift_simulated_windows.png` | Simulated file-order windows on `KDDTest+`; not chronological drift. |

---

## 13. Table and data provenance

| Reported quantity | Source artifact |
|---|---|
| Dataset validation, label counts | `data/metadata/dataset_validation.json` |
| Split sizes, overlap, integrity | `data/splits/split_metadata.json` |
| Leakage audit | `data/metadata/leakage_audit.json` |
| Feature ablation, selected PCA-8 | `data/metadata/feature_selection.json`, `results/tables/feature_reduction_ablation.csv` |
| PCA explained variance | `results/metrics/pca_explained_variance.json` |
| Main comparison | `results/final_summary.json` → `master_table`, `results/tables/master_comparison.csv` |
| Per-model metrics and confusion matrices | `results/metrics/cnn_lstm.json`, `qsvm.json`, `vqc.json` |
| Cross-validation | `results/tables/cross_validation_summary.csv`, `results/metrics/cross_validation_details.json` |
| McNemar, bootstrap, Wilcoxon | `results/metrics/statistical_tests.json` |
| Unseen category | `results/metrics/unseen_category_details.json`, `results/tables/unseen_category_results.csv` |
| External novel types | `results/metrics/external_novel_details.json`, `results/tables/external_novel_attack_results.csv` |
| Encoding comparison | `results/tables/encoding_comparison.csv` |
| Latency | `results/metrics/latency.json`, `results/tables/latency_comparison.csv` |
| Quantum resources | `results/metrics/quantum_resources.json`, `results/tables/quantum_resources.csv` |
| Explainability | `results/explainability/explainability_summary.json` |
| Drift | `results/metrics/drift.json`, `results/tables/drift_controlled.csv`, `results/tables/drift_windows_external.csv` |
| Alert triage | `results/metrics/alert_summary.json` |
| Environment | `results/environment.json` |
| Runtime and stage log | `results/logs/run_quick_20261021_132234.log` |
| QSVM kernel validation | `models/qsvm_meta.json` |

---

## 14. References

1. Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). A detailed analysis of the KDD CUP
   99 data set. *IEEE Symposium on Computational Intelligence for Security and Defense
   Applications (CISDA)*, 1–6.
2. Havlíček, V., Córcoles, A. D., Temme, K., Harrow, A. W., Kandala, A., Chow, J. M., & Gambetta,
   J. M. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature*, 567, 209–212.
3. Schuld, M., & Killoran, N. (2019). Quantum machine learning in feature Hilbert spaces.
   *Physical Review Letters*, 122(4), 040504.
4. Cerezo, M., Arrasmith, A., Babbush, R., Benjamin, S. C., Endo, S., Fujii, K., ... & Coles, P. J.
   (2021). Variational quantum algorithms. *Nature Reviews Physics*, 3, 625–644.
5. Qiskit contributors. Qiskit: An open-source framework for quantum computing (2024–2026).
6. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine
   Learning Research*, 12, 2825–2830.
7. Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library.
   *Advances in Neural Information Processing Systems*, 32.
8. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions.
   *Advances in Neural Information Processing Systems*, 30.
9. Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification
   learning algorithms. *Neural Computation*, 10(7), 1895–1923.
10. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
11. Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*.
12. Powell, M. J. D. (1994). A direct search optimization method that models the objective and
    constraint functions by linear interpolation. *Advances in Optimization and Numerical
    Analysis*, 51–67.
13. Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
14. Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to
    regularized likelihood methods. In *Advances in Large Margin Classifiers* (pp. 61–74). MIT
    Press.
15. McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions
    or percentages. *Psychometrika*, 12(2), 153–157.
16. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.

---

## Appendix A. Exact values by result file

### A.1 Main comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN+LSTM | 0.9566667 | 0.9567308 | 0.9498807 | 0.9532934 | 0.9890865 | 0.0374220 | 0.0501193 |
| QSVM | 0.9244444 | 0.8774194 | 0.9737470 | 0.9230769 | 0.9903009 | 0.1185031 | 0.0262530 |
| VQC | 0.7933333 | 0.7223282 | 0.9033413 | 0.8027572 | 0.8981773 | 0.3024948 | 0.0966587 |

| Model | Specificity | NPV | PPV | Balanced accuracy | MCC | Brier | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN+LSTM | 0.9625780 | 0.9566116 | 0.9567308 | 0.9562293 | 0.9129004 | 0.0307790 | 0.1124238 |
| QSVM | 0.8814969 | 0.9747126 | 0.8774194 | 0.9276219 | 0.8536865 | 0.0418707 | 0.1398414 |
| VQC | 0.6975052 | 0.8922872 | 0.7223282 | 0.8004232 | 0.6076920 | 0.1527255 | 0.4783961 |

### A.2 Bootstrap confidence intervals

| Model | Metric | Point | 95% CI | Bootstrap std |
|---|---|---:|---|---:|
| CNN+LSTM | Accuracy | 0.9566667 | [0.9466667, 0.9666667] | 0.0050189 |
| CNN+LSTM | F1 | 0.9532934 | [0.9420288, 0.9640644] | 0.0055742 |
| CNN+LSTM | ROC-AUC | 0.9890865 | [0.9842129, 0.9930325] | 0.0022235 |
| QSVM | Accuracy | 0.9244444 | [0.9127778, 0.9372222] | 0.0062852 |
| QSVM | F1 | 0.9230769 | [0.9104564, 0.9365483] | 0.0065920 |
| QSVM | ROC-AUC | 0.9903009 | [0.9873742, 0.9930423] | 0.0014838 |
| VQC | Accuracy | 0.7933333 | [0.7744445, 0.8116667] | 0.0094960 |
| VQC | F1 | 0.8027572 | [0.7825601, 0.8213557] | 0.0097418 |
| VQC | ROC-AUC | 0.8981773 | [0.8826362, 0.9112613] | 0.0072108 |

### A.3 Unseen-category exact values

| Metric | CNN+LSTM | QSVM | VQC |
|---|---:|---:|---:|
| Accuracy | 0.5119048 | 0.4940476 | 0.7976190 |
| Precision | 0.6666667 | 0.3333333 | 0.8205128 |
| Recall | 0.0476190 | 0.0119048 | 0.7619048 |
| F1 | 0.0888889 | 0.0229885 | 0.7901235 |
| Specificity | 0.9761905 | 0.9761905 | 0.8333333 |
| FPR | 0.0238095 | 0.0238095 | 0.1666667 |
| FNR | 0.9523810 | 0.9880952 | 0.2380952 |
| NPV | 0.5061728 | 0.4969697 | 0.7777778 |
| PPV | 0.6666667 | 0.3333333 | 0.8205128 |
| Balanced accuracy | 0.5119048 | 0.4940476 | 0.7976190 |
| MCC | 0.0641500 | −0.0449467 | 0.5967624 |
| ROC-AUC | 0.7484410 | 0.6558957 | 0.8707483 |
| Average precision | 0.6421355 | 0.5961552 | 0.8170352 |
| Brier | 0.4468286 | 0.4225673 | 0.1665405 |
| Log loss | 1.7604064 | 1.6502244 | 0.5126787 |

### A.4 External novel-type exact values

| Metric | CNN+LSTM | QSVM | VQC |
|---|---:|---:|---:|
| Threshold | 0.35 | 0.35 | 0.50 |
| Accuracy | 0.7596018 | 0.8199242 | 0.7039596 |
| Precision | 0.6234390 | 0.6995786 | 0.4740790 |
| Recall | 0.3461333 | 0.6197333 | 0.5730667 |
| F1 | 0.4451303 | 0.6572398 | 0.5188941 |
| Specificity | 0.9192668 | 0.8972299 | 0.7545052 |
| FPR | 0.0807332 | 0.1027701 | 0.2454948 |
| FNR | 0.6538667 | 0.3802667 | 0.4269333 |
| NPV | 0.7845153 | 0.8593550 | 0.8206765 |
| PPV | 0.6234390 | 0.6995786 | 0.4740790 |
| Balanced accuracy | 0.6327001 | 0.7584816 | 0.6637859 |
| MCC | 0.3290458 | 0.5375389 | 0.3107308 |
| ROC-AUC | 0.6664596 | 0.8916764 | 0.7800652 |
| Average precision | 0.5801306 | 0.6389853 | 0.5720533 |
| Brier | 0.2085969 | 0.1538133 | 0.1761800 |
| Log loss | 0.8900874 | 0.5424333 | 0.5266662 |
| TN | 8927 | 8713 | 7327 |
| FP | 784 | 998 | 2384 |
| FN | 2452 | 1426 | 1601 |
| TP | 1298 | 2324 | 2149 |

### A.5 Encoding exact values

| Model | Encoding | Qubits | Depth | Size | Params | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QSVM | angle | 8 | 31 | 74 | 8 | 0.9244 | 0.8774 | 0.9737 | 0.9231 | 0.9903 | 0.1185 | 0.0263 | 0.35 |
| QSVM | amplitude | 3 | 1 | 1 | 0 | 0.9433 | 0.9804 | 0.8962 | 0.9364 | 0.9783 | 0.0156 | 0.1038 | 0.70 |
| VQC | angle | 8 | 38 | 112 | 24 | 0.7933 | 0.7223 | 0.9033 | 0.8028 | 0.8982 | 0.3025 | 0.0967 | 0.50 |
| VQC | amplitude | 3 | 8 | 14 | 9 | 0.9083 | 0.9517 | 0.8461 | 0.8958 | 0.9609 | 0.0374 | 0.1539 | 0.55 |

---

## Appendix B. LaTeX-ready main table

```latex
\begin{table*}[t]
\centering
\caption{Common-test performance on NSL-KDD at each model's validation-selected threshold.
Common test set: 1,800 rows (962 Normal, 838 Attack). Best value per column in bold.}
\label{tab:main-results}
\begin{tabular}{lccccccccc}
\toprule
Model & Acc. & Prec. & Rec. & F1 & ROC-AUC & FPR & FNR & MCC & Brier \\
\midrule
CNN+LSTM & \textbf{0.9567} & \textbf{0.9567} & 0.9499 & \textbf{0.9533} & 0.9891 & \textbf{0.0374} & 0.0501 & \textbf{0.9129} & \textbf{0.0308} \\
QSVM     & 0.9244 & 0.8774 & \textbf{0.9737} & 0.9231 & \textbf{0.9903} & 0.1185 & \textbf{0.0263} & 0.8537 & 0.0419 \\
VQC      & 0.7933 & 0.7223 & 0.9033 & 0.8028 & 0.8982 & 0.3025 & 0.0967 & 0.6077 & 0.1527 \\
\bottomrule
\end{tabular}
\end{table*}
```

```latex
\begin{table}[t]
\centering
\caption{Paired statistical comparison on the common test set.}
\label{tab:paired-tests}
\begin{tabular}{llccc}
\toprule
Comparison & Metric & Difference & 95\% CI & Crosses zero \\
\midrule
CNN vs QSVM & F1 & 0.0302 & [0.0166, 0.0434] & No \\
CNN vs QSVM & ROC-AUC & $-0.0012$ & $[-0.0061, 0.0030]$ & Yes \\
CNN vs VQC & F1 & 0.1505 & [0.1308, 0.1712] & No \\
CNN vs VQC & ROC-AUC & 0.0909 & [0.0768, 0.1055] & No \\
QSVM vs VQC & F1 & 0.1203 & [0.1026, 0.1384] & No \\
QSVM vs VQC & ROC-AUC & 0.0921 & [0.0792, 0.1074] & No \\
\bottomrule
\end{tabular}
\end{table}
```

**McNemar values for LaTeX:** CNN vs QSVM: \(\chi^2=22.2534\), \(p=2.389\times10^{-6}\),
\(h=0.1375\); CNN vs VQC: \(\chi^2=214.6225\), \(p=1.347\times10^{-48}\), \(h=0.5245\);
QSVM vs VQC: \(\chi^2=152.5552\), \(p=4.792\times10^{-35}\), \(h=0.3869\).

---

## Appendix C. One-page number sheet

| Item | Value |
|---|---|
| Train / validation / test | 8,399 / 1,801 / 1,800 |
| Test normal / attack | 962 / 838 |
| Features | PCA-8, 66.6556% variance |
| CNN F1 / AUC / FPR | 0.9533 / 0.9891 / 0.0374 |
| QSVM F1 / AUC / FPR | 0.9231 / 0.9903 / 0.1185 |
| VQC F1 / AUC / FPR | 0.8028 / 0.8982 / 0.3025 |
| CNN − QSVM F1 | 0.0302 [0.0166, 0.0434] |
| CNN − QSVM AUC | −0.0012 [−0.0061, 0.0030] |
| McNemar CNN–QSVM | p = 2.389e-06, h = 0.1375 |
| CV F1 | 0.9346 / 0.9180 / 0.7452 |
| Unseen recall | 0.0476 / 0.0119 / 0.7619 |
| External novel F1 | 0.4451 / 0.6572 / 0.5189 |
| External novel AUC | 0.6665 / 0.8917 / 0.7801 |
| Single-sample latency | 0.9323 / 3.9535 / 32.7402 ms |
| CNN / QSVM / VQC qubits | 0 / 8 / 8 |
| VQC convergence | 151 evals, loss 0.4585, not converged |
| Leakage audit | 8/8 passed |
| Test suite | 31 passed |
| Clean run | 603.3 s |
| Scope | Quick profile, exact statevector simulation, no hardware |

---

**Final audit (22 October 2026).** `python -m pytest -q` → **31 passed**. All 54 Markdown tables
have consistent column counts. All 23 referenced figure paths exist. The corrected E5 values were
checked against `results/tables/external_novel_attack_results.csv` and
`results/final_summary.json`.

*End of sourcebook. All values are measured; only the author block is intentionally left as a placeholder.*
