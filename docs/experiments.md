# Experiments and Measured Outcomes

Every experiment below was executed by `run_pipeline.py`. The numbers are copied from the
generated artefacts after the clean quick-profile run (`python run_all.py --mode quick --force`).
Where an experiment has a known limitation, it is stated next to the result rather than deferred.

Common setup:

- Seed 42, quick profile.
- Common test set: 1,800 rows, attack rate 46.56%.
- Features: 8 PCA components, [0, π] scaling, identical for all models.
- Thresholds: selected on validation only (CNN 0.35, QSVM 0.35, VQC 0.50).

---

## E1 — Main three-way comparison

**Protocol.** Train CNN+LSTM (8,399 rows), QSVM (600 rows), VQC (600 rows) on the same PCA-8
matrix. Score each once on the common 1,800-row test set at its validation-selected threshold.
Report accuracy, precision, recall, F1, ROC-AUC, FPR, FNR, training time and circuit resources.

**Result.**

| Model | Encoding | Acc | Prec | Rec | F1 | ROC-AUC | FPR | FNR | Train (s) | Qubits | Depth |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN+LSTM | none | **0.9567** | **0.9567** | 0.9499 | **0.9533** | 0.9891 | **0.0374** | 0.0501 | 9.56 | 0 | 0 |
| QSVM | angle | 0.9244 | 0.8774 | **0.9737** | 0.9231 | **0.9903** | 0.1185 | **0.0263** | 1.72 | 8 | 31 |
| VQC | angle | 0.7933 | 0.7223 | 0.9033 | 0.8028 | 0.8982 | 0.3025 | 0.0967 | 8.61 | 8 | 38 |

**Interpretation.** The CNN+LSTM has the best F1, accuracy and false-positive rate. The QSVM has
the best recall and a marginally higher ROC-AUC (0.9903 vs 0.9891), but its FPR is 3.2× higher.
The VQC is clearly weaker on every ranking metric and operates at a 30% false-positive rate.

**Artefacts.** `results/tables/master_comparison.csv`, `results/metrics/{cnn_lstm,qsvm,vqc}.json`,
`results/figures/{model}_confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`.

---

## E2 — Cross-validation

**Protocol.** Build 3 stratified folds from the train+validation pool (3,000-row cap, 300 quantum
training rows per fold). Refit the entire preprocessing chain inside each fold on that fold's
training rows only. Report per-fold F1 and mean ± std with a 95% t-based CI.

**Result.**

| Model | Folds | Mean F1 | Std | 95% CI |
|---|---:|---:|---:|---|
| CNN+LSTM | 3 | **0.9346** | 0.0135 | [0.9010, 0.9682] |
| QSVM | 3 | 0.9180 | 0.0161 | [0.8779, 0.9580] |
| VQC | 3 | 0.7452 | 0.0527 | [0.6144, 0.8761] |

**Interpretation.** The ranking is stable across folds. The VQC has the largest variance,
consistent with a shallow ansatz and a budget-limited COBYLA run. Three folds are enough to see
the ordering but not enough for a powerful Wilcoxon test (see E3).

**Artefacts.** `results/tables/cross_validation_raw.csv`,
`results/tables/cross_validation_summary.csv`, `results/metrics/cross_validation_details.json`,
`results/figures/cross_validation_f1.png`.

---

## E3 — Statistical significance

**Protocol.** Pairwise McNemar on the common test predictions, paired bootstrap (1,000 resamples)
for F1 and ROC-AUC differences, and Wilcoxon signed-rank on per-fold F1.

**McNemar.**

| Comparison | Discordant | p-value | Significant | Cohen's h | Effect |
|---|---:|---:|---|---:|---|
| CNN+LSTM vs QSVM | 146 | 2.389 × 10⁻⁶ | yes | 0.138 | negligible |
| CNN+LSTM vs VQC | 400 | 1.347 × 10⁻⁴⁸ | yes | 0.524 | medium |
| QSVM vs VQC | 362 | 4.792 × 10⁻³⁵ | yes | 0.387 | small |

**Paired bootstrap.**

| Comparison | Metric | Difference (A−B) | 95% CI | Crosses zero |
|---|---|---:|---|---|
| CNN vs QSVM | F1 at tuned thresholds | **0.0302** | [0.0166, 0.0434] | no |
| CNN vs QSVM | ROC-AUC | −0.0012 | [−0.0061, 0.0030] | **yes** |
| CNN vs VQC | F1 at tuned thresholds | 0.1505 | [0.1308, 0.1712] | no |
| CNN vs VQC | ROC-AUC | 0.0909 | [0.0768, 0.1055] | no |
| QSVM vs VQC | F1 at tuned thresholds | 0.1203 | [0.1026, 0.1384] | no |
| QSVM vs VQC | ROC-AUC | 0.0921 | [0.0792, 0.1074] | no |

**Wilcoxon.** All pairwise p-values are 0.25–0.50 with 3 folds; none is significant. This is a
power limitation, not evidence of equivalence.

**Interpretation.** The F1 advantage of the CNN over the QSVM is small but resolved on this test
set. The ROC-AUC difference is unresolved and slightly favours the QSVM. McNemar says the error
patterns differ, not that the CNN dominates every operating point. No quantum-advantage claim is
supported.

**Artefacts.** `results/metrics/statistical_tests.json`.

---

## E4 — Unseen attack-category generalisation

**Protocol.** Remove **all R2L and U2R rows** from the training pool. Retrain all models on
Normal + DoS + Probe. Build a balanced holdout of Normal and held-out-category attacks from the
test portion (84 normal + 84 attacks = 168 rows). Tune each model's threshold on its own
held-out-category validation split, then score once.

**Result.**

| Model | Threshold | Recall | Precision | F1 | ROC-AUC | FPR | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN+LSTM | 0.30 | 0.0476 | 0.6667 | 0.0889 | 0.7484 | 0.0238 | 0.0642 |
| QSVM | 0.50 | 0.0119 | 0.3333 | 0.0230 | 0.6559 | 0.0238 | −0.0449 |
| VQC | 0.45 | **0.7619** | **0.8205** | **0.7901** | **0.8707** | 0.1667 | 0.5968 |

Held-out counts in the full quick pool: R2L 78, U2R 6. Test label composition: 84 Normal, 70
`warezclient`, 4 `guess_passwd`, 3 `buffer_overflow`, 2 `loadmodule`, 2 `warezmaster`, 1 each of
`multihop`, `rootkit`, `imap`. Per-category VQC detection is 78.2% for R2L and 50.0% for U2R; the
CNN detects 3.8% of R2L and 16.7% of U2R; the QSVM detects 0% of R2L and 16.7% of U2R.

**Interpretation.** This is the one experiment where a quantum model leads by a wide margin. The
VQC flags 76.2% of held-out attacks at 16.7% FPR, while the CNN and QSVM collapse to near-zero
recall. The result must be read with care:

- The holdout is small (168 rows, 84 attacks) and comes from a single seed.
- The VQC's threshold was tuned on a small held-out-category validation split; the operating
  point is therefore sensitive.
- The CNN and QSVM were selected for high precision/F1 on the original distribution; their
  thresholds do not transfer to a new category.
- This is **unseen-category generalisation**, not zero-day detection.

The result is reported as an interesting negative-for-classical finding, not as evidence of
general quantum superiority.

**Artefacts.** `results/tables/unseen_category_results.csv`,
`results/metrics/unseen_category_details.json`, `results/figures/unseen_attack_performance.png`.

---

## E5 — External novel attack types (KDDTest+)

**Protocol.** Define novel types against the full KDDTrain+ label set. Select KDDTest+ rows whose
type is novel plus all Normal rows; exclude known attack types. Score the already-trained main
models at their original thresholds. Novelty is therefore relative to the full training
vocabulary, while covariate shift is also present because KDDTest+ differs from KDDTrain+.

**Composition.** 17 novel types, **3,750 novel attack rows + 9,711 Normal rows = 13,461 rows**.

**Result.**

| Model | Recall | Precision | F1 | ROC-AUC | FPR | MCC |
|---|---:|---:|---:|---:|---:|---:|
| CNN+LSTM | 0.3461 | **0.6234** | 0.4451 | 0.6665 | **0.0807** | 0.3290 |
| QSVM | **0.6197** | 0.6996 | **0.6572** | **0.8917** | 0.1028 | **0.5375** |
| VQC | 0.5731 | 0.4741 | 0.5189 | 0.7801 | 0.2455 | 0.3107 |

Thresholds are the main validation-selected values: 0.35 for the CNN and QSVM, 0.50 for the VQC.

**Interpretation.** The ordering on this subset differs from the primary common test. The QSVM is
strongest by F1, recall, ROC-AUC and MCC; the CNN is the most conservative detector, with the highest
precision and lowest FPR; the VQC is intermediate on F1 and ROC-AUC but has the highest FPR. The
QSVM's F1 advantage over the CNN is 0.2121 and its ROC-AUC advantage is 0.2252.

The external subset is normal-heavy (72.14% Normal), comes from a single file, and inherits the main
validation-selected thresholds. No external-set confidence intervals or multi-seed repeats were
computed. The QSVM's lead is therefore a scope-specific result, not a general quantum advantage or a
deployment recommendation.

**Artefacts.** `results/tables/external_novel_attack_results.csv`,
`results/metrics/external_novel_details.json`.

---

## E6 — Encoding comparison (angle vs amplitude)

**Protocol.** Retrain QSVM and VQC with each encoding on the same quantum training subset and
common test set. Both VQC encodings receive the same COBYLA budget (150 iterations). Thresholds
are tuned on validation for each run.

**Result.**

| Model | Encoding | Qubits | Depth | F1 | ROC-AUC | Train (s) |
|---|---|---:|---:|---:|---:|---:|
| QSVM | angle | 8 | 31 | 0.9231 | **0.9903** | 1.97 |
| QSVM | amplitude | 3 | 1 | **0.9364** | 0.9783 | 0.31 |
| VQC | angle | 8 | 38 | 0.8028 | 0.8982 | 9.58 |
| VQC | amplitude | 3 | 8 | **0.8958** | **0.9609** | 0.77 |

**Interpretation.** Amplitude encoding uses 3 qubits instead of 8 and improves F1 for both
models, with a small ROC-AUC cost for the QSVM and a large AUC gain for the VQC. It is also much
faster to train. However, amplitude encoding with basis-state preparation produces a
squared-cosine fidelity kernel:

\[
K(x,y) = \left|\sum_i \sqrt{p_i(x)p_i(y)}\,e^{i(\phi_i(x)-\phi_i(y))}\right|^2,
\]

which for real non-negative amplitudes is a classical polynomial kernel. The comparison is an
empirical resource/performance check, not evidence of quantum advantage. The angle encoding is
kept as the primary configuration because it uses the feature map explicitly and is the setting
usually claimed to be hard to simulate classically.

**Artefacts.** `results/tables/encoding_comparison.csv`,
`results/figures/encoding_comparison.png`.

---

## E7 — Feature-reduction ablation

**Protocol.** Fit PCA and mutual-information reducers at 8, 12 and 16 dimensions on train only.
Train a logistic-regression probe on each reduced training matrix and score on validation. The
test set is not used.

**Result.** See the table in [methodology.md](methodology.md#5-dimensionality-reduction-and-feature-selection).
PCA-16 is best overall (validation F1 0.9295); PCA-8 is the best feasible configuration under the
8-qubit simulation cap (validation F1 0.9161). MI is consistently weaker than PCA at the same
dimension.

**Interpretation.** The 8-qubit cap costs roughly 1.3 F1 points on the validation probe relative
to PCA-16. This is a real cost of the quantum-feasibility constraint and applies equally to all
models in the main comparison. It is not corrected post hoc.

**Artefacts.** `results/tables/feature_reduction_ablation.csv`,
`data/metadata/feature_selection.json`, `results/figures/feature_reduction.png`,
`results/figures/pca_explained_variance.png`.

---

## E8 — Latency and quantum resources

**Protocol.** Measure single-sample end-to-end latency (preprocessing + reduction + inference),
batch inference at batch size 256, and throughput. Repeat five times and take the median. Count
circuit qubits, depth, size, parameters and two-qubit gates from the actual circuits.

**Latency (single sample, CPU).**

| Model | Preprocess (ms) | Reduction (ms) | Inference (ms) | Throughput (samples/s) | Slowdown |
|---|---:|---:|---:|---:|---:|
| CNN+LSTM | 0.00986 | 0.00084 | **0.9216** | **183,947.7** | 1.00× |
| QSVM | 0.00986 | 0.00084 | 3.9428 | 385.2 | 4.28× |
| VQC | 0.00986 | 0.00084 | 32.7295 | 332.0 | 35.51× |

**Quantum resources (simulation).**

| Model | Encoding | Qubits | Depth | Size | Parameters | 2Q gates | Backend |
|---|---|---:|---:|---:|---:|---:|---|
| QSVM | angle | 8 | 31 | 74 | 8 | 28 | Statevector, exact fidelity |
| VQC | angle | 8 | 38 | 112 | 24 | 42 | Statevector, analytic probabilities |

**Interpretation.** Batch inference is much faster than single-sample inference because circuit
construction/state-preparation overhead is amortised. The QSVM's batch throughput is competitive
with the CNN in this simulation, but its single-sample latency is 4.3× higher. The VQC is 35.5×
slower per sample. These numbers are CPU-only and load-dependent; they are not a hardware
benchmark. On real quantum hardware, additional sampling, communication, queueing and error
correction costs would apply.

**Artefacts.** `results/tables/latency_comparison.csv`, `results/tables/quantum_resources.csv`,
`results/circuits/`, `results/figures/latency_comparison.png`, `quantum_resources.png`.

---

## E9 — Explainability

**Protocol.**

- **Permutation importance** on the CNN+LSTM: 10 repeats, 500 test samples, ROC-AUC drop.
- **PCA attribution**: project absolute PCA loadings back to original features.
- **SHAP**: KernelSHAP with 50 background rows and 100 explained rows on the CNN probability
  function.
- **Quantum input sensitivity**: perturb each reduced feature by δ = 0.25 and measure the mean
  absolute change in predicted probability (100 test samples).

**Result (top signals).**

| Analysis | Top signals |
|---|---|
| Permutation (ROC-AUC drop) | PC1 0.350, PC6 0.040, PC2 0.024, PC4 0.014 |
| SHAP (mean absolute) | PC1 0.356, PC2 0.068, PC4 0.064, PC5 0.034 |
| PCA attribution | `dst_host_srv_serror_rate` 0.122, `dst_host_serror_rate` 0.122, `serror_rate` 0.122, `srv_serror_rate` 0.122 |
| QSVM sensitivity | PC3 0.178, PC5 0.169, PC2 0.122, PC6 0.117 |
| VQC sensitivity | PC3 0.203, PC2 0.161, PC1 0.139, PC4 0.025 |

**Interpretation.** The classical model concentrates on PC1 plus a small number of secondary
components. The PCA attribution maps these to connection-error-rate features, which is plausible
for DoS/Probe detection. The quantum sensitivity rankings differ from the classical importance
ranking; this is an input-sensitivity analysis of the encoded circuit, not a Shapley attribution.

The VQC is almost completely insensitive to PC6–PC8 (changes ~1e-16). This is an architectural
property of the chosen circuit/measurement, not evidence that those features are irrelevant. It
is a useful warning against interpreting circuit sensitivity as causal feature importance.

**Artefacts.** `results/explainability/`, `results/figures/permutation_importance.png`,
`shap_summary.png`, `qsvm_feature_sensitivity.png`, `vqc_feature_sensitivity.png`.

---

## E10 — Distribution-shift stress test

**Protocol.** Two controlled covariate shifts on the common test set:

1. **Gaussian noise** at severities 0.0, 0.1, 0.25, 0.5, 1.0 (standardised feature space).
2. **Feature permutation** at fractions 0.0, 0.1, 0.25, 0.5 (randomly permute that fraction of
   features across rows, destroying their relationship with the label).

Also scan the external KDDTest+ file in five file-order windows of ~4,509 rows. NSL-KDD has no
timestamps, so this is a *simulated* window scan, not real temporal drift.

Only the CNN+LSTM is evaluated in this experiment. This is a deliberate scope limit: the point
is to characterise the baseline's failure envelope, not to compare models under shift.

**Controlled-shift result.**

| Shift | Severity | F1 | ROC-AUC | KS mean |
|---|---:|---:|---:|---:|
| Gaussian noise | 0.0 | 0.9533 | 0.9891 | 0.0197 |
| Gaussian noise | 0.5 | 0.8984 | 0.9629 | 0.1371 |
| Gaussian noise | 1.0 | 0.7882 | 0.8755 | 0.2160 |
| Feature permutation | 0.1 | 0.9167 | 0.9778 | 0.0197 |
| Feature permutation | 0.25 | 0.4546 | 0.5613 | 0.0197 |
| Feature permutation | 0.5 | 0.4685 | 0.5803 | 0.0197 |

**File-order windows (KDDTest+).** F1 ranges 0.7308–0.7470 across five windows; ROC-AUC
0.8351–0.8457. The external F1 is much lower than the common-test F1 because KDDTest+ contains
novel attack types and a different distribution, not because of temporal drift.

**Interpretation.** The model is robust to moderate Gaussian noise but collapses when more than
~25% of features are permuted, as expected. The KS statistic rises with noise severity. The
window scan shows no meaningful trend across file order, which is consistent with there being no
real time axis. Do not describe this as concept-drift detection.

**Artefacts.** `results/tables/drift_controlled.csv`, `drift_summary.csv`,
`drift_windows_external.csv`, `results/metrics/drift.json`,
`results/figures/drift_gaussian_noise.png`, `drift_feature_permutation.png`,
`drift_simulated_windows.png`.

---

## E11 — Tiered alert engine and dashboard

**Protocol.** Map each predicted attack probability to a five-level policy:

| Probability | Level | Action |
|---|---|---|
| < 0.20 | NORMAL | ALLOW |
| 0.20–0.45 | MONITOR | LOG_AND_MONITOR |
| 0.45–0.70 | ALERT | NOTIFY_ANALYST |
| 0.70–0.90 | QUARANTINE | QUARANTINE_HOST |
| ≥ 0.90 | ESCALATE | BLOCK_AND_ESCALATE |

R2L/U2R category hints bump severity by one level (capped at ESCALATE). The engine is assistive;
the policy text explicitly says no automatic blocking below ESCALATE.

**Result on the common test set (CNN+LSTM scores).**

| Level | Count | Fraction |
|---|---:|---:|
| NORMAL | 920 | 51.1% |
| MONITOR | 67 | 3.7% |
| ALERT | 33 | 1.8% |
| QUARANTINE | 36 | 2.0% |
| ESCALATE | 744 | 41.3% |

**Interpretation.** The high ESCALATE fraction reflects the 46.6% attack rate in the benchmark
test set and the conservative bands; it is not a realistic enterprise alert volume. The point is
to demonstrate a traceable mapping from probability to analyst action, not to claim operational
calibration.

The Streamlit dashboard reads the saved artefacts and provides overview, model comparison,
quantum-resource, robustness, alert-triage and explainability pages. The static
`results/dashboard.html` is generated for offline viewing.

**Artefacts.** `results/metrics/alert_summary.json`, `results/dashboard.html`,
`src/dashboard/app.py`.

---

## Summary of outcomes

| Finding | Status |
|---|---|
| Classical CNN+LSTM beats QSVM on F1 by 3.02 points, CI excludes zero | Confirmed on this test set |
| QSVM ROC-AUC is 0.0012 higher, CI includes zero | Not resolved |
| VQC is weaker than both on the main task | Confirmed |
| Amplitude encoding improves F1 and reduces qubit count | Confirmed empirically |
| VQC leads on unseen R2L/U2R recall, but with 16.7% FPR and n=168 | Suggestive, not conclusive |
| QSVM leads external novel-type F1 (0.6572) and AUC (0.8917) | Scope-specific; one normal-heavy file; no external CIs |
| CNN has the lowest external novel-type FPR (0.0807) | Confirmed on this subset |
| Quantum models are slower per sample in simulation | Confirmed |
| No quantum advantage on the primary common test | Supported |
