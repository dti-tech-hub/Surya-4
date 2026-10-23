# QE-IDS — Talk Outline

**Target:** 12–15 minute conference-style talk, plus 3–5 minutes of questions.
**Deck theme:** one narrow question, one controlled protocol, one honest negative result.
**All numbers are from the executed quick profile (seed 42, common test n = 1,800).**

---

## Slide 1 — Title (30 s)

**On slide**

> Quantum-Enhanced Intrusion Detection System
> A leakage-audited comparison of CNN+LSTM, QSVM and VQC on NSL-KDD
> B.Tech Final-Year Project · 2026

**Speaker notes**

- Introduce the project as a *measurement* study, not a product pitch.
- State up front: all quantum results are exact statevector simulations on a CPU.
- State the answer early: the classical baseline wins on F1; no quantum advantage is claimed.

**Asset:** `presentation_assets/architecture.png`

---

## Slide 2 — The question (45 s)

**On slide**

> Given identical reduced inputs, identical splits, identical test samples and an identical
> evaluation protocol, does a quantum classifier improve binary intrusion detection over a
> properly tuned classical CNN+LSTM?

Three sub-questions:

1. Is QSVM competitive with a tuned CNN+LSTM?
2. Does a VQC add value under a realistic optimisation budget?
3. Where do quantum models fail under distribution shift?

**Speaker notes**

- Emphasise “identical”. Most claimed advantages change the baseline, the features or the split.
- This project deliberately removes those degrees of freedom.
- Negative results are a valid outcome and are reported as such.

---

## Slide 3 — Why this is hard to compare fairly (45 s)

**On slide**

Common failure modes in QML-for-IDS papers:

- classical baseline not tuned to convergence;
- quantum model sees different features or a different split;
- preprocessing fitted on train + test;
- threshold chosen on the test set;
- “zero-day” claimed from a test file that was never trained on, without separating
  novel attack *types* from covariate shift.

**Speaker notes**

- Our protocol closes each of these holes.
- The leakage audit is automated and saved as JSON.
- We distinguish “unseen attack category” from “external novel type” from “zero-day”.

---

## Slide 4 — Dataset and splits (60 s)

**On slide**

| Item | Value |
|---|---|
| Training file | `KDDTrain+` — 125,973 rows |
| External file | `KDDTest+` — 22,544 rows, never trained on |
| Quick pool | 12,000 stratified rows from `KDDTrain+` |
| Train / validation / test | 8,399 / 1,801 / 1,800 |
| Attack rate | 46.54% / 46.53% / 46.56% |
| Train labels | 23 attack types + normal |
| Test-only attack types | 17 |

**Speaker notes**

- The quick pool exists to make repeated statevector experiments affordable.
- The external file supplies the 17 novel types and 9,711 normal records.
- No exact duplicate rows and no missing values in either raw file.

**Asset:** `results/figures/class_distribution.png`

---

## Slide 5 — Leakage-safe preprocessing (60 s)

**On slide**

Fitted on the **training split only**:

- median imputation,
- one-hot vocabulary (111 design columns),
- standardisation,
- PCA loadings / mutual-information ranking,
- min–max rescaling to `[0, π]`.

Eight automated checks — all passed:

`scaler_fit_on_train`, `onehot_vocabulary_from_train`, `transform_is_stateless`,
`reducer_input_dimension`, `reducer_output_dimension`, `no_exact_train_test_row_overlap`,
`reduced_shapes`, `pca_explained_variance_monotone`.

**Speaker notes**

- The audit is part of the pipeline, not a manual appendix.
- Validation chooses the feature configuration, early-stopping epoch and threshold.
- Test is scored once.

**Asset:** `results/figures/pca_explained_variance.png`

---

## Slide 6 — Feature reduction: why 8 dimensions (45 s)

**On slide**

Validation F1 of a logistic-regression probe:

| Method | 8 dims | 12 dims | 16 dims |
|---|---:|---:|---:|
| PCA | 0.9161 | 0.9252 | **0.9295** |
| Mutual information | 0.8715 | 0.8734 | 0.9132 |

- PCA-16 is best overall.
- The statevector simulator is capped at 8 qubits.
- PCA-8 is therefore used for **all three models**, including the classical baseline.
- Explained variance at 8 components: 66.7%.

**Speaker notes**

- The cap costs about 1.3 validation F1 points relative to PCA-16.
- This is a disclosed simulator constraint, not a post-hoc choice.
- Giving the classical model the same 8 features is essential for fairness.

**Asset:** `results/figures/feature_reduction.png`

---

## Slide 7 — Three models, one representation (75 s)

**On slide**

**CNN+LSTM** — pseudo-sequence of length 8; BatchNorm → Conv1d(32) → MaxPool → Dropout →
LSTM(32) → dense head; class-weighted BCE; AdamW; early stopping; 9,667 parameters.

**QSVM** — 8-qubit ZZ feature map (2 reps, linear entanglement); exact fidelity kernel
`K = |S S†|²`; precomputed-kernel SVC with Platt scaling.

**VQC** — 8-qubit ZZ feature map + RealAmplitudes (2 reps); 24 parameters; COBYLA;
`P(qubit 0 = |1>)` from exact statevector probabilities.

Training budgets: CNN 8,399 rows; QSVM/VQC 600 rows (200 validation) because statevector
cost grows with training-set size.

**Speaker notes**

- The budget asymmetry is explicit and is a limitation, not a hidden advantage.
- The QSVM kernel was cross-checked against Qiskit `FidelityQuantumKernel` with 100,000 shots:
  max absolute difference 0.00171 against a 0.00632 tolerance.
- The VQC fast path is mathematically equivalent to full-circuit simulation and is unit-tested.

**Asset:** `results/circuits/qsvm_circuit.txt`, `results/circuits/vqc_circuit.txt`

---

## Slide 8 — Main result (75 s)

**On slide**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|
| **CNN+LSTM** | **0.9567** | **0.9567** | 0.9499 | **0.9533** | 0.9891 | **0.0374** |
| QSVM | 0.9244 | 0.8774 | **0.9737** | 0.9231 | **0.9903** | 0.1185 |
| VQC | 0.7933 | 0.7223 | 0.9033 | 0.8028 | 0.8982 | 0.3025 |

- CNN leads F1 by **0.0302** over QSVM.
- QSVM leads ROC-AUC by **0.0012**, but the paired CI crosses zero.
- VQC trails on every headline metric.

**Speaker notes**

- Do not say “quantum is worse” in general; say “under this design, on this dataset”.
- The QSVM’s higher recall comes at a 3.2× higher false-positive rate than the CNN.
- Thresholds: CNN 0.35, QSVM 0.35, VQC 0.50, all selected on validation.

**Asset:** `results/figures/model_comparison.png`, `results/figures/roc_comparison.png`

---

## Slide 9 — Statistics: the difference is real but small (60 s)

**On slide**

- McNemar CNN vs QSVM: **p = 2.389 × 10⁻⁶**, 146 discordant pairs, Cohen’s h = 0.138
  (negligible).
- McNemar CNN vs VQC: p = 1.347 × 10⁻⁴⁸.
- Paired bootstrap F1 (CNN − QSVM): **0.0302 [0.0166, 0.0434]**.
- Paired bootstrap AUC (CNN − QSVM): **−0.0012 [−0.0061, 0.0030]** — crosses zero.
- 3-fold Wilcoxon: not significant; with 3 folds the minimum two-sided p is 0.25.

**Speaker notes**

- McNemar says the *pattern of errors* differs; it does not say the gap is large.
- The AUC difference is unresolved, so we do not claim the CNN is better on ranking.
- Wilcoxon is underpowered by design; report it as supporting evidence only.

**Asset:** `results/figures/cross_validation_f1.png`

---

## Slide 10 — Cross-validation (45 s)

**On slide**

3-fold CV F1:

| Model | Mean ± SD | 95% CI |
|---|---:|---:|
| CNN+LSTM | **0.9346 ± 0.0135** | 0.9010–0.9682 |
| QSVM | 0.9180 ± 0.0161 | 0.8779–0.9580 |
| VQC | 0.7452 ± 0.0527 | 0.6144–0.8761 |

The ordering is stable across folds. Fold count is reported explicitly (3 in quick mode,
5 in the unexecuted full profile).

**Speaker notes**

- CV uses internal train/validation splits per fold; the common test set is untouched.
- Three folds give wide intervals; do not over-read the second decimal.

**Asset:** `results/figures/cross_validation.png`

---

## Slide 11 — Generalisation stress tests (75 s)

**On slide**

**Unseen category** — R2L+U2R removed from training; balanced holdout n = 168:

| Model | Recall | F1 | FPR |
|---|---:|---:|---:|
| VQC | **0.7619** | **0.7901** | 0.1667 |
| CNN+LSTM | 0.0476 | 0.0889 | 0.0238 |
| QSVM | 0.0119 | 0.0230 | 0.0238 |

**External novel types** — 17 types, 3,750 novel attacks + 9,711 normal:

| Model | F1 | Recall | FPR | MCC |
|---|---:|---:|---:|---:|
| CNN+LSTM | 0.4451 | 0.3461 | **0.0807** | 0.3290 |
| VQC | 0.5189 | 0.5731 | 0.2455 | 0.3107 |
| QSVM | **0.6572** | **0.6197** | 0.1028 | **0.5375** |

**Speaker notes**

- The VQC result on the small balanced holdout is striking but single-seed and not evidence of
  general quantum superiority.
- On the external set the QSVM leads by F1 and ROC-AUC, but its FPR is 10.3%; the CNN is the most
  conservative detector at 8.1% FPR. This is a scope-specific result on one normal-heavy file, not
  a general quantum advantage.
- “External novel type” is not “zero-day”: KDDTest+ also differs in covariate distribution.

**Asset:** `presentation_assets/zero_day_results.png`, `results/figures/unseen_attack_performance.png`

---

## Slide 12 — Encoding, latency and resources (60 s)

**On slide**

Encoding comparison (equal optimizer budget):

| Model | Angle F1 | Amplitude F1 | Qubits (angle → amplitude) |
|---|---:|---:|---:|
| QSVM | 0.9231 | **0.9364** | 8 → 3 |
| VQC | 0.8028 | **0.8958** | 8 → 3 |

Single-sample CPU latency:

| Model | ms/sample | Slowdown |
|---|---:|---:|
| CNN+LSTM | 0.9216 | 1.00× |
| QSVM | 3.9428 | 4.28× |
| VQC | 32.7295 | 35.51× |

Quantum resources: QSVM 8 qubits, depth 31, 28 two-qubit gates; VQC 8 qubits, depth 38,
112 gates, 24 parameters, COBYLA used all 151 evaluations and did not converge.

**Speaker notes**

- Amplitude encoding is cheaper and stronger here, but for computational-basis preparation the
  kernel reduces to a squared-cosine polynomial kernel. It is not evidence of quantum advantage.
- Latency is simulator-only on one CPU; it is not a hardware or deployment claim.
- VQC final loss 0.4585, converged = false.

**Asset:** `results/figures/encoding_comparison.png`, `results/figures/latency_comparison.png`,
`results/figures/quantum_resources.png`

---

## Slide 13 — Explainability, drift and alerting (60 s)

**On slide**

- Permutation importance, PCA attribution, SHAP and quantum input sensitivity generated from the
  trained models.
- Controlled drift (classical model only):
  - feature permutation: F1 0.9533 → **0.4685** (−0.4848);
  - Gaussian noise: F1 0.9533 → **0.7882** (−0.1651).
- Tiered alert engine on 1,800 test records:
  - NORMAL 920, MONITOR 67, ALERT 33, QUARANTINE 36, ESCALATE 744.
- Static dashboard: `results/dashboard.html`; interactive: `streamlit run src/dashboard/app.py`.

**Speaker notes**

- Drift is evaluated only for the classical model; this is a stated scope limitation.
- Alert tiers are assistive; no automatic blocking below ESCALATE.
- The high ESCALATE count reflects confident attack probabilities, not a calibrated SOC workload.

**Asset:** `results/figures/permutation_importance.png`,
`results/explainability/shap_summary.png`, `results/figures/drift_feature_permutation.png`

---

## Slide 14 — Limitations (60 s)

**On slide**

1. Quick profile only; full profile implemented but not executed.
2. Quantum training subsets are 600 rows vs 8,399 for the CNN — budget asymmetry.
3. All quantum results are exact statevector simulation; no hardware, no noise model.
4. The 8-qubit cap forces PCA-8; PCA-16 was stronger on validation.
5. VQC did not converge within 150 COBYLA iterations.
6. Drift experiment covers only the classical model.
7. NSL-KDD is old, synthetic in composition, and has no timestamps.
8. The unseen-category holdout is small (n = 168) and single-seed.
9. No claim of zero-day detection, production readiness or quantum advantage.

**Speaker notes**

- Owning the limitations is part of the contribution.
- The negative result is robust to the main sanity checks, but it is not a universal statement.

---

## Slide 15 — Conclusion and questions (45 s)

**On slide**

> Under identical inputs, splits, test samples and evaluation, the classical CNN+LSTM is the
> strongest model on F1 (0.9533), accuracy (0.9567), MCC and FPR (0.0374).
> QSVM is competitive (F1 0.9231) with an unresolved AUC edge.
> VQC is weaker (F1 0.8028) and budget-limited.
> No quantum advantage is observed; all quantum results are simulator-only.

**One-line takeaway**

> Fair comparison is the contribution; the negative result is the finding.

**Speaker notes**

- Invite questions about the protocol first, then the quantum implementation.
- Point to `results/final_summary.json` as the source of truth.
- Mention the repository commands for reproduction.

---

## Backup slides (do not present unless asked)

- **B1:** Full McNemar contingency tables and effect sizes.
- **B2:** VQC convergence curve and optimizer diagnostics.
- **B3:** QSVM kernel cross-check against Qiskit shot-based estimation.
- **B4:** Per-category precision/recall for the external novel-type test.
- **B5:** Alert-tier confusion matrix and policy thresholds.
- **B6:** Environment and package versions (`results/environment.json`).

## Timing plan

| Section | Slides | Minutes |
|---|---|---:|
| Framing | 1–3 | 3.0 |
| Data and protocol | 4–7 | 4.0 |
| Main results | 8–10 | 3.0 |
| Stress tests and systems | 11–13 | 3.5 |
| Limitations and conclusion | 14–15 | 1.5 |
| **Total** | | **15.0** |

## Delivery checklist

- [ ] Rehearse the 15-minute version twice; cut Slide 10 first if running long.
- [ ] Have `results/final_summary.json` open in a second window for exact numbers.
- [ ] Do not improvise new numbers; every figure on the slides is traceable to a result file.
- [ ] Say “simulation” whenever quantum results are shown.
- [ ] Say “unseen category” or “external novel type”, never “zero-day”.
