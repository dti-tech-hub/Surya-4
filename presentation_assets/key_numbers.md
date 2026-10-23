# QE-IDS — Key Numbers

**One-page number sheet for viva, report and slides.**
Source of truth: `results/final_summary.json`. Profile: **quick**, seed **42**, common test **n = 1,800**.
All quantum results are **exact statevector simulations**, not hardware.

---

## 1. Dataset and splits

| Item | Value |
|---|---:|
| `KDDTrain+` rows | 125,973 |
| `KDDTest+` rows (external, never trained on) | 22,544 |
| Quick stratified pool | 12,000 |
| Train / validation / test | 8,399 / 1,801 / 1,800 |
| Attack rate (train / val / test) | 46.54% / 46.53% / 46.56% |
| Raw features | 41 |
| Design-matrix columns | 111 |
| Train attack types | 23 |
| External test-only attack types | 17 |
| Missing values / exact duplicates | 0 / 0 |

---

## 2. Feature reduction (validation probe F1)

| Method | 8 dims | 12 dims | 16 dims |
|---|---:|---:|---:|
| PCA | 0.9161 | 0.9252 | **0.9295** |
| Mutual information | 0.8715 | 0.8734 | 0.9132 |

- Selected configuration: **PCA-8**, explained variance **0.6666**.
- Reason: quantum simulator capped at 8 qubits; applied to **all** models.
- Cost relative to PCA-16: **1.34 validation F1 points**.

---

## 3. Main test results

| Model | Encoding | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR | MCC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **CNN+LSTM** | none | **0.9567** | **0.9567** | 0.9499 | **0.9533** | 0.9891 | **0.0374** | 0.0501 | **0.9129** |
| QSVM | angle | 0.9244 | 0.8774 | **0.9737** | 0.9231 | **0.9903** | 0.1185 | **0.0263** | 0.8537 |
| VQC | angle | 0.7933 | 0.7223 | 0.9033 | 0.8028 | 0.8982 | 0.3025 | 0.0967 | 0.6077 |

Validation-selected thresholds: **CNN 0.35**, **QSVM 0.35**, **VQC 0.50**.

Confusion matrices (TN / FP / FN / TP):

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| CNN+LSTM | 926 | 36 | 42 | 796 |
| QSVM | 848 | 114 | 22 | 816 |
| VQC | 671 | 291 | 81 | 757 |

---

## 4. Cross-validation (3-fold, F1)

| Model | Mean ± SD | 95% CI | Fold values |
|---|---:|---:|---|
| CNN+LSTM | **0.9346 ± 0.0135** | 0.9010–0.9682 | 0.9500, 0.9293, 0.9246 |
| QSVM | 0.9180 ± 0.0161 | 0.8779–0.9580 | 0.9012, 0.9333, 0.9194 |
| VQC | 0.7452 ± 0.0527 | 0.6144–0.8761 | 0.6906, 0.7494, 0.7957 |

CV accuracy: CNN 0.9400 ± 0.0123; QSVM 0.9283 ± 0.0120; VQC 0.7613 ± 0.0580.

---

## 5. Paired statistics

| Test | Comparison | Result |
|---|---|---|
| McNemar | CNN vs QSVM | χ² = 22.253, **p = 2.389 × 10⁻⁶**, 146 discordant |
| McNemar | CNN vs VQC | χ² = 214.623, p = 1.347 × 10⁻⁴⁸ |
| McNemar | QSVM vs VQC | χ² = 152.555, p = 4.792 × 10⁻³⁵ |
| Paired bootstrap F1 | CNN − QSVM | **0.0302 [0.0166, 0.0434]** |
| Paired bootstrap AUC | CNN − QSVM | **−0.0012 [−0.0061, 0.0030]** (crosses 0) |
| Paired bootstrap F1 | CNN − VQC | 0.1505 [0.1308, 0.1712] |
| Paired bootstrap AUC | CNN − VQC | 0.0909 [0.0768, 0.1055] |
| Wilcoxon (3 folds) | CNN vs QSVM | p = 0.50, not significant |
| Wilcoxon (3 folds) | CNN vs VQC | p = 0.25, not significant |
| Wilcoxon (3 folds) | QSVM vs VQC | p = 0.25, not significant |

Effect size CNN vs QSVM: Cohen’s h = **0.138** (negligible).
Caveat: with 3 folds the minimum two-sided Wilcoxon p is 0.25; the test is underpowered.

---

## 6. Unseen category (R2L+U2R held out; balanced n = 168)

| Model | Threshold | Accuracy | Recall | F1 | FPR | MCC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| VQC | 0.45 | **0.7976** | **0.7619** | **0.7901** | 0.1667 | **0.5968** | **0.8707** |
| CNN+LSTM | 0.30 | 0.5119 | 0.0476 | 0.0889 | 0.0238 | 0.0642 | 0.7484 |
| QSVM | 0.50 | 0.4940 | 0.0119 | 0.0230 | 0.0238 | −0.0449 | 0.6559 |

Held-out counts: R2L 78, U2R 6. Train categories: Normal, DoS, Probe.
**This is unseen-category generalisation, not a zero-day claim.**

---

## 7. External novel attack types (17 types, 3,750 novel + 9,711 normal)

| Model | Threshold | Accuracy | Precision | Recall | F1 | FPR | MCC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN+LSTM | 0.35 | 0.7596 | **0.6234** | 0.3461 | 0.4451 | **0.0807** | 0.3290 | 0.6665 |
| VQC | 0.50 | 0.7040 | 0.4741 | 0.5731 | 0.5189 | 0.2455 | 0.3107 | 0.7801 |
| QSVM | 0.35 | **0.8199** | 0.6996 | **0.6197** | **0.6572** | 0.1028 | **0.5375** | **0.8917** |

Novelty reference: **full `KDDTrain+` label set**, not the 12,000-row quick pool.
The QSVM leads F1 and ROC-AUC; the CNN has the lowest FPR. This is a scope-specific result on one
normal-heavy external file, not a general quantum advantage.
Caveat: `KDDTest+` also differs in covariate distribution, so this is novel-type generalisation
under shift, not an isolated zero-day effect.

---

## 8. Encoding comparison (equal optimizer budget)

| Model | Encoding | Qubits | Depth | F1 | ROC-AUC | FPR |
|---|---|---:|---:|---:|---:|---:|
| QSVM | angle | 8 | 31 | 0.9231 | **0.9903** | 0.1185 |
| QSVM | amplitude | 3 | 1 | **0.9364** | 0.9783 | **0.0156** |
| VQC | angle | 8 | 38 | 0.8028 | 0.8982 | 0.3025 |
| VQC | amplitude | 3 | 8 | **0.8958** | **0.9609** | **0.0374** |

Amplitude encoding is stronger and cheaper **on this dataset**, but for computational-basis
preparation its fidelity kernel is a squared-cosine polynomial kernel. Not evidence of quantum
advantage.

---

## 9. Latency and quantum resources

Single-sample CPU medians (1,800 samples, 5 repeats):

| Model | Inference ms/sample | End-to-end ms/sample | Throughput samples/s | Slowdown |
|---|---:|---:|---:|---:|
| CNN+LSTM | 0.9216 | 0.9323 | 183,947.7 | 1.00× |
| QSVM | 3.9428 | 3.9535 | 385.2 | **4.28×** |
| VQC | 32.7295 | 32.7402 | 332.0 | **35.51×** |

Resources:

| Model | Qubits | Depth | Size | Two-qubit gates | Trainable parameters | Optimizer |
|---|---:|---:|---:|---:|---:|---|
| QSVM | 8 | 31 | 74 | 28 | 8 (feature map; 0 trainable) | — |
| VQC | 8 | 38 | 112 | 42 | 24 | COBYLA |

VQC optimiser: 150 iterations, 151 evaluations, final loss **0.4585**, **converged = false**.
QSVM kernel cross-check: max abs difference **0.00171** vs tolerance **0.00632** (100,000 shots,
16 samples).

---

## 10. Drift, alerting and explainability

Controlled drift (classical CNN+LSTM only):

| Shift | Baseline F1 | Worst F1 | Degradation |
|---|---:|---:|---:|
| Feature permutation | 0.9533 | **0.4685** | **0.4848** |
| Gaussian noise | 0.9533 | **0.7882** | **0.1651** |

Alert tiers on 1,800 common-test records:

| Tier | Count | Fraction | Action |
|---|---:|---:|---|
| NORMAL | 920 | 51.11% | ALLOW |
| MONITOR | 67 | 3.72% | LOG_AND_MONITOR |
| ALERT | 33 | 1.83% | NOTIFY_ANALYST |
| QUARANTINE | 36 | 2.00% | QUARANTINE_HOST |
| ESCALATE | 744 | 41.33% | BLOCK_AND_ESCALATE |

Explainability outputs: permutation importance, PCA component attribution, SHAP summary,
QSVM/VQC input sensitivity. All are in `results/explainability/` and `results/figures/`.

---

## 11. Environment and reproducibility

| Item | Value |
|---|---|
| Python | 3.14.5 |
| Qiskit / Aer / ML | 2.6.0 / 0.18.0 / 0.9.1 |
| PyTorch | 2.14.1 (CPU) |
| scikit-learn | 1.8.0 |
| Hardware | 12 logical CPUs, 15.72 GB RAM, no GPU |
| Seed | 42 |
| End-to-end quick run | 603.3 s (14 stages) |
| Test suite | 31 tests passing (re-verify before submission) |

---

## 12. Three sentences to memorise

1. **The classical CNN+LSTM wins on F1 (0.9533) and FPR (0.0374); the QSVM is close on F1
   (0.9231) and has an unresolved AUC edge.**
2. **The VQC is weaker (F1 0.8028) and did not converge in its budget, but it detects unseen
   R2L/U2R attacks far better in the small balanced holdout (recall 0.7619).**
3. **All quantum results are exact statevector simulations with no hardware claim, and no quantum
   advantage is observed under this controlled design.**
