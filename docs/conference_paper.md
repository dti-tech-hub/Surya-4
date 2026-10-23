# QE-IDS: A Leakage-Audited Comparison of Classical and Quantum Classifiers for Network Intrusion Detection

**B.Tech Final-Year Project — Conference Draft**

*All numerical values in this paper were produced by the executed pipeline described in the
Reproducibility Statement. No placeholders or simulated performance numbers are used.*

---

## Abstract

Quantum machine learning is frequently proposed as a route to stronger network intrusion
detection, but reported gains are often difficult to interpret because classical baselines,
feature representations, splits and evaluation protocols differ. We present a controlled,
leakage-audited comparison on NSL-KDD between a classical CNN+LSTM, a Quantum Support Vector
Machine (QSVM) using an exact fidelity quantum kernel, and a Variational Quantum Classifier (VQC).
All three models consume the same eight-component PCA representation, rescaled to [0, π] using
training statistics, and are scored on the same 1,800-row common test set. Thresholds are selected
on validation data only. On the primary task the classical CNN+LSTM achieves F1 = 0.9533 and
ROC-AUC = 0.9891; the QSVM achieves F1 = 0.9231 and ROC-AUC = 0.9903; the VQC achieves F1 = 0.8028
and ROC-AUC = 0.8982. The CNN+LSTM's F1 advantage over the QSVM is 0.0302 (paired-bootstrap 95% CI
0.0166–0.0434), while the ROC-AUC difference of −0.0012 is unresolved (95% CI −0.0061 to 0.0030).
McNemar's test rejects equal error patterns (p = 2.389 × 10⁻⁶), but the effect size is negligible
(Cohen's h = 0.138). We also report cross-validation, unseen attack-category generalisation,
external novel attack types, angle versus amplitude encoding, latency, circuit resources,
explainability, controlled distribution shift and a tiered alert policy. No quantum advantage is
observed on the primary common test. On the secondary external novel-type subset, the QSVM leads by
F1 (0.6572 vs 0.4451 for the CNN) and ROC-AUC (0.8917 vs 0.6665), a scope-specific nuance rather
than a general advantage. All quantum results come from exact statevector simulation and represent a
shot-noise-free upper bound relative to noisy hardware, not a hardware claim.

**Keywords:** intrusion detection, NSL-KDD, quantum machine learning, quantum kernel, variational
quantum classifier, CNN-LSTM, reproducibility, negative results

---

## 1. Introduction

Network intrusion detection systems (IDS) must separate malicious from benign traffic under
severe class imbalance, evolving attack behaviour and operational latency constraints. Classical
deep learning is the default tool, but it is data-hungry and can be difficult to calibrate and
explain. Quantum classifiers promise richer feature spaces through high-dimensional state
representations [1, 2]. That promise is often evaluated with inconsistent classical baselines,
different feature subsets, or test-set tuning, which makes claimed advantages hard to reproduce.

This paper asks a narrow, testable question:

> Given identical reduced inputs, identical splits, identical test samples and an identical
> evaluation protocol, does a quantum classifier improve binary intrusion detection over a
> properly tuned classical CNN+LSTM on NSL-KDD?

We answer it with a fully executed pipeline. The contributions are:

1. A leakage-audited preprocessing and evaluation protocol in which imputation, one-hot encoding,
   scaling, PCA/mutual-information reduction and quantum angle rescaling are fitted exclusively on
   training data.
2. A three-way comparison on the same reduced representation and the same test rows, with
   validation-only threshold selection.
3. An unseen attack-category experiment that removes R2L and U2R from training entirely, plus an
   external novel-type test on KDDTest+ that reveals a scope-specific QSVM advantage.
4. A systems analysis covering cross-validation, paired statistics, latency, circuit resources,
   explainability and controlled distribution shift.
5. Explicit reporting of the negative primary result, the scope-specific external exception and
   inconclusive results.

The main finding is negative for quantum advantage on the primary common test: the classical
CNN+LSTM is the strongest model on F1, accuracy, MCC and false-positive rate. The QSVM is
competitive on F1 and marginally higher on ROC-AUC, but the AUC difference is not resolved. The VQC
is clearly weaker and did not converge within its optimisation budget. The secondary external
novel-type subset reverses the ordering: the QSVM leads by F1 and ROC-AUC, while the CNN retains the
lowest FPR. We report that as a scope-specific result, not a general advantage.

---

## 2. Related work

Quantum kernel methods estimate similarity in a feature Hilbert space induced by a quantum feature
map [1, 2]. They are attractive because they can be analysed with classical kernel theory and
because the kernel may be hard to estimate classically for certain maps. Variational quantum
classifiers instead train a parameterised circuit with a classical optimiser [3]. Both approaches
have been applied to intrusion detection, but comparisons often use small subsamples, classical
baselines that are not tuned to convergence, or no paired statistical testing.

NSL-KDD [4] remains a standard benchmark despite its age. It is useful for controlled comparison,
not for claims about modern enterprise traffic. The KDD Cup 99 lineage, the synthetic attack mix
and the absence of timestamps are well-known limitations. We use NSL-KDD because it permits a
clean, reproducible protocol and because it is small enough to run exact statevector simulations
on a CPU.

Our protocol addresses three recurring weaknesses: (i) the same reduced matrix is given to every
model; (ii) all fitted transformations are train-only; and (iii) differences are tested on paired
predictions rather than compared as independent point estimates.

---

## 3. Dataset and experimental design

### 3.1 Data

NSL-KDD provides 125,973 labelled records in `KDDTrain+.txt` and 22,544 in `KDDTest+.txt`. Each
record has 41 features (3 categorical: `protocol_type`, `service`, `flag`; 38 numeric), a label and
a difficulty score. The difficulty column is dropped. Labels are mapped to the five NSL-KDD
categories (Normal, DoS, Probe, R2L, U2R) and to a binary attack target.

Verified properties: 23 distinct labels in KDDTrain+, 38 in KDDTest+, 17 attack types present only
in KDDTest+, zero missing values and zero exact duplicate rows in both files. The official
KDDTest+ file is never used for training, early stopping, threshold selection or calibration.

### 3.2 Splits and leakage control

The quick profile draws a stratified 12,000-row subsample of KDDTrain+ and splits it 70/15/15 on
the binary label. The resulting train/validation/test sizes are 8,399/1,801/1,800, with attack
rates 46.54%/46.53%/46.56%. The test split is the common test set for all models.

Every fitted transformation is estimated on the training split only:

- median imputation values,
- one-hot vocabulary (rare levels collapsed using training frequencies),
- numeric standardisation statistics,
- PCA loadings or mutual-information scores,
- post-reduction min–max scaling to [0, π].

Validation selects the early-stopping epoch, decision threshold and feature configuration. The
test split is scored once. An eight-check automated audit verifies scaler statistics, one-hot
vocabulary, stateless transforms, reducer dimensions, exact row overlap, split integrity, design
matrix ranges and PCA variance monotonicity. All eight checks passed.

The full profile in the configuration (full pool, 3,000 quantum training rows, 2,500 test rows,
5 folds) was not executed; no full-profile numbers are reported.

### 3.3 Preprocessing and feature reduction

Categorical features are one-hot encoded with a train-fit vocabulary; numeric features are
median-imputed and standardised; infinite values are replaced before imputation; constant columns
are dropped. The resulting design matrix has 111 columns.

PCA and mutual-information ranking are compared at 8, 12 and 16 dimensions using a validation
logistic-regression probe. The test set is not consulted. Measured validation F1:

| Method | 8 dims | 12 dims | 16 dims |
|---|---:|---:|---:|
| PCA | 0.9161 | 0.9252 | **0.9295** |
| Mutual information | 0.8715 | 0.8734 | 0.9132 |

PCA-16 is the best overall, but the quantum statevector simulation is capped at eight qubits. The
best feasible configuration is PCA-8, which is therefore used for **every** model, including the
classical baseline. The cap costs about 1.3 validation F1 points relative to PCA-16 and is a
disclosed constraint, not a post-hoc choice.

The selected eight components are rescaled to [0, π] using training min/max. All models receive
the same transformed matrix.

### 3.4 Models and training budgets

**CNN+LSTM.** The eight-vector is reshaped into a single-channel sequence of length eight. This is
a pseudo-temporal representation: the order is the PCA component index, not packet timing. The
network is BatchNorm1d → Conv1d(1→32, kernel 3, padding 1) → ReLU → MaxPool1d(2) → Dropout(0.3) →
LSTM(input 32, hidden 32, one layer) → Linear(32→32) → ReLU → Dropout(0.3) → Linear(32→1). Training
uses class-weighted BCE, AdamW (lr 1e-3, weight decay 1e-4), batch size 256, at most 25 epochs,
early stopping with patience 6 on validation ROC-AUC, gradient clipping at 1.0, and best-epoch
checkpointing. Training size: 8,399 rows.

**QSVM.** Features are encoded with a `zz_feature_map` (8 qubits, 2 repetitions, linear
entanglement). The exact statevector fidelity kernel is

\[
K(x_i,x_j)=\left|\langle\phi(x_i)|\phi(x_j)\rangle\right|^2,
\]

computed as \(K=|SS^\dagger|^2\) from the matrix of statevectors \(S\). An `SVC` with a precomputed
kernel, \(C=1.0\), balanced class weights and Platt scaling is trained on a stratified 600-row
subset with 200 validation rows. The exact kernel is cross-checked against Qiskit's
`FidelityQuantumKernel` with `ComputeUncompute(StatevectorSampler, shots=100000)` on 16 samples:
maximum absolute difference 0.00171, tolerance 0.00632, passed.

**VQC.** The same ZZ feature map is followed by a `RealAmplitudes` ansatz (2 repetitions, linear
entanglement). The decision score is \(P(q_0=1)\), optimised with COBYLA (maxiter 150) on binary
cross-entropy with L2 regularisation 1e-4. The implementation precomputes feature states and
applies `Operator(ansatz).data`, which is mathematically equivalent to full-circuit simulation and
is verified against direct simulation in the test suite. Training size: 600 rows.

**Budget asymmetry.** The CNN sees 8,399 training rows; the quantum models see 600. The test set is
identical. The asymmetry is forced by exact statevector cost and is stated explicitly. It favours
the classical model in data volume, so a quantum win would have been a strong result; a classical
win must be read with this constraint in mind.

### 3.5 Thresholds and metrics

Thresholds are selected on validation F1 over a grid from 0.05 to 0.95 and then frozen. The
selected thresholds are CNN 0.35, QSVM 0.35 and VQC 0.50. Reported metrics are accuracy, precision,
recall, F1, ROC-AUC, FPR, FNR, specificity, MCC, balanced accuracy and Brier score, plus confusion
matrices and per-sample scores.

### 3.6 Statistical protocol

Pairwise McNemar tests use the exact binomial form when discordant pairs are fewer than 25 and the
chi-square form with continuity correction otherwise. Paired bootstrap (1,000 resamples) estimates
the difference in F1 at each model's validation-selected threshold and the difference in ROC-AUC.
Wilcoxon signed-rank is applied to per-fold F1 but is underpowered with three folds (minimum
two-sided p = 0.25) and is reported as supporting evidence only. No family-wise correction is
applied; p-values are descriptive on this benchmark.

---

## 4. Results

### 4.1 Main comparison

| Model | Encoding | Acc | Prec | Rec | F1 | ROC-AUC | FPR | FNR | MCC | Brier |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **CNN+LSTM** | none | **0.9567** | **0.9567** | 0.9499 | **0.9533** | 0.9891 | **0.0374** | 0.0501 | **0.9129** | **0.0308** |
| QSVM | angle | 0.9244 | 0.8774 | **0.9737** | 0.9231 | **0.9903** | 0.1185 | **0.0263** | 0.8537 | 0.0419 |
| VQC | angle | 0.7933 | 0.7223 | 0.9033 | 0.8028 | 0.8982 | 0.3025 | 0.0967 | 0.6077 | 0.1527 |

Confusion matrices at the selected thresholds:

| Model | Threshold | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|
| CNN+LSTM | 0.35 | 926 | 36 | 42 | 796 |
| QSVM | 0.35 | 848 | 114 | 22 | 816 |
| VQC | 0.50 | 671 | 291 | 81 | 757 |

The CNN+LSTM has the best F1, accuracy, MCC, Brier score and false-positive rate. Its F1 advantage
over the QSVM is 3.02 points; its false-positive count is 3.2× lower. The QSVM has the best recall
and misses only 22 attacks, but raises 114 false alarms. The VQC is weakest on every ranking
metric and operates at a 30.25% FPR.

### 4.2 Cross-validation and statistical tests

Three-fold cross-validation (preprocessing refit inside each fold):

| Model | Mean F1 | Std | 95% CI |
|---|---:|---:|---|
| CNN+LSTM | **0.9346** | 0.0135 | [0.9010, 0.9682] |
| QSVM | 0.9180 | 0.0161 | [0.8779, 0.9580] |
| VQC | 0.7452 | 0.0527 | [0.6144, 0.8761] |

Per-fold F1 values are CNN 0.9500/0.9293/0.9246, QSVM 0.9012/0.9333/0.9194 and VQC
0.6906/0.7494/0.7957. The ordering is stable.

McNemar:

| Comparison | Discordant | p-value | Cohen's h | Effect |
|---|---:|---:|---:|---|
| CNN vs QSVM | 146 | 2.389 × 10⁻⁶ | 0.138 | negligible |
| CNN vs VQC | 400 | 1.347 × 10⁻⁴⁸ | 0.524 | medium |
| QSVM vs VQC | 362 | 4.792 × 10⁻³⁵ | 0.387 | small |

Paired bootstrap:

| Comparison | Metric | Difference (A−B) | 95% CI | Crosses zero |
|---|---|---:|---|---|
| CNN vs QSVM | F1 at tuned thresholds | **0.0302** | [0.0166, 0.0434] | no |
| CNN vs QSVM | ROC-AUC | −0.0012 | [−0.0061, 0.0030] | **yes** |
| CNN vs VQC | F1 at tuned thresholds | 0.1505 | [0.1308, 0.1712] | no |
| CNN vs VQC | ROC-AUC | 0.0909 | [0.0768, 0.1055] | no |
| QSVM vs VQC | F1 at tuned thresholds | 0.1203 | [0.1026, 0.1384] | no |
| QSVM vs VQC | ROC-AUC | 0.0921 | [0.0792, 0.1074] | no |

The F1 gap between CNN and QSVM is small but resolved; the AUC gap is not. McNemar's p-value is
driven by 146 discordant pairs, but the effect size is negligible. The VQC is significantly worse.

### 4.3 Unseen attack-category generalisation

R2L and U2R are removed from training entirely. Models are retrained on Normal + DoS + Probe and
scored on a balanced holdout of 84 Normal and 84 held-out-category attacks (168 rows).

| Model | Threshold | Recall | Precision | F1 | ROC-AUC | FPR | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN+LSTM | 0.30 | 0.0476 | 0.6667 | 0.0889 | 0.7484 | 0.0238 | 0.0642 |
| QSVM | 0.50 | 0.0119 | 0.3333 | 0.0230 | 0.6559 | 0.0238 | −0.0449 |
| VQC | 0.45 | **0.7619** | **0.8205** | **0.7901** | **0.8707** | 0.1667 | **0.5968** |

Per-category detection rates are 3.8% (R2L) and 16.7% (U2R) for the CNN, 0.0% and 16.7% for the
QSVM, and 78.2% and 50.0% for the VQC. The VQC's advantage is large but comes with a 16.7% FPR on
168 rows and a single seed. It is suggestive, not conclusive, and it is **not** a zero-day claim.

### 4.4 External novel attack types

Novelty is defined against the full KDDTrain+ label set. The evaluation subset contains 17 novel
types, 3,750 novel attacks and 9,711 Normal rows (13,461 total); known attack types are excluded.
KDDTest+ also differs in feature distribution, so this is a joint novelty and covariate-shift test.

| Model | Threshold | Recall | Precision | F1 | ROC-AUC | FPR | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN+LSTM | 0.35 | 0.3461 | **0.6234** | 0.4451 | 0.6665 | **0.0807** | 0.3290 |
| QSVM | 0.35 | **0.6197** | 0.6996 | **0.6572** | **0.8917** | 0.1028 | **0.5375** |
| VQC | 0.50 | 0.5731 | 0.4741 | 0.5189 | 0.7801 | 0.2455 | 0.3107 |

The ordering on this subset differs from the primary common test. The QSVM is strongest by F1,
recall, ROC-AUC and MCC; the CNN is the most conservative detector, with the highest precision and
lowest FPR; the VQC is intermediate on F1 and ROC-AUC but has the highest FPR. The external subset is
normal-heavy (72.14% Normal), comes from a single file, and inherits the main validation-selected
thresholds. The QSVM's F1 advantage over the CNN is 0.2121 and its ROC-AUC advantage is 0.2252.
This is a scope-specific result, not a general quantum advantage. External F1 is not comparable to
common-test F1 because class composition and attack types differ.

### 4.5 Angle versus amplitude encoding

Both encodings are retrained under the same protocol; VQC runs receive an equal COBYLA budget.

| Model | Encoding | Qubits | Depth | Size | Parameters | F1 | ROC-AUC | Train (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| QSVM | angle | 8 | 31 | 74 | 8 | 0.9231 | **0.9903** | 1.97 |
| QSVM | amplitude | 3 | 1 | 1 | 0 | **0.9364** | 0.9783 | 0.31 |
| VQC | angle | 8 | 38 | 112 | 24 | 0.8028 | 0.8982 | 9.58 |
| VQC | amplitude | 3 | 8 | 14 | 9 | **0.8958** | **0.9609** | 0.77 |

Amplitude encoding uses three qubits instead of eight, trains faster and improves F1 for both
models. It costs the QSVM 1.2 ROC-AUC points and gains the VQC 6.3. However, basis-state amplitude
encoding with real non-negative amplitudes produces a squared-cosine fidelity kernel, a classical
polynomial kernel. The comparison is a resource/performance check, not evidence of quantum
advantage.

### 4.6 Latency and circuit resources

Single-sample end-to-end CPU latency (median of five repeats):

| Model | Preprocess (ms) | Reduction (ms) | Inference (ms) | Throughput (samples/s) | Slowdown |
|---|---:|---:|---:|---:|---:|
| CNN+LSTM | 0.00986 | 0.00084 | **0.9216** | **183,947.7** | 1.00× |
| QSVM | 0.00986 | 0.00084 | 3.9428 | 385.2 | 4.28× |
| VQC | 0.00986 | 0.00084 | 32.7295 | 332.0 | 35.51× |

| Model | Encoding | Qubits | Depth | Size | Parameters | 2Q gates |
|---|---|---:|---:|---:|---:|---:|
| QSVM | angle | 8 | 31 | 74 | 8 | 28 |
| VQC | angle | 8 | 38 | 112 | 24 | 42 |

Batch inference is much faster than single-sample inference because circuit and state-preparation
overhead is amortised. Absolute timings are CPU-only and load-dependent; an earlier staged run
measured a 6.83× QSVM slowdown versus 4.28× in the clean run. These are not production capacity
numbers, and hardware would add sampling, communication and error-correction costs.

### 4.7 Explainability

| Analysis | Top signals |
|---|---|
| Permutation importance (ROC-AUC drop) | PC1 0.3503, PC6 0.0400, PC2 0.0241, PC4 0.0136 |
| SHAP (mean absolute) | PC1 0.3556, PC2 0.0677, PC4 0.0640, PC5 0.0344 |
| PCA attribution | `dst_host_srv_serror_rate` 0.1222, `dst_host_serror_rate` 0.1221, `serror_rate` 0.1220 |
| QSVM sensitivity (δ=0.25) | PC3 0.1778, PC5 0.1690, PC2 0.1225, PC6 0.1173 |
| VQC sensitivity (δ=0.25) | PC3 0.2033, PC2 0.1610, PC1 0.1386, PC4 0.0250 |

The CNN concentrates on PC1 plus a few secondary components. PCA attribution maps these to
connection-error-rate features, plausible for DoS/Probe detection. Quantum sensitivity is a local
input-perturbation analysis, not a Shapley attribution. The VQC is numerically insensitive to
PC6–PC8 (~1e-16), which is a circuit/measurement property, not evidence that those components are
irrelevant.

### 4.8 Controlled distribution shift

CNN+LSTM only, on the common test set:

| Shift | Severity | F1 | ROC-AUC | KS mean |
|---|---:|---:|---:|---:|
| None | 0.0 | 0.9533 | 0.9891 | 0.0197 |
| Gaussian noise | 0.5 | 0.8984 | 0.9629 | 0.1371 |
| Gaussian noise | 1.0 | 0.7882 | 0.8755 | 0.2160 |
| Feature permutation | 0.1 | 0.9167 | 0.9778 | 0.0197 |
| Feature permutation | 0.25 | 0.4546 | 0.5613 | 0.0197 |
| Feature permutation | 0.5 | 0.4685 | 0.5803 | 0.0197 |

The model tolerates moderate Gaussian noise but collapses when 25% or more of features are
permuted. A five-window scan of KDDTest+ in file order gives F1 0.7308–0.7470 and ROC-AUC
0.8351–0.8457. NSL-KDD has no timestamps, so this is a simulated stress test, not concept-drift
detection. Only the classical model is evaluated; no comparison under drift is claimed.

### 4.9 Tiered alert triage

CNN+LSTM probabilities mapped to a five-level policy:

| Level | Count | Fraction | Action |
|---|---:|---:|---|
| NORMAL | 920 | 51.1% | ALLOW |
| MONITOR | 67 | 3.7% | LOG_AND_MONITOR |
| ALERT | 33 | 1.8% | NOTIFY_ANALYST |
| QUARANTINE | 36 | 2.0% | QUARANTINE_HOST |
| ESCALATE | 744 | 41.3% | BLOCK_AND_ESCALATE |

The high ESCALATE fraction reflects the benchmark's 46.6% attack rate and fixed conservative
bands. The engine is an assistive demonstration, not a calibrated production policy.

---

## 5. Discussion

### 5.1 Why the classical model wins here

The CNN+LSTM sees 14× more training data and can fit a flexible decision boundary in the reduced
space. The QSVM, despite seeing only 600 rows, is within 3.02 F1 points and has a marginally
higher ROC-AUC. That is a credible result for a kernel method on a small sample, but it is not an
advantage. The VQC is the weakest model: its optimiser did not converge, its ansatz is shallow and
its 30% FPR is operationally unacceptable.

The result should not be read as “quantum methods cannot work for IDS”. It says that, under an
8-qubit, 600-training-row, exact-simulation budget with a PCA-8 representation, the quantum models
do not beat a tuned classical baseline. Larger circuits, larger training sets, better ansätze and
hardware-aware training could change the picture; none of those was tested.

### 5.2 The ROC-AUC nuance

The QSVM's ROC-AUC is 0.9903 versus 0.9891 for the CNN, a difference of 0.0012 with a CI that
crosses zero. Reporting this as a quantum win would be unjustified. It is better read as a tie on
ranking quality, with the CNN better calibrated at the selected threshold and much better on FPR.

### 5.3 The unseen-category result

The VQC's 76.2% recall on held-out R2L/U2R is the most interesting positive result for a quantum
model. It is also the most fragile: 168 rows, one seed, a 16.7% FPR, and a threshold tuned on a
small held-out-category validation split. The CNN and QSVM collapse to near-zero recall because
their decision boundaries are tuned to the training categories. This is a real generalisation
phenomenon worth follow-up, not a deployment recommendation.

### 5.4 A scope-specific external novel-type result

On external novel types, the QSVM leads by F1 (0.6572 vs 0.5189 for the VQC and 0.4451 for the
CNN) and ROC-AUC (0.8917 vs 0.7801 and 0.6665). The CNN remains the most conservative detector, with
the lowest FPR (0.0807) and highest precision (0.6234). The QSVM's external lead is threshold-free
for AUC but threshold-dependent for F1, and the external subset is normal-heavy (72.14% Normal) and
drawn from one file. It is a genuine, scope-specific result, not a general quantum advantage. The
primary common-test result still favours the CNN on F1, and the QSVM's main AUC advantage is
unresolved. The external ordering is therefore best reported as a hypothesis about kernel ranking
under novel-type shift, not as a deployment recommendation.

### 5.5 Resource and simulation caveats

The quantum models are slower per sample in simulation, and hardware would add sampling and error
costs. Batching does amortise state preparation: QSVM batch inference is 2.60 ms/sample versus
3.94 ms single-sample, and VQC is 3.01 ms/sample versus 32.73 ms single-sample. Both remain far
slower than the CNN, whose batch throughput is 183,947.7 samples/s. Single-sample latency is 4.28×
the CNN for the QSVM and 35.51× for the VQC. These numbers are indicative, not a capacity plan.

Amplitude encoding is cheaper and produces higher F1 here, but it reduces to a classical polynomial
kernel for real non-negative amplitudes. It cannot be used as evidence of quantum advantage.

---

## 6. Limitations

1. **Quick profile only.** The full 125,973-row profile was not executed. Absolute metrics are
   not the best achievable on NSL-KDD.
2. **Training-budget asymmetry.** Quantum models see 600 rows versus 8,399 for the CNN. The
   comparison is fair in evaluation, not in data volume.
3. **Eight-qubit cap.** PCA-8 costs about 1.3 validation F1 points relative to PCA-16 and limits
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
8. **NSL-KDD is old.** No timestamps, synthetic attack mix, 1998-era traffic. Results do not
   transfer automatically to modern networks.
9. **“Unseen category” is not “zero-day”.** The held-out categories are known and present in the
   test set.
10. **Amplitude encoding is classically simulable.** Its empirical gains are not quantum
    advantage.
11. **Latency is load-dependent.** CPU-only medians of five repeats; earlier runs differ.
12. **Explainability is post hoc.** Permutation/SHAP/PCA attribution are correlational; quantum
    sensitivity is not a Shapley value.
13. **Drift is simulated.** File order is not chronological; Gaussian noise and feature permutation
    are stress tests.
14. **No production validation.** No adversarial, streaming, encrypted-traffic or live deployment
    evaluation was performed.
15. **External CIs not computed.** The external novel-type metrics are point estimates; the QSVM's
    external F1/AUC lead was not accompanied by a paired bootstrap or McNemar test on that subset.

---

## 7. Conclusion

We built and executed a reproducible, leakage-audited benchmark for classical versus quantum
intrusion detection on NSL-KDD. With identical eight-component inputs, identical splits and
identical test samples, the classical CNN+LSTM achieved F1 = 0.9533, the QSVM 0.9231 and the VQC
0.8028. The CNN's F1 advantage over the QSVM is 3.02 points with a bootstrap CI excluding zero;
the ROC-AUC difference is unresolved. No quantum advantage is supported on the primary common
test. On the secondary external novel-type subset the QSVM leads by F1 (0.6572 vs 0.4451) and
ROC-AUC (0.8917 vs 0.6665), a scope-specific result that does not overturn the primary ranking.

The contribution is methodological as much as empirical: a fully executable protocol that makes
the comparison auditable, reports negative results and scope-specific exceptions, and separates
simulation from hardware claims. Extending it to larger circuits, multi-seed runs, real hardware
and modern datasets is the natural next step.

---

## Reproducibility Statement

The reported results were produced by:

```bash
python run_all.py --mode quick --force
```

on Python 3.14.5, Windows 11, CPU only (Intel64 Family 6 Model 186, 12 logical cores, 15.72 GB
RAM), with NumPy 2.4.4, SciPy 1.17.1, pandas 3.0.1, scikit-learn 1.8.0, PyTorch 2.14.1, Qiskit
2.6.0, qiskit-aer 0.18.0 and qiskit-machine-learning 0.9.1. Seed 42. Preprocessing objects, model
binaries, metadata, environment capture, run manifest and per-stage logs are saved under
`data/processed/`, `models/`, `results/environment.json`, `results/run_manifest.json` and
`results/logs/`. The test suite (`python -m pytest -q`) reports 31 passing tests. The exact values
in this paper are read from `results/final_summary.json`; no number was typed from memory or
fabricated.

---

## References

1. Havlíček, V., Córcoles, A. D., Temme, K., Harrow, A. W., Kandala, A., Chow, J. M., & Gambetta,
   J. M. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature*, 567, 209–212.
2. Schuld, M., & Killoran, N. (2019). Quantum machine learning in feature Hilbert spaces.
   *Physical Review Letters*, 122(4), 040504.
3. Cerezo, M., Arrasmith, A., Babbush, R., Benjamin, S. C., Endo, S., Fujii, K., ... & Coles, P. J.
   (2021). Variational quantum algorithms. *Nature Reviews Physics*, 3, 625–644.
4. Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). A detailed analysis of the KDD
   CUP 99 data set. *IEEE Symposium on Computational Intelligence for Security and Defense
   Applications (CISDA)*, 1–6.
5. Qiskit contributors. Qiskit: An open-source framework for quantum computing (2024–2026).
6. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine
   Learning Research*, 12, 2825–2830.
7. Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning
   library. *Advances in Neural Information Processing Systems*, 32.
8. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions.
   *Advances in Neural Information Processing Systems*, 30.
9. Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification
   learning algorithms. *Neural Computation*, 10(7), 1895–1923.
10. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
11. Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*.
12. Powell, M. J. D. (1994). A direct search optimization method that models the objective and
    constraint functions by linear interpolation. *Advances in Optimization and Numerical
    Analysis*, 51–67.
