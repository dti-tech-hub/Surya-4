# Quantum-Enhanced Intrusion Detection: A Fair Comparison of a Classical CNN+LSTM Baseline with Quantum SVM and Variational Quantum Classifier on NSL-KDD

> **Generated from executed experiments.** Every numeric value in this draft was read from `results/` after the pipeline completed. Values are not placeholders. Re-running `python run_all.py --mode full` will regenerate this document with the full-mode numbers.

## Abstract

Network intrusion detection systems must separate malicious traffic from benign traffic under severe class imbalance and evolving attack behaviour. Quantum machine learning is frequently proposed for this task, but comparisons are rarely conducted on identical reduced features, identical splits and identical evaluation protocols. We present a controlled benchmark on NSL-KDD comparing three classifiers: a classical CNN+LSTM, a Quantum Support Vector Machine (QSVM) using a fidelity quantum kernel, and a Variational Quantum Classifier (VQC). All models consume the same 8-component PCA representation, are trained with leakage-safe pipelines fitted only on the training split, and are scored on the same held-out test set. The classical CNN+LSTM reached F1 = 0.9533 and ROC-AUC = 0.9891; the best quantum model reached F1 = 0.9231 and ROC-AUC = 0.9903. The quantum models did not outperform the classical baseline. We report unseen attack-category generalisation, latency, quantum resource usage and threshold-dependent operating points, and we explicitly state the limits of simulation-based evidence.

**Keywords:** intrusion detection, NSL-KDD, quantum machine learning, quantum kernel, variational quantum classifier, CNN-LSTM, reproducibility

## 1. Introduction

Intrusion detection remains a core defensive control, yet signature-based systems struggle with novel behaviour and classical deep models can be expensive to tune and explain. Quantum classifiers promise richer feature spaces through high-dimensional state representations. That promise is frequently evaluated with inconsistent baselines, different feature subsets, or test-set tuning. This work asks a narrower and testable question: *given identical reduced inputs and an identical evaluation protocol, does a quantum classifier improve intrusion detection over a properly tuned classical CNN+LSTM?* The answer reported here is derived only from measured results.

### Contributions

1. A leakage-audited pipeline in which imputation, one-hot encoding, scaling, PCA/mutual-information selection and quantum angle rescaling are fitted exclusively on training data.
2. A fair three-way comparison on the same reduced representation and the same test samples, including validation-only threshold selection.
3. An unseen attack-category experiment in which R2L and U2R are removed from training entirely, plus an external novel-type test on KDDTest+.
4. A systems analysis covering latency, throughput, quantum resources, controlled distribution shift and explainability.
5. Explicit reporting of negative or inconclusive results.

## 2. Related work

Quantum kernel methods estimate similarity in a state space induced by a feature map and are attractive because they can be analysed with the machinery of kernel methods. Variational quantum classifiers instead train a parameterised circuit with a classical optimiser. Prior IDS studies report strong quantum results, but often on small subsamples, with classical baselines that are not tuned to convergence, or without paired statistical testing. Our protocol addresses those three weaknesses directly. NSL-KDD remains a standard benchmark despite its age, which makes it useful for controlled comparison rather than for claims about modern networks.

## 3. Dataset

NSL-KDD provides 125,973 labelled records in KDDTrain+ and 22,544 in KDDTest+. Each record has 41 features (3 categorical: protocol_type, service, flag; 38 numeric) plus a label and a difficulty score. The training pool contains 23 distinct labels; KDDTest+ contains 38, of which 17 attack types never appear in KDDTrain+.

| Split | Rows | Attack rate |
|---|---|---|
| Train (70%) | 8,399 | 46.54% |
| Validation | 1,801 | 46.53% |
| Test | 1,800 | 46.56% |

Attack categories follow the NSL-KDD taxonomy (Normal, DoS, Probe, R2L, U2R). The official KDDTest+ file is used only as an external, never-trained-on distribution-shift set.

## 4. Methodology

### 4.1 Preprocessing

Constant columns are detected on the training split and dropped. Numeric features are median-imputed (train medians) and standardised; categorical features are one-hot encoded with an unknown-level-safe vocabulary learned on the training split. Infinite values are replaced before imputation. No transformation is fitted on validation or test data.

### 4.2 Dimensionality reduction

Two families were evaluated on validation data only: PCA and mutual-information ranking at 8, 12 and 16 dimensions. The comparative study uses **PCA with 8 components**, selected by validation F1 subject to the simulator budget of 8 qubits. The full 8/12/16 ablation is reported in `results/tables/feature_reduction_ablation.csv`. The selected reduced features are then min–max rescaled to [0, π] using training statistics, and this identical matrix is supplied to all three models.

### 4.3 Classical baseline

The CNN+LSTM receives the reduced vector as a single-channel sequence of length equal to the number of components. This is a *pseudo-temporal* reshaping, not real packet timing: the order is the component index. The network is BatchNorm1d → Conv1d → ReLU → MaxPool → Dropout → LSTM → Dense → logit, trained with AdamW, class-weighted BCE, early stopping on validation ROC-AUC and best-epoch checkpointing.

### 4.4 QSVM

Reduced features are encoded with a ZZ feature map (angle encoding). The fidelity kernel K(x, y) = |⟨φ(x)|φ(y)⟩|² is computed exactly from statevectors produced by Qiskit. A soft-margin SVM with Platt scaling is trained on the precomputed kernel. The exact kernel is cross-checked against Qiskit's circuit-based FidelityQuantumKernel/ComputeUncompute estimator; the comparison tolerance is derived from the finite-shot binomial noise of that reference.

### 4.5 VQC

The same feature map is followed by a RealAmplitudes ansatz. The decision is P(qubit 0 = |1⟩), optimised with COBYLA (SPSA is supported) on binary cross-entropy. Training uses the identity |ψ(x;θ)⟩ = U_ansatz(θ)|φ(x)⟩ to apply one ansatz unitary to a precomputed batch of feature states; this is mathematically identical to full-circuit simulation and is verified in the test suite against direct simulation.

### 4.6 Evaluation protocol

All models are scored on the identical common test subset. The decision threshold is selected on validation data only, using either F1 or a cost-sensitive objective (a missed attack costs 10× a false alarm). Reported metrics are accuracy, precision, recall, F1, ROC-AUC, FPR, FNR, specificity, MCC and Brier score. Pairwise differences are tested with McNemar's exact/chi-square test on paired predictions, paired bootstrap confidence intervals and, across folds, the Wilcoxon signed-rank test.

## 5. Results

### 5.1 Main comparison

| Model | Encoding | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR | Train (s) | Inference (ms/sample) |
|---|---|---|---|---|---|---|---|---|---|---|
| CNN+LSTM | none | 0.9567 | 0.9567 | 0.9499 | 0.9533 | 0.9891 | 0.0374 | 0.0501 | 9.56 | 0.0073 |
| QSVM | angle | 0.9244 | 0.8774 | 0.9737 | 0.9231 | 0.9903 | 0.1185 | 0.0263 | 1.72 | 2.7440 |
| VQC | angle | 0.7933 | 0.7223 | 0.9033 | 0.8028 | 0.8982 | 0.3025 | 0.0967 | 8.61 | 2.8822 |

### 5.2 Cross-validation

| Model | Folds | Mean F1 | Std | 95% CI |
|---|---|---|---|---|
| CNN+LSTM | 3 | 0.9346 | 0.0135 | [0.9010, 0.9682] |
| QSVM | 3 | 0.9180 | 0.0161 | [0.8779, 0.9580] |
| VQC | 3 | 0.7452 | 0.0527 | [0.6144, 0.8761] |

### 5.3 Statistical tests

| Comparison | Discordant pairs | McNemar p | Significant | Cohen's h | Effect |
|---|---|---|---|---|---|
| cnn_lstm vs qsvm | 146 | 0.000002 | True | 0.138 | negligible |
| cnn_lstm vs vqc | 400 | 0.000000 | True | 0.524 | medium |
| qsvm vs vqc | 362 | 0.000000 | True | 0.387 | small |

### 5.4 Unseen attack-category generalisation

| Model | Recall | Precision | F1 | ROC-AUC | Unseen-attack recall |
|---|---|---|---|---|---|
| CNN+LSTM | 0.0476 | 0.6667 | 0.0889 | 0.7484 | 0.0476 |
| QSVM | 0.0119 | 0.3333 | 0.0230 | 0.6559 | 0.0119 |
| VQC | 0.7619 | 0.8205 | 0.7901 | 0.8707 | 0.7619 |

### 5.5 External novel attack types

| Model | Novel types | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|---|
| CNN+LSTM | 17 | 0.3461 | 0.6234 | 0.4451 | 0.6665 |
| QSVM | 17 | 0.6197 | 0.6996 | 0.6572 | 0.8917 |
| VQC | 17 | 0.5731 | 0.4741 | 0.5189 | 0.7801 |

### 5.6 Encoding comparison

| Model | Encoding | Qubits | Depth | F1 | ROC-AUC |
|---|---|---|---|---|---|
| QSVM | angle | 8 | 31 | 0.9231 | 0.9903 |
| QSVM | amplitude | 3 | 1 | 0.9364 | 0.9783 |
| VQC | angle | 8 | 38 | 0.8028 | 0.8982 |
| VQC | amplitude | 3 | 8 | 0.8958 | 0.9609 |

### 5.7 Latency and quantum resources

| Model | Inference (ms/sample) | Throughput (samples/s) | Slowdown vs classical |
|---|---|---|---|
| CNN+LSTM | 0.9216 | 183947.7 | 1.00× |
| QSVM | 3.9428 | 385.2 | 4.28× |
| VQC | 32.7295 | 332.0 | 35.51× |

## 6. Discussion

The classical CNN+LSTM outperforms the best quantum model (QSVM) by 3.02 F1 points. The quantum model's ROC-AUC is 0.0012 higher (0.9903 vs 0.9891). McNemar p = 2.389e-06. The quantum models did not provide an advantage on the primary common test under this experimental design.

The classical baseline benefits from being trained on the full training split, whereas the quantum models are trained on a stratified subset whose size is bounded by simulation cost. This is a property of the experiment, not a hidden advantage: every model is scored on the same test samples, and the asymmetry is stated explicitly. It also means the comparison answers a practical question — what a student-scale quantum simulation can deliver — rather than an idealised hardware question.

On the secondary external novel-type subset, QSVM leads by F1 0.6572 versus 0.4451 for the CNN+LSTM, with ROC-AUC 0.8917 versus 0.6665. This is a scope-specific result on one normal-heavy external file, not a general quantum advantage.

Latency is dominated by quantum state preparation and kernel evaluation. Because every quantum prediction requires fresh circuit simulation, the quantum models are not competitive for real-time inline detection in this form. A hardware deployment would face additional sampling, queueing and error-correction costs.

## 7. Limitations

- All quantum experiments use Qiskit's exact statevector simulator. They are simulation-based and do not reflect quantum-hardware noise, decoherence or gate errors.
- The quantum feature dimension is capped at 8 qubits for tractability. Larger circuits (12–16 qubits) are reported only for the classical probe ablation.
- The quantum models are trained on a smaller stratified subset than the classical model because circuit evaluation cost grows with training-set size. The test set is identical for every model.
- NSL-KDD is a 2009 benchmark with known limitations (age, synthetic attack mix, no timestamps). Results do not transfer automatically to modern enterprise traffic.
- Unseen attack categories are measured by holding out R2L and U2R. This is unseen-category generalisation, not a zero-day claim.
- Concept drift is studied with controlled covariate-shift injections and a simulated file-order scan. NSL-KDD has no capture timestamps, so no real temporal drift detection is claimed.
- The external novel-type subset is one normal-heavy KDDTest+ file with strong covariate shift; its metrics are point estimates without external-set confidence intervals.
- Amplitude encoding reduces to a squared-cosine fidelity kernel, which is a classical polynomial kernel. The encoding comparison is therefore an empirical check, not evidence of quantum advantage.
- McNemar and Wilcoxon tests establish whether differences are unlikely under a null of equal performance on this sample; they do not establish a general quantum advantage.

## 8. Conclusion

We built a reproducible, leakage-audited benchmark for quantum versus classical intrusion detection and executed it end to end. On NSL-KDD, with 8 reduced features, identical splits and identical test samples, the classical CNN+LSTM remained the strongest model on the primary common test. The contribution is methodological as much as empirical: an honest, fully executable protocol that others can extend to larger circuits or real quantum hardware without changing the evaluation contract.

## References

1. Tavallaee, M., Bagheri, E., Lu, W., Ghorbani, A. A. (2009). A detailed analysis of the KDD CUP 99 data set. *IEEE CISDA*.
2. Havlíček, V. et al. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature* 567, 209–212.
3. Schuld, M., Killoran, N. (2019). Quantum machine learning in feature Hilbert spaces. *Physical Review Letters* 122, 040504.
4. Cerezo, M. et al. (2021). Variational quantum algorithms. *Nature Reviews Physics* 3, 625–644.
5. Qiskit contributors. Qiskit: An open-source framework for quantum computing (2024–2026).
6. Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation* 10(7), 1895–1923.
