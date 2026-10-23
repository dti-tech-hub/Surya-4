# Quantum-Enhanced Intrusion Detection System — Final Summary

**Mode:** `quick` · **Seed:** 42 · **Dataset:** nsl-kdd

## 1. Dataset and preprocessing

| Item | Value |
|---|---|
| Raw training pool | 125,973 rows |
| External hold-out (KDDTest+) | 22,544 rows |
| Input features | 41 |
| Reduction | PCA → 8 components |
| Leakage audit | PASSED |
| Train size | 8,399 (attack rate 46.54%) |
| Validation size | 1,801 (attack rate 46.53%) |
| Test size | 1,800 (attack rate 46.56%) |

## 2. Model comparison (common test set)

| Model | Encoding | Acc | Prec | Rec | F1 | ROC-AUC | FPR | FNR | Train (s) | Batch infer (ms/sample) | Qubits | Depth |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN+LSTM | none | 0.9567 | 0.9567 | 0.9499 | 0.9533 | 0.9891 | 0.0374 | 0.0501 | 9.56 | 0.0073 | 0 | 0 |
| QSVM | angle | 0.9244 | 0.8774 | 0.9737 | 0.9231 | 0.9903 | 0.1185 | 0.0263 | 1.72 | 2.7440 | 8 | 31 |
| VQC | angle | 0.7933 | 0.7223 | 0.9033 | 0.8028 | 0.8982 | 0.3025 | 0.0967 | 8.61 | 2.8822 | 8 | 38 |

## 3. Main finding

The classical CNN+LSTM outperforms the best quantum model (QSVM) by 3.02 F1 points. The quantum model's ROC-AUC is 0.0012 higher (0.9903 vs 0.9891). McNemar p = 2.389e-06. The quantum models did not provide an advantage on the primary common test under this experimental design.

Additional observations:

- Single-sample QSVM inference is 4.3x slower than the CNN+LSTM in simulation.
- Unseen-category recall on a balanced Normal/attack hold-out (R2L+U2R withheld; per-model validation-selected threshold): VQC 0.7619, CNN+LSTM 0.0476, QSVM 0.0119
- External novel-type F1: QSVM 0.6572, VQC 0.5189, CNN+LSTM 0.4451; best is QSVM. This is a scope-specific external-subset result, not a general advantage over the primary common test.
- 3-fold CV F1: CNN+LSTM 0.9346 ± 0.0135, QSVM 0.9180 ± 0.0161, VQC 0.7452 ± 0.0527
- All quantum results are simulation-based (Qiskit Statevector) and shot-noise free; they do not represent quantum-hardware performance.

## 4. Cross-validation

| Model | Folds | Mean F1 | Std | 95% CI |
|---|---|---|---|---|
| CNN+LSTM | 3 | 0.9346 | 0.0135 | [0.9010, 0.9682] |
| QSVM | 3 | 0.9180 | 0.0161 | [0.8779, 0.9580] |
| VQC | 3 | 0.7452 | 0.0527 | [0.6144, 0.8761] |

## 5. Statistical significance

| Comparison | Discordant | McNemar p | Significant | Cohen's h | Effect |
|---|---|---|---|---|---|
| cnn_lstm vs qsvm | 146 | 0.000002 | True | 0.138 | negligible |
| cnn_lstm vs vqc | 400 | 0.000000 | True | 0.524 | medium |
| qsvm vs vqc | 362 | 0.000000 | True | 0.387 | small |

Paired bootstrap (1,000 resamples; F1 at each model's validation-selected threshold):

| Comparison | Metric | Difference | 95% CI | Crosses zero |
|---|---|---|---|---|
| cnn_lstm vs qsvm | f1@validation_threshold | 0.0302 | [0.0166, 0.0434] | False |
| cnn_lstm vs qsvm | roc_auc | -0.0012 | [-0.0061, 0.0030] | True |
| cnn_lstm vs vqc | f1@validation_threshold | 0.1505 | [0.1308, 0.1712] | False |
| cnn_lstm vs vqc | roc_auc | 0.0909 | [0.0768, 0.1055] | False |
| qsvm vs vqc | f1@validation_threshold | 0.1203 | [0.1026, 0.1384] | False |
| qsvm vs vqc | roc_auc | 0.0921 | [0.0792, 0.1074] | False |

Wilcoxon signed-rank across 3 CV folds (supporting evidence only; the attainable two-sided p-value is bounded by the fold count):

| Comparison | Mean F1 (A) | Mean F1 (B) | p-value |
|---|---|---|---|
| cnn_lstm vs qsvm | 0.9346 | 0.9180 | 0.500 |
| cnn_lstm vs vqc | 0.9346 | 0.7452 | 0.250 |
| qsvm vs vqc | 0.9180 | 0.7452 | 0.250 |

## 6. Unseen attack-category generalisation

| Model | Encoding | Recall | Precision | F1 | ROC-AUC | FPR | Unseen recall |
|---|---|---|---|---|---|---|---|
| CNN+LSTM | none | 0.0476 | 0.6667 | 0.0889 | 0.7484 | 0.0238 | 0.0476 |
| QSVM | angle | 0.0119 | 0.3333 | 0.0230 | 0.6559 | 0.0238 | 0.0119 |
| VQC | angle | 0.7619 | 0.8205 | 0.7901 | 0.8707 | 0.1667 | 0.7619 |

Held-out categories: **R2L, U2R**. Training categories: DoS, Normal, Probe. Test categories: Normal, R2L, U2R.


### External novel attack types (KDDTest+)

| Model | Novel types | Novel / normal samples | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| CNN+LSTM | 17 | 3750 / 9711 | 0.3461 | 0.6234 | 0.4451 | 0.6665 |
| QSVM | 17 | 3750 / 9711 | 0.6197 | 0.6996 | 0.6572 | 0.8917 |
| VQC | 17 | 3750 / 9711 | 0.5731 | 0.4741 | 0.5189 | 0.7801 |

Novel attack types: apache2, httptunnel, mailbomb, mscan, named, processtable, ps, saint, sendmail, snmpgetattack, snmpguess, sqlattack, udpstorm, worm, xlock, xsnoop, xterm.

## 7. Encoding comparison

| Model | Encoding | Qubits | Depth | F1 | ROC-AUC | Train (s) |
|---|---|---|---|---|---|---|
| QSVM | angle | 8 | 31 | 0.9231 | 0.9903 | 1.97 |
| QSVM | amplitude | 3 | 1 | 0.9364 | 0.9783 | 0.31 |
| VQC | angle | 8 | 38 | 0.8028 | 0.8982 | 9.58 |
| VQC | amplitude | 3 | 8 | 0.8958 | 0.9609 | 0.77 |

## 8. Latency

| Model | Preprocess (ms) | Reduction (ms) | Single-sample inference (ms) | Throughput (samples/s) | Slowdown vs classical |
|---|---|---|---|---|---|
| CNN+LSTM | 0.00986 | 0.00084 | 0.9216 | 183947.7 | 1.00x |
| QSVM | 0.00986 | 0.00084 | 3.9428 | 385.2 | 4.28x |
| VQC | 0.00986 | 0.00084 | 32.7295 | 332.0 | 35.51x |

Single-sample end-to-end latency, median of 5 repeats on CPU. Throughput uses batch size 256. Timings are hardware-dependent; batch inference in Section 2 is not directly comparable.


## 9. Quantum resources (simulation)

| Model | Encoding | Qubits | Depth | Size | Parameters | 2Q gates | Backend |
|---|---|---|---|---|---|---|---|
| QSVM | angle | 8 | 31 | 74 | 8 | 28 | qiskit.quantum_info.Statevector (exact fidelity kernel) |
| VQC | angle | 8 | 38 | 112 | 24 | 42 | qiskit.quantum_info.Statevector (analytic probabilities) |

## 10. Distribution shift

| Shift | Model | Baseline F1 | Worst F1 | ΔF1 | Worst KS |
|---|---|---|---|---|---|
| feature_permutation | cnn_lstm | 0.9533 | 0.4685 | 0.4848 | 0.020 |
| gaussian_noise | cnn_lstm | 0.9533 | 0.7882 | 0.1651 | 0.216 |

## 11. Limitations

- All quantum experiments use Qiskit's exact statevector simulator. They are simulation-based and do not reflect quantum-hardware noise, decoherence or gate errors.
- The quantum feature dimension is capped at 8 qubits for tractability. Larger circuits (12–16 qubits) are reported only for the classical probe ablation.
- The quantum models are trained on a smaller stratified subset than the classical model because circuit evaluation cost grows with training-set size. The test set is identical for every model.
- NSL-KDD is a 2009 benchmark with known limitations (age, synthetic attack mix, no timestamps). Results do not transfer automatically to modern enterprise traffic.
- Unseen attack categories are measured by holding out R2L and U2R. This is unseen-category generalisation, not a zero-day claim.
- Concept drift is studied with controlled covariate-shift injections and a simulated file-order scan. NSL-KDD has no capture timestamps, so no real temporal drift detection is claimed.
- The external novel-type subset is one normal-heavy KDDTest+ file with strong covariate shift; its metrics are point estimates without external-set confidence intervals.
- Amplitude encoding reduces to a squared-cosine fidelity kernel, which is a classical polynomial kernel. The encoding comparison is therefore an empirical check, not evidence of quantum advantage.
- McNemar and Wilcoxon tests establish whether differences are unlikely under a null of equal performance on this sample; they do not establish a general quantum advantage.

## 12. Conclusion

This study compared a properly tuned classical CNN+LSTM with two quantum classifiers on an identical, leakage-controlled, dimensionality-reduced NSL-KDD benchmark. The classical CNN+LSTM outperforms the best quantum model (QSVM) by 3.02 F1 points. The quantum model's ROC-AUC is 0.0012 higher (0.9903 vs 0.9891). McNemar p = 2.389e-06. The quantum models did not provide an advantage on the primary common test under this experimental design. All quantum results were produced with exact statevector simulation; no quantum hardware was used and no production deployment was attempted.

## 13. Reproducibility

```
Python: 3.14.5
Platform: Windows 11
CPU: Intel64 Family 6 Model 186 Stepping 3, GenuineIntel
RAM: 15.72 GB
Packages: numpy==2.4.4, scipy==1.17.1, pandas==3.0.1, scikit-learn==1.8.0, matplotlib==3.10.8, seaborn==0.13.2, torch==2.14.1, torchvision==0.29.1, qiskit==2.6.0, qiskit-aer==0.18.0, qiskit-machine-learning==0.9.1, shap==0.53.0, streamlit==1.65.0, pyarrow==25.0.1, joblib==1.5.3, pytest==9.1.1, PyYAML==6.0.3, tabulate==0.10.0
```
