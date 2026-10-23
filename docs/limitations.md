# Limitations and Threats to Validity

The project reports a negative result for quantum advantage. This document records the limits of
that claim and the ways the study could mislead if read too broadly. None of these limits is
hidden in the results tables.

---

## 1. Only the quick profile was executed

The reported results use the quick profile:

- 12,000-row stratified subsample of KDDTrain+,
- 8,399 training rows,
- 600 quantum training rows,
- 1,800 common test rows,
- 3 CV folds.

The full profile in `config/config.yaml` (full 125,973-row pool, 3,000 quantum training rows,
2,500 test rows, 5 folds) is implemented but was **not run**. Full-profile numbers are not
reported and must not be inferred from the quick run. The quick profile is a legitimate
controlled experiment, but its absolute metrics are not the best achievable on NSL-KDD.

---

## 2. Training-budget asymmetry

The CNN+LSTM sees 8,399 training rows, while QSVM and VQC see 600. This is not a like-for-like
data budget. It is forced by the cost of exact statevector simulation: the QSVM kernel is
600×600 and each VQC objective evaluation touches 600 statevectors.

Implications:

- The comparison answers “what can a CPU-scale statevector experiment deliver?”, not “what is the
  best possible quantum model?”.
- The asymmetry favours the classical model in data volume. A quantum win would therefore have
  been a strong result; a classical win is expected and must be interpreted with the asymmetry in
  mind.
- A larger quantum training set could change the ranking, especially for the QSVM, whose
  decision boundary is data-hungry. This was not tested.

The test set is identical for every model, so the comparison is fair in evaluation even though it
is asymmetric in training data.

---

## 3. Eight-qubit cap and reduced feature space

Quantum circuits are capped at 8 qubits for tractability. The main comparison therefore uses
PCA-8 for **all** models, including the classical baseline.

- PCA-16 has better validation F1 (0.9295 vs 0.9161) and would likely improve the CNN.
- The cap costs roughly 1.3 validation F1 points relative to PCA-16.
- The 8-component representation may discard signal that either classical or quantum models
  could exploit at higher dimension.
- The cap is a simulator constraint, not a property of the intrusion-detection problem.

The 8/12/16 ablation is reported for a validation probe only; it is not a quantum experiment at
12–16 qubits.

---

## 4. Simulation only, no hardware noise

All quantum results come from **exact statevector simulation**:

- no gate errors,
- no decoherence,
- no readout noise,
- no finite-shot sampling in the reported metrics,
- no error mitigation,
- no hardware topology or connectivity constraints.

Real hardware would add sampling overhead and noise, and would require transpilation to a device
coupling map. The reported quantum metrics are therefore an upper bound relative to today's
noisy hardware, not a hardware performance claim. The QSVM kernel was cross-checked against a
finite-shot Qiskit reference only to validate correctness, not to model a device.

---

## 5. VQC did not converge

COBYLA used all 151 objective evaluations (150 iterations plus the initial point) without
satisfying its convergence criterion. Final training loss was 0.4585. The VQC result is therefore
a **budget-limited** result.

- A longer optimisation, a different ansatz, a different initial point or a gradient-based
  optimiser could improve it.
- The comparison does not establish that VQCs are intrinsically weak for IDS; it establishes that
  this VQC, with this budget and circuit, is weak.
- The VQC's external F1 of 0.5189 is intermediate, but its FPR of 0.2455 is the highest of the
  three models; this is consistent with an under-calibrated model, not a robust detector.

---

## 6. Statistical power and multiplicity

- The common test set has 1,800 rows. Confidence intervals for F1 are roughly ±0.01.
- McNemar is sensitive on paired data, but a significant p-value does not imply a large effect.
  Cohen's h for CNN vs QSVM is 0.138 (“negligible”).
- The ROC-AUC difference between CNN and QSVM is unresolved: 95% CI [−0.0061, 0.0030] crosses
  zero.
- Wilcoxon with 3 folds has a minimum attainable two-sided p-value of 0.25. It cannot detect
  significance at α = 0.05 even with large differences; it is reported as supporting evidence.
- No family-wise correction is applied across the set of experiments. Treat individual p-values
  as descriptive on this benchmark.
- The unseen-category holdout has only 168 rows and 84 attacks; its CIs are wide. The VQC's
  advantage there is suggestive, not established.

A single seed is used. The CNN training is stochastic at the batch level; the saved run is one
realisation. Cross-validation provides some stability evidence, but it is not a multi-seed study.
No seed-sensitivity analysis was performed.

---

## 7. NSL-KDD is old and not representative

NSL-KDD is a 2009 research benchmark derived from KDD Cup 99. Known limitations:

- traffic is from a 1998 simulated network environment,
- attack mix is synthetic and dated,
- no capture timestamps, so no real temporal modelling or drift analysis is possible,
- duplicate removal and difficulty scoring introduce their own biases,
- modern encrypted, cloud, and lateral-movement traffic is absent,
- label taxonomy is coarse and category boundaries are debatable.

The absolute performance numbers do not transfer to modern enterprise networks. The value of the
study is the controlled comparison protocol, not a claim about production detection quality.

---

## 8. “Unseen category” is not “zero-day”

The unseen-category experiment removes R2L and U2R from training. This measures generalisation to
a family the model never saw. It is **not** zero-day detection:

- the held-out categories are known and present in the test set,
- the model is still trained on Normal, DoS and Probe,
- the test distribution is fixed and offline,
- the holdout is small and class-balanced by construction,
- a true zero-day claim would require a genuinely novel exploit observed after deployment.

The phrase “zero-day” is avoided throughout the results and paper.

---

## 9. External novel-type test conflates novelty and covariate shift

KDDTest+ differs from KDDTrain+ in feature distribution as well as attack-type composition. The
external experiment therefore measures **novel-type generalisation under covariate shift**, not an
isolated novelty effect.

- Novelty is defined against the full KDDTrain+ label set (17 types), which is stable across
  profiles.
- The evaluation subset contains 3,750 novel attacks and 9,711 Normal rows; known attack types
  are excluded.
- Precision/recall depend on this composition, so external F1 is not directly comparable to the
  common-test F1 (which has a different attack rate and known types).
- On this subset the QSVM leads by F1 0.6572 and ROC-AUC 0.8917, while the CNN has the lowest FPR
  (0.0807). This ordering differs from the primary common test and is a scope-specific result, not a
  general quantum advantage.
- The external subset is normal-heavy (72.14% Normal), comes from one file, and has no external-set
  confidence intervals. External F1 is not directly comparable to common-test F1.

---

## 10. Amplitude encoding is classically simulable

With basis-state preparation and real non-negative amplitudes, the fidelity kernel is a
squared-cosine (polynomial) kernel. Amplitude encoding's better F1 and lower qubit count are
empirical findings about this implementation, not evidence of quantum advantage. The angle
encoding is retained as the primary configuration because it uses an explicit ZZ feature map.

---

## 11. Threshold selection and calibration

- Thresholds are tuned on validation F1, not on a deployment cost model.
- The cost-sensitive objective (missed attack = 10× false alarm) is implemented but not used in
  the reported run.
- Validation is small (1,801 rows; 200 for quantum models), so the selected threshold has
  uncertainty.
- The VQC threshold sits at 0.50, the middle of the grid, and its probabilities are not
  necessarily well calibrated.
- Test threshold curves are diagnostic only; they were not used to pick the operating point.
- A different threshold objective would change FPR/recall trade-offs and could change the
  perceived ranking on recall-heavy metrics.

The alert engine's bands are fixed policy choices, not calibrated probabilities. Its ESCALATE
fraction (41.3%) reflects the benchmark's attack rate and the bands, not a realistic alert volume.

---

## 12. Latency measurements are indicative

- CPU-only, single machine, no GPU.
- Five repeats, median reported.
- Absolute timings varied between runs (e.g. QSVM single-sample slowdown was 6.83× in an earlier
  staged run and 4.28× in the clean run). This is normal load-dependent variance.
- Batch and single-sample timings measure different things; they are labelled separately in the
  report.
- No hardware, network, queueing or sampling costs are included.
- The VQC's 35.5× slowdown is a simulation artefact in part; a hardware implementation would have
  a different cost structure, including shot sampling and communication.

Do not use these numbers as a production capacity plan.

---

## 13. Explainability is post hoc and correlational

- Permutation importance and SHAP describe the trained CNN's behaviour on this test set; they do
  not establish causal feature relevance.
- PCA attribution projects loadings back to original features; correlated raw features can share
  or split attribution.
- Quantum “sensitivity” is a local input perturbation of the encoded circuit. It is not SHAP, not
  a Shapley value and not a causal explanation.
- The VQC's near-zero sensitivity to PC6–PC8 is a property of the circuit/measurement, not
  evidence that those components are irrelevant to intrusion detection.
- Explanations are computed on a subset (100–500 rows), not the full test set.

---

## 14. Drift experiment is a stress test, not drift detection

- NSL-KDD has no timestamps.
- Gaussian noise and feature permutation are controlled corruptions, not real distribution
  change.
- The “window scan” follows file order in KDDTest+, which is not chronological.
- Only the CNN+LSTM is evaluated; no model comparison under drift is claimed.
- The experiment characterises failure envelopes, not the ability to detect drift online.

---

## 15. Class imbalance handling

- No SMOTE/undersampling; class-weighted BCE and threshold tuning are used.
- This avoids synthesising network flows, but it does not solve imbalance for the quantum models,
  whose training subsets are small.
- U2R has only six rows in the quick pool; category-level conclusions for U2R are especially
  fragile.
- The common test set has a 46.56% attack rate, far higher than most real networks. Precision,
  FPR and alert volumes would differ under realistic base rates.

---

## 16. Implementation and reproducibility limits

- TensorFlow was unavailable on Python 3.14, so the classical baseline uses PyTorch. Results are
  specific to this implementation and hyperparameter set.
- Qiskit 2.x API changes required function-form feature maps and ansatz; versions are pinned.
- The VQC fast path is mathematically equivalent to full-circuit simulation and is tested, but it
  relies on `Operator(ansatz).data`; a future Qiskit change could alter behaviour.
- Absolute timings and last-decimal metrics can vary with CPU, BLAS and library versions.
- The run manifest and logs record what executed, but the repository is not a container image.
- The project has no git history in the delivered workspace; provenance is the manifest,
  environment capture and logs.

---

## 17. No adversarial, streaming or production evaluation

The project does not test:

- adversarial evasion or poisoning,
- encrypted traffic,
- packet-level parsing or flow reconstruction,
- online/streaming inference,
- concept drift with real timestamps,
- alert fatigue or analyst workflow,
- regulatory or privacy constraints,
- deployment on constrained network hardware.

It is a research benchmark and teaching artefact. The dashboard and alert engine are
demonstrations, not production controls.

---

## 18. What would strengthen the study

1. Run the full profile and report both quick and full results.
2. Repeat the main comparison across at least 5–10 seeds and report mean ± std.
3. Increase the quantum training set and test whether the QSVM closes the F1 gap.
4. Run on real quantum hardware or a calibrated noise model to quantify the simulation-to-hardware
   gap.
5. Evaluate 12–16 qubit circuits with a tensor-network or GPU simulator.
6. Pre-register the threshold objective and primary metric before seeing test results.
7. Add a proper cost-sensitive analysis with realistic base rates.
8. Test modern datasets (CIC-IDS2017/2018, UNSW-NB15) and encrypted traffic.
9. Evaluate calibration (reliability diagrams, ECE) separately from ranking.
10. Add an adversarial robustness evaluation.

These are explicitly out of scope for the delivered run.
