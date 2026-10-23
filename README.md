# QE-IDS — Quantum-Enhanced Intrusion Detection System

A reproducible, leakage-audited comparison of a classical **CNN+LSTM** against a **Quantum Support
Vector Machine (QSVM)** and a **Variational Quantum Classifier (VQC)** for binary network intrusion
detection on **NSL-KDD**.

The project asks one narrow question:

> Given identical reduced inputs, identical splits, identical test samples and an identical
> evaluation protocol, does a quantum classifier improve intrusion detection over a properly tuned
> classical baseline?

**Answer on the primary common test: no — the classical CNN+LSTM is the strongest model.** On the
secondary external novel-type subset the QSVM leads by F1 and ROC-AUC, but that is a scope-specific
result, not a general quantum-advantage claim.

All quantum results come from **exact Qiskit statevector simulation**, not quantum hardware. Only
the **quick profile** was executed end to end; no full-profile numbers are reported.

---

## What this project is

This is a research and teaching artefact built as a B.Tech final-year project. It provides:

- a leakage-audited data pipeline for NSL-KDD;
- a shared PCA-8 representation used by every model;
- a classical CNN+LSTM baseline and two quantum classifiers;
- a common test set with validation-only threshold selection;
- paired statistical tests, cross-validation, calibration and confusion matrices;
- unseen-category and external novel-type generalisation experiments;
- encoding, latency, circuit-resource, explainability and controlled-shift studies;
- a tiered alert-triage demonstration and a Streamlit dashboard;
- a full paper sourcebook with measured values, tables, figures, references and claim guardrails.

The important design choice is that the comparison is **fair in evaluation**: every model is scored
on exactly the same test rows, with the same features, metrics and preprocessing. It is **not fair in
training data volume**: the CNN sees 8,399 rows while each quantum model sees 600 rows. That asymmetry
is stated explicitly and treated as a limitation, not hidden.

---

## Why this project exists

Quantum machine-learning papers often report promising intrusion-detection results, but comparisons
are difficult to reproduce when classical baselines, preprocessing, splits or evaluation protocols
differ. This project was built to make the comparison auditable:

1. **Leakage-safe preprocessing.** Imputation, one-hot encoding, scaling, PCA and quantum rescaling
   are fitted on training data only. Eight automated leakage checks pass.
2. **One shared representation.** All models consume the same eight PCA components, rescaled to
   `[0, π]` using training statistics.
3. **One common test set.** Every model is scored on the same 1,800 rows; thresholds are selected on
   validation data only.
4. **Honest reporting.** The primary result is negative. The external novel-type result is positive
   for the QSVM but is reported with its scope and uncertainty attached.
5. **Full reproducibility.** One YAML config, fixed seed, saved artifacts, environment capture, run
   manifest and per-stage logs.

---

## Key results

### Primary common test — 1,800 rows

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| **CNN+LSTM** | **0.9567** | **0.9567** | 0.9499 | **0.9533** | 0.9891 | **0.0374** | 0.0501 |
| QSVM | 0.9244 | 0.8774 | **0.9737** | 0.9231 | **0.9903** | 0.1185 | **0.0263** |
| VQC | 0.7933 | 0.7223 | 0.9033 | 0.8028 | 0.8982 | 0.3025 | 0.0967 |

- CNN+LSTM − QSVM F1 difference: **0.0302** (paired-bootstrap 95% CI **0.0166–0.0434**).
- CNN+LSTM − QSVM ROC-AUC difference: **−0.0012** (95% CI −0.0061 to +0.0030; crosses zero).
- McNemar test, CNN+LSTM vs QSVM: **p = 2.389 × 10⁻⁶**, Cohen's h = **0.138** (negligible effect).
- The QSVM has the best recall and lowest FNR, but raises **114 false positives** versus **36** for
  the CNN.
- 3-fold CV F1: CNN **0.9346 ± 0.0135**, QSVM **0.9180 ± 0.0161**, VQC **0.7452 ± 0.0527**.

### Secondary external novel-type test — 13,461 rows

Novelty is defined against the **full KDDTrain+ label set**. The subset contains 17 attack types
absent from training, with 3,750 novel attacks and 9,711 Normal rows. The already-trained main models
are scored with the main preprocessing and their original validation-selected thresholds.

| Model | Threshold | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|
| CNN+LSTM | 0.35 | 0.4451 | 0.6665 | **0.0807** |
| QSVM | 0.35 | **0.6572** | **0.8917** | 0.1028 |
| VQC | 0.50 | 0.5189 | 0.7801 | 0.2455 |

This is a **scope-specific result on one normal-heavy external file**, not a general quantum
advantage. The QSVM's F1 lead over the CNN is 0.2121 and its ROC-AUC lead is 0.2252. The CNN remains
the most conservative detector with the lowest FPR. No external-set confidence intervals or
multi-seed repeats were computed.

### Unseen attack-category holdout — 168 rows

R2L and U2R are removed from training entirely. The VQC reaches **0.7619 recall** on the held-out
categories versus **0.0476** for the CNN and **0.0119** for the QSVM. This is a small, single-seed
holdout and is **not** a zero-day detection claim.

---

## What we use and why

| Technology | Why it is used |
|---|---|
| **Python 3.14.5** | Runtime used for the reported quick-profile results. |
| **PyTorch 2.14.1** | CNN+LSTM implementation; TensorFlow was not available for Python 3.14 at setup time. |
| **Qiskit 2.6.0 + Aer 0.18.0** | Exact statevector simulation for the QSVM fidelity kernel and VQC. No hardware noise or finite-shot sampling in the reported metrics. |
| **Qiskit Machine Learning 0.9.1** | `FidelityQuantumKernel` cross-check against the custom kernel implementation. |
| **scikit-learn 1.8.0** | PCA, mutual-information reduction, SVM, metrics, cross-validation and calibration utilities. |
| **NumPy / pandas / PyArrow** | Data loading, split handling, tables and Parquet artifacts. |
| **Matplotlib / seaborn** | Result figures, ROC/PR curves, confusion matrices and diagnostics. |
| **SHAP 0.53.0** | Post-hoc explanation of the CNN+LSTM. |
| **Streamlit 1.65.0** | Interactive dashboard over the generated results. |
| **pytest 9.1.1** | Unit and contract tests; **31 tests pass**. |
| **Joblib / PyYAML** | Model serialization and one central experiment configuration. |

### Why PCA-8?

PCA and mutual-information reduction are compared at 8, 12 and 16 dimensions on validation data.
**PCA-8** is used for the main comparison because it is the largest shared representation that fits
the eight-qubit simulation cap while keeping the comparison identical across models. It retains
**66.6556%** of total variance. Using PCA-16 for the CNN only would break the identical-input
contract.

### Why exact statevector simulation?

The project compares algorithms rather than hardware. Exact simulation removes shot noise, queueing,
decoherence and readout error, so differences between models reflect the models and protocol rather
than a particular device. The reported quantum metrics are therefore a **shot-noise-free upper bound**
relative to noisy hardware — not a hardware claim.

---

## What is in the repository

```text
config/                     Central YAML configuration and experiment registry
data/                       Dataset download, processed arrays, splits and metadata
docs/                       Methodology, experiments, results, limitations and paper
models/                     Trained model binaries and metadata
notebooks/                  Six guided analysis notebooks
presentation_assets/        Talk outline, key numbers and presentation figures
results/
  figures/                  Generated plots
  tables/                   CSV/Markdown result tables
  metrics/                  JSON metrics and statistical tests
  predictions/              Per-sample test scores
  circuits/                 QSVM/VQC circuit diagrams and resource JSON
  explainability/           Permutation, SHAP, PCA and quantum sensitivity outputs
  logs/                     Per-run logs
src/
  data/                     Loading, cleaning, splitting and leakage validation
  features/                 Scaling, PCA, mutual information and reduction
  classical/                CNN+LSTM model and training loop
  quantum/                  Encodings, circuits, QSVM and VQC
  evaluation/               Metrics, CV, statistics, drift, latency and zero-day experiments
  explainability/           Post-hoc explanations
  alerting/                 Tiered risk engine
  pipeline/                 Orchestration, training and reporting
  dashboard/                Streamlit application
tests/                      Unit and contract tests
run_pipeline.py             Main CLI
run_all.py                  Convenience wrapper
demo.py                     Inference-only demonstration CLI
```

### What is not committed

The repository keeps code, configuration, docs, metadata, result tables and figures in version
control. Large or regenerable artifacts are intentionally ignored:

- `data/raw/KDDTrain+.txt` and `data/raw/KDDTest+.txt` — downloaded automatically from the public
  NSL-KDD mirror; see [data/README.md](data/README.md).
- `data/processed/*.npz`, `*.parquet`, `*.joblib` — rebuilt by the pipeline.
- `models/*.pt`, `*.joblib`, `*.pkl` — rebuilt by training.
- `results/logs/*.log` — regenerated on every run.

This keeps clones small and avoids redistributing the raw benchmark. Result tables, figures, metrics
and the trained-model metadata remain in the repository.

---

## Quick start

### 1. Environment

Tested with **Python 3.14.5 on Windows 11, CPU only**. Pinned versions are in
[requirements.txt](requirements.txt); a Conda definition is in [environment.yml](environment.yml).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Dataset

The pipeline downloads NSL-KDD automatically on first run:

```bash
python run_pipeline.py --mode quick
```

To download and validate only:

```bash
python scripts/download_dataset.py
```

Expected files and SHA-256 checksums are documented in [data/README.md](data/README.md).

### 3. Full quick-profile pipeline

```bash
python run_all.py --mode quick
# or
python run_pipeline.py --mode quick --force
```

The clean 14-stage quick run took **603.3 s** (about 10 minutes) on the tested CPU. `--force`
re-runs stages even when the run manifest says they are complete.

### 4. Other entry points

```bash
python run_pipeline.py --mode classical
python run_pipeline.py --mode quantum
python run_pipeline.py --mode evaluate
python run_pipeline.py --mode dashboard
python run_pipeline.py --model cnn_lstm
python run_pipeline.py --experiment zero_day
python run_pipeline.py --experiment encoding
python run_pipeline.py --experiment latency
python run_pipeline.py --experiment drift
python run_pipeline.py --stage report
python demo.py --all-models
python demo.py --all-models --json
python -m pytest -q
```

Interactive dashboard:

```bash
streamlit run src/dashboard/app.py
```

A static dashboard is also written to `results/dashboard.html`.

---

## Experimental design in one page

1. **Pool**: `KDDTrain+` (125,973 rows) is split 70/15/15. Quick mode first draws a stratified
   12,000-row pool; the resulting split is **8,399 train / 1,801 validation / 1,800 common test**.
2. **External set**: the official `KDDTest+` (22,544 rows) is never trained on. It supplies the
   17 novel attack types used in the external-generalisation experiment.
3. **Preprocessing**: all statistics are fitted on the training split only. Validation selects the
   feature configuration, early-stopping epoch, decision threshold and encoding comparison; the test
   split is scored once.
4. **Reduction**: PCA and mutual information at 8/12/16 dimensions are compared on validation.
   PCA-8 is selected for the main comparison because the quantum simulator is capped at 8 qubits.
5. **Common representation**: every model receives the same 8-column matrix, min–max rescaled to
   `[0, π]` using training statistics.
6. **Training budgets**: the CNN+LSTM uses all 8,399 training rows. QSVM and VQC use a stratified
   600-row quantum training subset and 200-row validation subset because statevector evaluation cost
   grows with training-set size. This asymmetry is explicit and is a limitation of the study.
7. **Thresholds**: selected on validation F1 for each model (CNN 0.35, QSVM 0.35, VQC 0.50), then
   frozen for the common test set.
8. **Statistics**: McNemar on paired predictions, paired bootstrap for F1/AUC differences, and
   Wilcoxon across folds (underpowered by design with 3 folds).

Full details: [docs/methodology.md](docs/methodology.md),
[docs/experiments.md](docs/experiments.md) and [docs/limitations.md](docs/limitations.md).

---

## Reproducibility

- Seed **42**; deterministic split and model initialisation where the libraries allow it.
- Saved artifacts: `data/processed/preprocessor.joblib`, `reducer.joblib`, `scaler.joblib`,
  `models/*.joblib|*.pt`, `models/*_meta.json`, `results/run_manifest.json`.
- Environment capture: `results/environment.json`.
- Package versions are logged at the start of every run and in
  [results/final_summary.md](results/final_summary.md).
- Absolute timings vary with CPU load. Latency values are medians of five repeats and should be read
  as indicative, not as a hardware benchmark.

### Known implementation notes

- TensorFlow was not available for Python 3.14, so the classical baseline uses PyTorch.
- Qiskit 2.x deprecates the class forms of `ZZFeatureMap` and `RealAmplitudes`; the code uses the
  function forms `zz_feature_map` and `real_amplitudes`.
- The VQC fast path precomputes feature states and applies `Operator(ansatz).data`. It is
  mathematically equivalent to full-circuit simulation and is checked in `tests/`.
- Amplitude encoding with computational-basis state preparation yields a squared-cosine fidelity
  kernel, i.e. a classical polynomial kernel. The encoding comparison is an empirical check, not
  evidence of quantum advantage.

---

## Scope, limitations and ethics

This is a research and teaching artefact, **not a production IDS**. It does not claim quantum
advantage, zero-day detection, real-time deployability or hardware performance.

Key limitations:

- **Quick profile only.** The full 125,973-row profile was not executed; absolute metrics are not the
  best achievable on NSL-KDD.
- **Training-budget asymmetry.** Quantum models see 600 rows versus 8,399 for the CNN. The comparison
  is fair in evaluation, not in data volume.
- **Eight-qubit cap.** PCA-8 limits the quantum feature space.
- **Simulation only.** No hardware noise, decoherence, readout error, transpilation or finite-shot
  sampling is included in the reported metrics.
- **VQC did not converge.** COBYLA used all 151 evaluations; final loss 0.4585.
- **Single seed.** Cross-validation provides stability evidence, but no multi-seed study was run.
- **Small external evidence.** The unseen-category holdout has 168 rows. The external novel-type
  subset is larger (13,461 rows) but comes from one normal-heavy file with strong covariate shift,
  and its metrics are point estimates without external-set confidence intervals.
- **Old benchmark.** NSL-KDD is a 2009 dataset with synthetic attack composition and no timestamps.
  Results do not transfer automatically to modern enterprise traffic.
- **No production validation.** No adversarial, streaming, encrypted-traffic or live deployment
  evaluation was performed.
- **Alert engine is assistive.** The tiered policy is a demonstration, not an automatic blocking
  policy.

NSL-KDD is a public benchmark. No new traffic was captured, no human subjects were involved and no
live network was monitored. Check the original dataset's redistribution terms before publishing raw
rows. This repository does not redistribute the raw files.

---

## Documentation map

| Document | Contents |
|---|---|
| **[docs/conference_paper_sourcebook.md](docs/conference_paper_sourcebook.md)** | **Authoritative paper sourcebook: measured values, methods, tables, statistics, figure captions, limitations, references and claim guardrails.** |
| [docs/conference_paper.md](docs/conference_paper.md) | Conference-style paper draft using only measured values |
| [docs/methodology.md](docs/methodology.md) | Dataset, preprocessing, leakage controls, models and evaluation protocol |
| [docs/experiments.md](docs/experiments.md) | Every experiment, its protocol, outputs and measured result |
| [docs/results.md](docs/results.md) | Full result tables and interpretation |
| [docs/limitations.md](docs/limitations.md) | Threats to validity, scope limits and honest caveats |
| [docs/paper_draft.md](docs/paper_draft.md) | Auto-generated paper draft from the results pipeline |
| [presentation_assets/talk_outline.md](presentation_assets/talk_outline.md) | Slide-by-slide talk outline |
| [presentation_assets/key_numbers.md](presentation_assets/key_numbers.md) | One-page viva/report number sheet |
| [data/README.md](data/README.md) | Dataset provenance, checksums and splitting rules |

For writing the paper, start with the sourcebook. It is the single source of truth; the generated
summary and paper draft are convenience outputs.

---

## Testing

```bash
python -m pytest -q
```

Expected: **31 passed**. Tests cover data validation, leakage checks, feature reduction, quantum
encodings and kernel properties, VQC equivalence to direct simulation, metrics, statistical tests,
alert tiers and pipeline output contracts.

---

## Citation

If you use this code or the evaluation protocol, cite the NSL-KDD dataset and the quantum-kernel
references listed in [docs/conference_paper.md](docs/conference_paper.md). The project itself can be
cited as:

> Quantum-Enhanced Intrusion Detection System (QE-IDS): a leakage-audited classical vs quantum
> benchmark on NSL-KDD. B.Tech final-year project, 2026.

---

## Repository status

- Quick-profile pipeline executed end to end.
- Primary common-test result: classical CNN+LSTM strongest on F1.
- Secondary external novel-type result: QSVM strongest on F1/ROC-AUC, scope-specific.
- Test suite: **31 passed**.
- Paper sourcebook: **[docs/conference_paper_sourcebook.md](docs/conference_paper_sourcebook.md)**.
