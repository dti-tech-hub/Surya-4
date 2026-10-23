# QE-IDS — Live Demo Script

**Goal:** show a trained model scoring real held-out NSL-KDD records and explain the alert tier.
**Duration:** 5 minutes (3 minutes if the pipeline has already been run).
**Prerequisites:** completed quick pipeline (`python run_all.py --mode quick`) and the saved artefacts
in `models/`, `data/processed/` and `results/`.

---

## 0. Before the audience arrives

```bash
# From the project root, confirm artefacts exist
python -c "from pathlib import Path; print(Path('models/cnn_lstm.pt').exists(), Path('data/processed/arrays.npz').exists())"

# Warm the model load (optional, avoids a pause on stage)
python demo.py --model cnn_lstm --n-samples 3
```

Expected: `True True`, then a three-row table with `p(attack)`, predicted label, true category,
risk tier and action.

**Fallback:** if anything is missing, run `python run_all.py --mode quick` (about 10 minutes on
the tested CPU). If there is no time, show `results/dashboard.html` instead.

---

## 1. Open with the source of truth (30 s)

```bash
python -c "import json; d=json.load(open('results/final_summary.json')); print(d['mode'], d['seed'], d['leakage_audit_passed']); [print(m['model_label'], round(m['f1'],4), round(m['roc_auc'],4)) for m in d['master_table']]"
```

Expected:

```
quick 42 True
CNN+LSTM 0.9533 0.9891
QSVM 0.9231 0.9903
VQC 0.8028 0.8982
```

**Say:** “These are the measured test numbers; nothing here is typed by hand.”

---

## 2. Single-record demo — classical model (60 s)

```bash
python demo.py --model cnn_lstm --sample-index 0
```

Expected shape:

```
idx  true  p(attack)  pred  category  tier  action
0    attack  0.9998     1      DoS   ESCALATE  BLOCK_AND_ESCALATE
```

**Say:**

- The probability is the model’s attack score, not a calibrated guarantee.
- The tier comes from the five-level policy in `config/config.yaml`.
- “BLOCK_AND_ESCALATE” is a recommendation; the engine is assistive, not an automatic firewall.

---

## 3. Compare all three models on the same record (75 s)

```bash
python demo.py --all-models --sample-index 0
```

Expected: three rows for the same test record, one per model, with potentially different
probabilities and tiers.

**Say:**

- Identical input vector, identical threshold policy, three different inductive biases.
- QSVM and VQC scores come from exact statevector simulation.
- The classical model is usually the most confident on clear DoS traffic.

---

## 4. Batch demo — 20 records (45 s)

```bash
python demo.py --all-models --n-samples 20
```

**Say:**

- Each row is a real held-out record from the common test set.
- The true category column is shown only for the demo; it is not available at inference time.
- Use this to discuss false positives and tier distribution.

---

## 5. JSON output for integration (30 s)

```bash
python demo.py --all-models --n-samples 5 --json
```

Expected: a JSON array with model, index, true label, probability, predicted label, category,
risk score, tier and action.

**Say:** “This is the shape an upstream SIEM or SOAR playbook would consume.”

---

## 6. Dashboard (45 s)

```bash
python run_pipeline.py --mode dashboard
start results/dashboard.html        # Windows
# open results/dashboard.html       # macOS
# xdg-open results/dashboard.html   # Linux
```

Or the interactive version:

```bash
streamlit run src/dashboard/app.py
```

**Say:**

- The static dashboard embeds the main figures and headline tables.
- The Streamlit app lets you filter by model and inspect per-sample predictions.
- Both read saved artefacts; neither retrains anything.

---

## 7. Close (30 s)

**Say:**

> “The demo is deliberately boring: load a saved model, score a real held-out record, map the
> probability to an alert tier. The interesting part is the protocol behind the numbers — identical
> features, identical splits, train-only preprocessing, and a negative result reported honestly.”

Then invite questions.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Model not found` | pipeline not run or models deleted | `python run_all.py --mode quick` |
| `Processed artefacts missing` | `data/processed/arrays.npz` absent | `python run_pipeline.py --mode prepare` then `--mode features` |
| Unicode console error on Windows | legacy code page | use Windows Terminal, or `set PYTHONUTF8=1` |
| Dashboard does not open | browser blocked local file | run `streamlit run src/dashboard/app.py` |
| Demo is slow on first call | model load + Qiskit import | warm up beforehand (step 0) |

## Do-not-say list

- Do **not** call the external test “zero-day detection”. Say “external novel attack types”.
- Do **not** call the simulator “quantum hardware”.
- Do **not** claim quantum advantage; the measured result is the opposite on F1.
- Do **not** quote the full-profile numbers; only quick-profile results were executed.
- Do **not** present the unseen-category holdout as a general quantum win; n = 168, single seed.
