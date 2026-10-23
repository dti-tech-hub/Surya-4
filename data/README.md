# Dataset: NSL-KDD

This project uses the **real NSL-KDD** benchmark. No synthetic or fabricated records are used.

## Files

| File | Rows | Size (bytes) | SHA-256 |
|---|---:|---:|---|
| `data/raw/KDDTrain+.txt` | 125,973 | 19,109,424 | `1B86D2F957B33082081BBA410FE129B475EFEBCC13C9014C3F447C8271AADF95` |
| `data/raw/KDDTest+.txt` | 22,544 | 3,441,513 | `FA46B0935342616AA83B7C2578DB355B6A7AAABBC492248172C7A1E8B7AB8F84` |

The files have **no header row** and 43 columns: 41 features, `label`, and `difficulty`.
`difficulty` is dropped before modelling. The loader assigns the standard NSL-KDD column names.

## Download

```bash
python scripts/download_dataset.py
```

The script downloads from the public mirror:

- <https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt>
- <https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt>

If the files already exist, `--force` is required to overwrite them. The pipeline also downloads
the files automatically during the `prepare` stage when they are missing.

## Verified composition

| Property | KDDTrain+ | KDDTest+ |
|---|---:|---:|
| Rows | 125,973 | 22,544 |
| Missing values | 0 | 0 |
| Exact duplicate rows | 0 | 0 |
| Distinct labels | 23 | 38 |
| Attack types absent from KDDTrain+ | — | 17 |

The 17 attack types absent from the **full KDDTrain+ label set** are:
`apache2, httptunnel, mailbomb, mscan, named, processtable, ps, saint, sendmail, snmpgetattack,
snmpguess, sqlattack, udpstorm, worm, xlock, xsnoop, xterm`.

The external novel-type experiment uses this full-vocabulary definition, not the quick-mode
subsample, so the definition is stable across quick and full profiles. The generated list and
counts are stored in `results/metrics/external_novel_details.json`.

## Role of each file

- **KDDTrain+** is the only source of training data. Quick mode draws a stratified 12,000-row
  pool and splits it 70/15/15 (8,399 / 1,801 / 1,800 rows).
- **KDDTest+** is an external, never-trained-on set. It is used only for:
  1. the external novel-attack-type experiment, and
  2. the simulated file-order window scan in the drift stress test.

The official KDDTest+ file is **not** used for hyperparameter selection, threshold selection or
feature selection.

## Preprocessing artefacts

Generated under `data/processed/`:

| File | Purpose |
|---|---|
| `preprocessor.joblib` | Train-fit imputer, one-hot vocabulary and numeric scaler |
| `reducer.joblib` | Train-fit PCA or mutual-information reducer |
| `scaler.joblib` | Train-fit [0, π] scaler applied after reduction |
| `arrays.npz` | Reduced train/validation/test matrices and labels |
| `common_test.parquet` | The shared test rows used by every model |
| `external.parquet` | Cleaned KDDTest+ rows |
| `preprocessor.json`, `reducer.json`, `scaler.json` | Human-readable metadata |

Generated metadata under `data/metadata/` includes the dataset validation report, cleaning
report, split report, feature-selection report and leakage audit. All of these are produced by
executed code, not written by hand.

## Leakage policy

Every fitted transformation is estimated on the training split only:

- median imputation values,
- one-hot vocabulary (rare levels collapsed using training frequencies),
- numeric standardisation statistics,
- PCA loadings or mutual-information scores,
- [0, π] min–max rescaling.

Validation is used for early stopping, threshold selection and feature-configuration selection.
The test split is touched once, for final reporting. The audit checks are listed in
`data/metadata/leakage_audit.json`; all eight passed.

## License and attribution

NSL-KDD is distributed for research use. Cite:

> Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). A detailed analysis of the KDD
> CUP 99 data set. *Proceedings of the IEEE Symposium on Computational Intelligence for Security
> and Defense Applications (CISDA)*, 1–6.

The raw files are not authored by this project and are excluded from version control by
`.gitignore`; download them with the script above.
