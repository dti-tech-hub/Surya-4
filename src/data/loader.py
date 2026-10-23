"""Dataset loading and the dataset abstraction layer.

Only NSL-KDD is bundled/validated.  CICIDS2017/2018 is registered as an
optional plugin so another dataset can be added without touching the pipeline.
"""
from __future__ import annotations

import io
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# 41 NSL-KDD features + label + difficulty, in file order.
NSL_KDD_FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
NSL_KDD_COLUMNS = NSL_KDD_FEATURES + ["label", "difficulty"]

NSL_KDD_CATEGORICAL = ["protocol_type", "service", "flag"]

# Standard NSL-KDD attack taxonomy (Tavallaee et al., 2009).
ATTACK_CATEGORY_MAP: dict[str, str] = {}
for _cat, _attacks in {
    "DoS": [
        "back", "land", "neptune", "pod", "smurf", "teardrop", "mailbomb",
        "processtable", "udpstorm", "apache2", "worm",
    ],
    "Probe": ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"],
    "R2L": [
        "ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
        "warezclient", "warezmaster", "xlock", "xsnoop", "snmpguess",
        "snmpgetattack", "httptunnel", "sendmail", "named",
    ],
    "U2R": ["buffer_overflow", "loadmodule", "perl", "rootkit", "ps", "sqlattack", "xterm"],
}.items():
    for _a in _attacks:
        ATTACK_CATEGORY_MAP[_a] = _cat


class DatasetError(RuntimeError):
    """Raised when a dataset is missing or malformed."""


class BaseDataset(ABC):
    """Minimal dataset interface used by the pipeline."""

    name: str = "base"

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return a clean dataframe with a ``label`` column and ``attack_category``."""

    @abstractmethod
    def validate(self) -> dict:
        """Return a validation report (counts, label cardinality, ...)."""


class NSLKDDDataset(BaseDataset):
    name = "nsl-kdd"

    def __init__(self, raw_dir: str | Path, train_file: str = "KDDTrain+.txt",
                 test_file: str = "KDDTest+.txt", has_header: bool = False) -> None:
        self.raw_dir = Path(raw_dir)
        self.train_file = train_file
        self.test_file = test_file
        self.has_header = has_header

    # -- paths -------------------------------------------------------------
    @property
    def train_path(self) -> Path:
        return self.raw_dir / self.train_file

    @property
    def test_path(self) -> Path:
        return self.raw_dir / self.test_file

    def require_files(self) -> None:
        missing = [p for p in (self.train_path, self.test_path) if not p.exists()]
        if not missing:
            return
        listing = "\n".join(f"  - {p}" for p in missing)
        raise DatasetError(
            "NSL-KDD files are missing:\n"
            f"{listing}\n\n"
            "Download the official files (free, no registration) from:\n"
            "  https://www.unb.ca/cic/datasets/nsl.html\n"
            "or the widely used mirror:\n"
            "  https://github.com/defcom17/NSL_KDD\n\n"
            "Then place them at:\n"
            f"  {self.train_path}\n"
            f"  {self.test_path}\n\n"
            "Automated option:\n"
            "  python scripts/download_dataset.py\n"
        )

    # -- loading -----------------------------------------------------------
    def _read(self, path: Path) -> pd.DataFrame:
        header = 0 if self.has_header else None
        df = pd.read_csv(path, header=header)
        if self.has_header:
            df.columns = [str(c).strip() for c in df.columns]
        else:
            if df.shape[1] != len(NSL_KDD_COLUMNS):
                raise DatasetError(
                    f"{path.name} has {df.shape[1]} columns, expected "
                    f"{len(NSL_KDD_COLUMNS)} (41 features + label + difficulty). "
                    "Check that you downloaded the KDDTrain+/KDDTest+ files and "
                    "not a different variant."
                )
            df.columns = NSL_KDD_COLUMNS
        return df

    def load_raw(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.require_files()
        return self._read(self.train_path), self._read(self.test_path)

    def load(self) -> pd.DataFrame:
        """Primary pool = KDDTrain+ (the file intended for model fitting)."""
        train, _ = self.load_raw()
        return harmonise_labels(train)

    def load_external(self) -> pd.DataFrame:
        """Official KDDTest+ — never used for fitting or model selection."""
        _, test = self.load_raw()
        return harmonise_labels(test)

    def validate(self) -> dict:
        train, test = self.load_raw()
        report = {
            "dataset": self.name,
            "train_file": str(self.train_path),
            "test_file": str(self.test_path),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "n_columns": int(train.shape[1]),
            "n_features": len(NSL_KDD_FEATURES),
            "missing_values_train": int(train.isna().sum().sum()),
            "missing_values_test": int(test.isna().sum().sum()),
            "duplicate_rows_train": int(train.duplicated().sum()),
            "duplicate_rows_test": int(test.duplicated().sum()),
            "train_labels": {str(k): int(v) for k, v in train["label"].value_counts().items()},
            "test_labels": {str(k): int(v) for k, v in test["label"].value_counts().items()},
            "train_attack_types": int(train["label"].nunique()),
            "test_attack_types": int(test["label"].nunique()),
            "test_only_attack_types": sorted(set(test["label"].unique()) - set(train["label"].unique())),
        }
        report["valid"] = (
            report["train_rows"] > 0
            and report["test_rows"] > 0
            and report["n_columns"] == len(NSL_KDD_COLUMNS)
            and report["missing_values_train"] == 0
        )
        return report


def harmonise_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary + multiclass + attack-type columns, drop the difficulty column."""
    df = df.copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    if "difficulty" in df.columns:
        df = df.drop(columns=["difficulty"])

    df["is_attack"] = (df["label"] != "normal").astype(np.int8)
    df["attack_type"] = df["label"]
    mapped = df["label"].map(ATTACK_CATEGORY_MAP)
    df["attack_category"] = np.where(
        df["label"] == "normal", "Normal", mapped.fillna("Unknown").astype(str)
    )
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"label", "difficulty", "is_attack", "attack_type", "attack_category", "split"}
    return [c for c in df.columns if c not in excluded]


class CICIDSDataset(BaseDataset):
    """Optional CICIDS2017/2018 plugin (not exercised by the default pipeline)."""

    name = "cicids"

    EXPECTED_FILES = {
        "cicids2017": "MachineLearningCSV.zip",
        "cicids2018": "02-14-2018.csv",
    }

    def __init__(self, raw_dir: str | Path, variant: str = "cicids2017") -> None:
        self.raw_dir = Path(raw_dir)
        self.variant = variant

    def require_files(self) -> None:
        if not any(self.raw_dir.glob("*.csv")):
            raise DatasetError(
                f"No CICIDS CSV files found in {self.raw_dir}.\n"
                "CICIDS2017 must be requested from:\n"
                "  https://www.unb.ca/cic/datasets/ids-2017.html\n"
                "CICIDS2018 from:\n"
                "  https://www.unb.ca/cic/datasets/ids-2018.html\n"
                "Extract the per-day CSVs into data/raw/ and re-run with "
                "--set dataset.name=cicids."
            )

    def load(self) -> pd.DataFrame:  # pragma: no cover - requires external data
        raise DatasetError(
            "CICIDS support is registered but intentionally not enabled in this "
            "release. NSL-KDD is the validated primary dataset. To add CICIDS, "
            "implement CICIDSDataset.load() with the same output contract "
            "(label, is_attack, attack_category columns) and add a handler in "
            "src/data/loader.py::build_dataset."
        )

    def validate(self) -> dict:  # pragma: no cover
        return {"dataset": self.name, "valid": False, "reason": "not implemented"}


def build_dataset(cfg) -> BaseDataset:
    name = str(cfg.get_path("dataset.name", "nsl-kdd")).lower()
    raw_dir = cfg.get_path("paths.data_raw")
    if name in ("nsl-kdd", "nslkdd", "nsl_kdd", "kdd"):
        return NSLKDDDataset(
            raw_dir=raw_dir,
            train_file=cfg.get_path("dataset.train_file", "KDDTrain+.txt"),
            test_file=cfg.get_path("dataset.test_file", "KDDTest+.txt"),
            has_header=bool(cfg.get_path("dataset.has_header", False)),
        )
    if name in ("cicids", "cicids2017", "cicids2018"):
        return CICIDSDataset(raw_dir=raw_dir, variant=name)
    raise DatasetError(f"Unknown dataset '{name}'. Supported: nsl-kdd, cicids (optional).")


def validate_dataset(cfg) -> dict:
    ds = build_dataset(cfg)
    return ds.validate()


# ---------------------------------------------------------------------------
# Optional automatic download (NSL-KDD is freely redistributable)
# ---------------------------------------------------------------------------
MIRRORS = {
    "KDDTrain+.txt": [
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
    ],
    "KDDTest+.txt": [
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt",
    ],
}


def download_nsl_kdd(raw_dir: str | Path, overwrite: bool = False) -> list[Path]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, urls in MIRRORS.items():
        target = raw_dir / filename
        if target.exists() and not overwrite:
            written.append(target)
            continue
        last_error: Optional[Exception] = None
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=120) as response:
                    payload = response.read()
                if len(payload) < 1_000_000:
                    raise ValueError(f"Downloaded payload too small ({len(payload)} bytes)")
                target.write_bytes(payload)
                written.append(target)
                last_error = None
                break
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
        if last_error is not None:
            raise DatasetError(
                f"Could not download {filename}: {last_error}\n"
                f"Manual download: https://www.unb.ca/cic/datasets/nsl.html\n"
                f"Place the file at {target}"
            )
    return written


def dataset_fingerprint(df: pd.DataFrame) -> dict:
    """Cheap integrity fingerprint for the processed data."""
    import hashlib

    payload = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
