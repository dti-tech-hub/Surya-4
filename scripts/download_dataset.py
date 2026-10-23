#!/usr/bin/env python
"""Download the public NSL-KDD files into data/raw/ and validate them.

No registration is required.  The script is idempotent: existing files are
kept unless ``--overwrite`` is passed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import NSLKDDDataset, download_nsl_kdd, validate_dataset  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download NSL-KDD")
    parser.add_argument("--raw-dir", default=None,
                        help="Destination (default: paths.data_raw from config)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(root=ROOT)
    raw_dir = Path(args.raw_dir) if args.raw_dir else Path(cfg.get_path("paths.data_raw"))
    print(f"Downloading NSL-KDD into {raw_dir} ...")
    paths = download_nsl_kdd(raw_dir, overwrite=args.overwrite)
    for path in paths:
        print(f"  OK  {path}  ({path.stat().st_size:,} bytes)")

    report = (NSLKDDDataset(raw_dir).validate() if args.raw_dir else validate_dataset(cfg))
    print(json.dumps(report, indent=2)[:4000])
    if not report.get("valid"):
        print("[FAILED] Dataset validation failed.", file=sys.stderr)
        return 1
    print("\nDataset ready. Next: python scripts/preprocess.py --mode quick")
    return 0


if __name__ == "__main__":
    sys.exit(main())
