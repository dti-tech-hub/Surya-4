#!/usr/bin/env python
"""Build leakage-safe splits and reduced feature matrices.

Equivalent to::

    python run_pipeline.py --mode MODE --stage prepare --stage features
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_pipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess NSL-KDD")
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args, passthrough = parser.parse_known_args(argv)

    forwarded = ["--mode", args.mode, "--stage", "prepare", "--stage", "features"]
    if args.force:
        forwarded.append("--force")
    if args.quiet:
        forwarded.append("--quiet")
    forwarded.extend(passthrough)
    return run_pipeline.main(forwarded)


if __name__ == "__main__":
    sys.exit(main())
