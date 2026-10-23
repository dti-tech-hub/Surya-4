#!/usr/bin/env python
"""Regenerate tables, statistics, dashboard and reports from existing runs.

This does not retrain any model.  It is equivalent to::

    python run_pipeline.py --mode evaluate
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
    parser = argparse.ArgumentParser(description="Generate result artefacts")
    parser.add_argument("--quiet", action="store_true")
    args, passthrough = parser.parse_known_args(argv)
    forwarded = ["--mode", "evaluate"]
    if args.quiet:
        forwarded.append("--quiet")
    forwarded.extend(passthrough)
    return run_pipeline.main(forwarded)


if __name__ == "__main__":
    sys.exit(main())
