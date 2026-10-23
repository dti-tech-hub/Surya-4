#!/usr/bin/env python
"""Train and evaluate the VQC (variational quantum classifier) only."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_pipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the VQC")
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--quiet", action="store_true")
    args, passthrough = parser.parse_known_args(argv)

    forwarded = [
        "--mode", args.mode, "--model", "vqc",
        "--stage", "prepare", "--stage", "features", "--stage", "train",
        "--stage", "evaluate", "--stage", "tables", "--stage", "report",
    ]
    if args.quiet:
        forwarded.append("--quiet")
    forwarded.extend(passthrough)
    return run_pipeline.main(forwarded)


if __name__ == "__main__":
    sys.exit(main())
