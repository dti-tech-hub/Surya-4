#!/usr/bin/env python
"""Master automation entry point.

python run_all.py            # quick mode (student laptop)
python run_all.py --mode full

Steps performed by the quick/full run:
  1. validate environment and dataset
  2. clean + split + persist
  3. leakage-safe feature reduction (validation-selected)
  4. train CNN+LSTM, QSVM, VQC
  5. evaluate on the common test set
  6. cross-validation + statistical tests
  7. unseen attack-category + external novel-attack experiments
  8. encoding comparison
  9. latency / quantum resource analysis
 10. distribution-shift stress test
 11. explainability
 12. figures, tables, dashboard, final summary and paper draft
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QE-IDS master automation")
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args, passthrough = parser.parse_known_args(argv)

    forwarded = ["--mode", args.mode]
    if args.force:
        forwarded.append("--force")
    if args.quiet:
        forwarded.append("--quiet")
    forwarded.extend(passthrough)
    return run_pipeline.main(forwarded)


if __name__ == "__main__":
    sys.exit(main())
