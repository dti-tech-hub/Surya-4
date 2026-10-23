#!/usr/bin/env python
"""QE-IDS command-line interface.

Examples
--------
python run_pipeline.py --mode quick
python run_pipeline.py --mode full
python run_pipeline.py --mode classical
python run_pipeline.py --mode quantum
python run_pipeline.py --mode evaluate
python run_pipeline.py --mode dashboard
python run_pipeline.py --model cnn_lstm
python run_pipeline.py --experiment zero_day
python run_pipeline.py --experiment encoding
python run_pipeline.py --experiment latency
python run_pipeline.py --experiment drift
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make ``src`` importable when the script is run directly.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, PROJECT_ROOT
from src.utils.io import ensure_dirs
from src.utils.logging import get_logger, setup_logging
from src.utils.reproducibility import environment_summary

LOGGER = get_logger("qeids.cli")

MODES = ("quick", "full", "classical", "quantum", "evaluate", "dashboard")
EXPERIMENTS = (
    "all", "zero_day", "encoding", "latency", "drift", "explain",
    "cv", "stats", "feature_reduction",
)

EXPERIMENT_STAGES = {
    "zero_day": ["zero_day", "tables", "report"],
    "encoding": ["encoding", "tables", "report"],
    "latency": ["latency", "tables", "report"],
    "drift": ["drift", "tables", "report"],
    "explain": ["explain", "tables", "report"],
    "cv": ["cv", "stats", "tables", "report"],
    "stats": ["stats", "tables", "report"],
    "feature_reduction": ["features", "tables", "report"],
}

FULL_STAGES = [
    "prepare", "features", "train", "evaluate", "cv", "stats",
    "zero_day", "encoding", "latency", "drift", "explain",
    "tables", "dashboard", "report",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Quantum-Enhanced Intrusion Detection System — experiment pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=MODES, default="quick",
                        help="Experiment mode (default: quick)")
    parser.add_argument("--model", choices=("cnn_lstm", "qsvm", "vqc", "all"),
                        default="all", help="Restrict to one model family")
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="all",
                        help="Run a single research experiment")
    parser.add_argument("--stage", action="append", default=None,
                        help="Run only these stage(s); can be repeated")
    parser.add_argument("--config", default=None, help="Alternative YAML config")
    parser.add_argument("--set", action="append", default=None, metavar="KEY=VALUE",
                        help="Override a config entry, e.g. --set project.seed=7")
    parser.add_argument("--force", action="store_true",
                        help="Re-run stages even if the manifest says they completed")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--list-stages", action="store_true",
                        help="Print available stages and exit")
    return parser


def resolve_stages(args) -> list[str]:
    if args.stage:
        return list(args.stage)
    if args.mode == "dashboard":
        return ["dashboard"]
    if args.mode == "evaluate":
        return ["evaluate", "stats", "tables", "dashboard", "report"]
    if args.experiment != "all":
        stages = list(EXPERIMENT_STAGES[args.experiment])
        # Prerequisites are added below if their artifacts are missing.
        return stages
    if args.mode == "classical":
        return ["prepare", "features", "train", "evaluate", "cv", "stats",
                "latency", "explain", "tables", "dashboard", "report"]
    if args.mode == "quantum":
        return ["prepare", "features", "train", "evaluate", "stats",
                "encoding", "latency", "zero_day", "tables", "dashboard", "report"]
    return list(FULL_STAGES)


def ensure_prerequisites(stages: list[str], cfg) -> list[str]:
    """Prepend prepare / features when their on-disk artifacts are absent."""
    splits = Path(cfg.get_path("paths.data_splits")) / "train.parquet"
    arrays = Path(cfg.get_path("paths.data_processed")) / "arrays.npz"

    needs_data = any(s in stages for s in FULL_STAGES if s not in ("dashboard", "report"))
    if not needs_data:
        return stages

    if "prepare" not in stages and not splits.exists():
        LOGGER.info("Split artifacts missing — adding the prepare stage")
        stages = ["prepare"] + stages

    if "prepare" in stages:
        # Features must be rebuilt whenever the split is rebuilt.
        if "features" not in stages:
            idx = stages.index("prepare")
            stages = stages[: idx + 1] + ["features"] + stages[idx + 1:]
    elif "features" not in stages and not arrays.exists():
        LOGGER.info("Feature artifacts missing — adding the features stage")
        stages = ["features"] + stages
    return stages


def _configure_console() -> None:
    """Make console output tolerant of Unicode on legacy Windows code pages."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: list[str] | None = None) -> int:
    _configure_console()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_stages:
        from src.pipeline.orchestrator import STAGES

        print("Available stages:")
        for stage in STAGES:
            print(f"  - {stage}")
        return 0

    overrides = list(args.set or [])
    overrides.append(f"experiment.mode={args.mode if args.mode in ('quick', 'full') else 'quick'}")
    cfg = load_config(path=args.config, overrides=overrides, root=PROJECT_ROOT)
    ensure_dirs(cfg)

    log_dir = Path(cfg.get_path("paths.logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(
        level=str(cfg.get_path("logging.level", "INFO")),
        log_file=log_file,
        console=not args.quiet,
        force=True,
    )

    models = ("cnn_lstm", "qsvm", "vqc") if args.model == "all" else (args.model,)
    stages = resolve_stages(args)
    stages = ensure_prerequisites(stages, cfg)

    LOGGER.info("=" * 78)
    LOGGER.info("Quantum-Enhanced IDS — mode=%s model=%s experiment=%s",
                args.mode, args.model, args.experiment)
    LOGGER.info("Stages: %s", stages)
    LOGGER.info("Log file: %s", log_file)
    LOGGER.info("-" * 78)
    for line in environment_summary().splitlines():
        LOGGER.info(line)
    LOGGER.info("=" * 78)

    from src.pipeline.orchestrator import Pipeline

    pipeline = Pipeline(cfg, models=models, verbose=not args.quiet)
    try:
        outputs = pipeline.run(stages=stages)
    except Exception as exc:
        LOGGER.exception("Pipeline failed: %s", exc)
        print(f"\n[FAILED] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    reports = outputs.get("reports", {})
    print("\n" + "=" * 78)
    print("RUN COMPLETE")
    print("=" * 78)
    for key, value in reports.items():
        if key in ("findings", "presentation_assets"):
            continue
        print(f"  {key:22s}: {value}")
    findings = reports.get("findings", {})
    if findings.get("quantum_comparison"):
        print("\nFinding:")
        print("  " + findings["quantum_comparison"])
    print(f"\n  elapsed              : {outputs.get('elapsed_s', 0):.1f} s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
