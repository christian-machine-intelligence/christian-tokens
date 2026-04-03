"""CLI entry point for the training data audit.

Usage:
    python -m src --quick                        # smoke test
    python -m src                                # full audit (Pile)
    python -m src --dataset pile dolma fineweb   # multiple datasets
    python -m src --calibrate                    # + LLM calibration
    python -m src --analyze results/FILE.json    # regenerate tables
    python -m src --visualize results/FILE.json  # regenerate figures
"""

import argparse
import json
import sys
from pathlib import Path

from .config import AuditConfig, DATASET_REGISTRY, RESULTS_DIR, FIGURES_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit religious content in LLM training datasets"
    )

    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=list(DATASET_REGISTRY.keys()),
        default=["pile"],
        help="Datasets to audit (default: pile)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100_000,
        help="Documents to sample per dataset (default: 100000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke test: 1000 docs, Pile only",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run LLM calibration phase after keyword scan",
    )
    parser.add_argument(
        "--analyze",
        type=str,
        metavar="FILE",
        help="Regenerate analysis tables from a saved results JSON",
    )
    parser.add_argument(
        "--visualize",
        type=str,
        metavar="FILE",
        help="Regenerate figures from a saved results JSON",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Analysis-only mode
    if args.analyze:
        from .analysis import print_full_report

        path = Path(args.analyze)
        if not path.exists():
            print(f"Error: file not found: {path}")
            sys.exit(1)
        with open(path) as f:
            data = json.load(f)
        print_full_report(data)
        return

    # Visualization-only mode
    if args.visualize:
        from .visualize import generate_all_figures

        path = Path(args.visualize)
        if not path.exists():
            print(f"Error: file not found: {path}")
            sys.exit(1)
        with open(path) as f:
            data = json.load(f)
        generate_all_figures(data, FIGURES_DIR)
        return

    # Build config
    if args.quick:
        config = AuditConfig(
            datasets=["pile"],
            sample_size=1000,
            seed=args.seed,
        )
        print("Running quick smoke test (1000 docs, Pile only)...\n")
    else:
        config = AuditConfig(
            datasets=args.dataset,
            sample_size=args.sample_size,
            seed=args.seed,
        )

    # Run audit
    from .sampler import run_audit
    from .analysis import print_full_report

    audit_result = run_audit(config)
    print_full_report(audit_result.to_dict())

    # Optional calibration
    if args.calibrate:
        from .calibrate import run_calibration

        print("\n\nRunning LLM calibration phase...\n")
        run_calibration(audit_result, config)

    # Generate figures
    from .visualize import generate_all_figures

    generate_all_figures(audit_result.to_dict(), FIGURES_DIR)


if __name__ == "__main__":
    main()
