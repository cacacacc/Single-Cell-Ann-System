"""Command-line helper for creating a gene-aligned joint .h5ad dataset.

This script is a thin wrapper around ``backend.dataset_preprocessor``.  It is
useful for offline preprocessing or reproducible demos where multiple source
datasets need to be cleaned, gene-aligned, merged and accompanied by a JSON
report outside the Flask UI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.dataset_preprocessor import prepare_joint_dataset


def parse_args() -> argparse.Namespace:
    """Parse CLI options and keep defaults aligned with the web UI workflow."""
    parser = argparse.ArgumentParser(
        description="Clean and align multiple single-cell .h5ad datasets."
    )
    parser.add_argument("inputs", nargs="+", help="Input .h5ad files.")
    parser.add_argument(
        "-o",
        "--output",
        default="data/joint_aligned.h5ad",
        help="Output merged .h5ad path.",
    )
    parser.add_argument(
        "--report",
        default="data/joint_aligned_report.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--join",
        choices=("inner", "outer"),
        default="inner",
        help="Gene alignment mode. inner keeps shared genes; outer keeps the union.",
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        dest="dataset_ids",
        help="Dataset label. Repeat once per input; defaults to file stems.",
    )
    parser.add_argument("--min-cells", type=int, default=1)
    parser.add_argument("--min-genes", type=int, default=1)
    parser.add_argument("--normalize-total", action="store_true")
    parser.add_argument("--log1p", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run preprocessing and print the structured report to stdout."""
    args = parse_args()
    report = prepare_joint_dataset(
        [Path(item) for item in args.inputs],
        Path(args.output),
        join=args.join,
        dataset_ids=args.dataset_ids,
        min_cells=args.min_cells,
        min_genes=args.min_genes,
        normalize_total=args.normalize_total,
        log1p=args.log1p,
        report_path=Path(args.report),
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

