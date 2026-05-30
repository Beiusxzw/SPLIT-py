from __future__ import annotations

import sys
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rctd-py followed by Python SPLIT purification."
    )
    parser.add_argument(
        "--ref-path",
        type=Path,
        required=True,
        help="Single-cell reference .h5ad path.",
    )
    parser.add_argument(
        "--spatial-path",
        type=Path,
        required=True,
        help="Spatial transcriptomics 10x .h5 path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for output .h5ad, parquet tables, summary, and log.",
    )
    parser.add_argument(
        "--cell-type-col",
        default="Annotation",
        help="Reference obs column used as rctd-py deconvolution cell types.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Cell chunk size for SPLIT purification.",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.01,
        help="Minimum RCTD weight used during SPLIT post-processing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from split import run_split_rctd_py_pipeline

    run_split_rctd_py_pipeline(
        ref_path=args.ref_path,
        spatial_path=args.spatial_path,
        out_dir=args.out_dir,
        cell_type_col=args.cell_type_col,
        chunk_size=args.chunk_size,
        min_weight=args.min_weight,
        log_fn=lambda message: print(message, flush=True),
    )


if __name__ == "__main__":
    main()
