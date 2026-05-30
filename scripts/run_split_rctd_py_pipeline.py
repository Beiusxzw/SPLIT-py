from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


REF_PATH = Path(
    "/mnt/volume2/snowxue/cell_segmentation_transcript_assignment/refscdata/"
    "Janesick_Chromium_FFPE_Human_Breast_Cancer_Chromium_FFPE_Human_Breast_Cancer_count_sample_filtered_feature_bc_matrix.h5ad"
)
SPATIAL_PATH = Path(
    "/mnt/volume2/snowxue/10x_Xenium_with_HE/Human_Breast_Biomarkers_S1_Bot/"
    "cell_feature_matrix.h5"
)
OUT_DIR = Path("/mnt/volume2/snowxue/SPLIT_python/run_outputs")


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rctd-py followed by Python SPLIT purification."
    )
    parser.add_argument(
        "--ref-path",
        type=Path,
        default=REF_PATH,
        help="Single-cell reference .h5ad path.",
    )
    parser.add_argument(
        "--spatial-path",
        type=Path,
        default=SPATIAL_PATH,
        help="Spatial transcriptomics 10x .h5 path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
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

    import scanpy as sc
    import split

    ref_path = args.ref_path
    spatial_path = args.spatial_path
    out_dir = args.out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    log("Loading single-cell reference ...")
    reference = sc.read_h5ad(ref_path)
    reference.var_names_make_unique()
    if args.cell_type_col not in reference.obs:
        raise KeyError(f"Reference AnnData is missing obs[{args.cell_type_col!r}].")
    reference.obs[args.cell_type_col] = reference.obs[args.cell_type_col].astype(str)

    log("Loading spatial Xenium count matrix ...")
    spatial = sc.read_10x_h5(spatial_path, gex_only=False)
    spatial.var_names_make_unique()

    common_genes = spatial.var_names.intersection(reference.var_names)
    if len(common_genes) == 0:
        raise ValueError("No shared genes between spatial data and reference.")
    log(
        "Shapes before gene intersection: "
        f"spatial={spatial.shape}, reference={reference.shape}, common_genes={len(common_genes)}"
    )

    spatial = spatial[:, common_genes].copy()
    reference = reference[:, common_genes].copy()
    log(f"Shapes after gene intersection: spatial={spatial.shape}, reference={reference.shape}")

    log("Running rctd-py followed by SPLIT purification ...")
    purified, split_rctd, rctd_py_result = split.run_rctd_py_and_split(
        spatial=spatial,
        reference_adata=reference,
        cell_type_col=args.cell_type_col,
        mode="doublet",
        min_weight=args.min_weight,
        purify_kwargs={"run_in_chunks": True, "chunk_size": args.chunk_size},
    )

    purified_path = out_dir / "xenium_split_purified.h5ad"
    metadata_path = out_dir / "split_cell_metadata.parquet"
    weights_path = out_dir / "rctd_deconvolution_weights.parquet"
    results_path = out_dir / "split_rctd_results.parquet"
    summary_path = out_dir / "run_summary.json"

    log(f"Writing purified AnnData to {purified_path} ...")
    purified.write_h5ad(purified_path, compression="gzip")

    log("Writing metadata and deconvolution tables ...")
    purified.obs.to_parquet(metadata_path)
    split_rctd.weights.to_parquet(weights_path)
    split_rctd.results_df.to_parquet(results_path)

    summary = {
        "reference_path": str(ref_path),
        "spatial_path": str(spatial_path),
        "output_h5ad": str(purified_path),
        "metadata_parquet": str(metadata_path),
        "weights_parquet": str(weights_path),
        "results_parquet": str(results_path),
        "cell_type_col": args.cell_type_col,
        "chunk_size": args.chunk_size,
        "min_weight": args.min_weight,
        "spatial_shape_after_intersection": list(spatial.shape),
        "reference_shape_after_intersection": list(reference.shape),
        "n_common_genes": int(len(common_genes)),
        "n_reference_cell_types": int(reference.obs[args.cell_type_col].nunique()),
        "reference_cell_types": sorted(reference.obs[args.cell_type_col].unique().tolist()),
        "purified_shape": list(purified.shape),
        "purification_status_counts": purified.obs["purification_status"].value_counts().to_dict(),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary, indent=2))
    log("Done.")


if __name__ == "__main__":
    main()
