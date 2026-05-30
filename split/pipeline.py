from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional


def run_split_rctd_py_pipeline(
    *,
    ref_path: str | Path,
    spatial_path: str | Path,
    out_dir: str | Path,
    cell_type_col: str = "Annotation",
    chunk_size: int = 50_000,
    min_weight: float = 0.01,
    log_fn: Optional[Callable[[str], None]] = print,
) -> dict[str, Any]:
    """Run rctd-py followed by SPLIT purification from file paths.

    Parameters
    ----------
    ref_path
        Single-cell reference `.h5ad` path.
    spatial_path
        Spatial transcriptomics 10x `.h5` path.
    out_dir
        Output directory for the purified AnnData and metadata tables.
    cell_type_col
        Reference `.obs` column used as rctd-py deconvolution cell types.
    chunk_size
        Cell chunk size for SPLIT purification.
    min_weight
        Minimum RCTD weight used during SPLIT post-processing.
    log_fn
        Optional logger callable. Set to `None` to silence progress messages.

    Returns
    -------
    dict
        Contains `purified`, `split_rctd`, `rctd_py_result`, `summary`, and
        output path entries.
    """
    import scanpy as sc

    from .rctd_py import run_rctd_py_and_split

    def log(message: str) -> None:
        if log_fn is not None:
            log_fn(message)

    ref_path = Path(ref_path)
    spatial_path = Path(spatial_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("Loading single-cell reference ...")
    reference = sc.read_h5ad(ref_path)
    reference.var_names_make_unique()
    if cell_type_col not in reference.obs:
        raise KeyError(f"Reference AnnData is missing obs[{cell_type_col!r}].")
    reference.obs[cell_type_col] = reference.obs[cell_type_col].astype(str)

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
    purified, split_rctd, rctd_py_result = run_rctd_py_and_split(
        spatial=spatial,
        reference_adata=reference,
        cell_type_col=cell_type_col,
        mode="doublet",
        min_weight=min_weight,
        purify_kwargs={"run_in_chunks": True, "chunk_size": chunk_size},
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
        "cell_type_col": cell_type_col,
        "chunk_size": chunk_size,
        "min_weight": min_weight,
        "spatial_shape_after_intersection": list(spatial.shape),
        "reference_shape_after_intersection": list(reference.shape),
        "n_common_genes": int(len(common_genes)),
        "n_reference_cell_types": int(reference.obs[cell_type_col].nunique()),
        "reference_cell_types": sorted(reference.obs[cell_type_col].unique().tolist()),
        "purified_shape": list(purified.shape),
        "purification_status_counts": purified.obs["purification_status"].value_counts().to_dict(),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary, indent=2))
    log("Done.")

    return {
        "purified": purified,
        "split_rctd": split_rctd,
        "rctd_py_result": rctd_py_result,
        "summary": summary,
        "purified_path": purified_path,
        "metadata_path": metadata_path,
        "weights_path": weights_path,
        "results_path": results_path,
        "summary_path": summary_path,
    }

