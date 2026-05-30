from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from ._matrix import align_series, as_csc_matrix, as_csr_matrix, ensure_dataframe, matrix_from_anndata


MatrixLike = Union[np.ndarray, sparse.spmatrix, pd.DataFrame]


@dataclass
class PurifyResult:
    purified_counts: sparse.csc_matrix
    cell_meta: pd.DataFrame
    counts: sparse.csc_matrix
    genes: pd.Index
    cells: pd.Index

    def to_anndata(self) -> ad.AnnData:
        """Return an AnnData object with raw counts in X and purified counts in a layer."""
        out = ad.AnnData(
            X=self.counts.T.tocsr(),
            obs=self.cell_meta.copy(),
            var=pd.DataFrame(index=self.genes.copy()),
        )
        out.layers["purified_counts"] = self.purified_counts.T.tocsr()
        return out


def _counts_with_names(counts: Union[ad.AnnData, MatrixLike], layer: Optional[str]):
    if isinstance(counts, ad.AnnData):
        return matrix_from_anndata(counts, layer=layer)
    if not isinstance(counts, pd.DataFrame):
        raise TypeError("Matrix counts must be a pandas DataFrame, or pass an AnnData object.")
    return as_csc_matrix(counts.to_numpy()), counts.index.copy(), counts.columns.copy()


def rctd_free_purify(
    counts: Union[ad.AnnData, pd.DataFrame],
    deconvolution_weights: pd.DataFrame,
    reference: pd.DataFrame,
    primary_cell_type: Optional[Union[pd.Series, Sequence[str]]] = None,
    cells_to_purify: Optional[Sequence[str]] = None,
    *,
    layer: Optional[str] = None,
    run_in_chunks: bool = True,
    chunk_size: int = 50_000,
    require_sumup_to_one: bool = True,
    output: str = "anndata",
) -> Union[ad.AnnData, PurifyResult]:
    """Purify expression counts using SPLIT's method-agnostic formula.

    Parameters follow the R implementation. `counts` is genes x cells when a
    DataFrame is supplied; AnnData inputs use cells x genes and return AnnData.
    `reference` must be cell types x genes.
    """
    raw_counts, gene_names, cell_names = _counts_with_names(counts, layer)
    weights = ensure_dataframe(deconvolution_weights)
    ref = ensure_dataframe(reference)

    shared_cells = cell_names.intersection(weights.index)
    shared_genes = gene_names.intersection(ref.columns)
    if len(shared_cells) == 0 or len(shared_genes) == 0:
        raise ValueError("No shared cells or genes between counts, weights, and reference.")

    cell_pos = cell_names.get_indexer(shared_cells)
    gene_pos = gene_names.get_indexer(shared_genes)
    counts_aligned = raw_counts[gene_pos, :][:, cell_pos].tocsc()
    weights = weights.loc[shared_cells]

    if primary_cell_type is None:
        warnings.warn(
            "`primary_cell_type` not provided; assigning max-weight cell type.",
            RuntimeWarning,
            stacklevel=2,
        )
        primary = weights.idxmax(axis=1)
    else:
        primary = align_series(primary_cell_type, shared_cells, "primary_cell_type").astype(str)

    weight_types = pd.Index(sorted(weights.columns.astype(str)))
    ref_types = pd.Index(sorted(ref.index.astype(str)))
    if not weight_types.equals(ref_types):
        raise ValueError("Reference and deconvolution cell types do not match.")
    weights.columns = weights.columns.astype(str)
    ref.index = ref.index.astype(str)
    weights = weights.loc[:, weight_types]
    ref = ref.loc[weight_types, shared_genes].astype(float)

    weights_mat = as_csr_matrix(weights.to_numpy(dtype=float))
    row_sums = np.asarray(weights_mat.sum(axis=1)).ravel()
    if np.any((row_sums < 1 - 1e-8) | (row_sums > 1 + 1e-8)):
        warnings.warn("Some deconvolution weights do not sum to 1.", RuntimeWarning, stacklevel=2)
        if require_sumup_to_one:
            safe = row_sums.copy()
            safe[safe == 0] = 1
            weights_mat = sparse.diags(1 / safe) @ weights_mat

    if cells_to_purify is not None:
        cells_to_purify = pd.Index(cells_to_purify)
        keep_raw = shared_cells.difference(cells_to_purify)
        if len(keep_raw):
            weights_lil = weights_mat.tolil()
            rows = shared_cells.get_indexer(keep_raw)
            type_cols = weight_types.get_indexer(primary.loc[keep_raw])
            valid = type_cols >= 0
            weights_lil[rows[valid], :] = 0
            for r, c in zip(rows[valid], type_cols[valid]):
                weights_lil[r, c] = 1.0
            weights_mat = weights_lil.tocsr()

    primary_cols = weight_types.get_indexer(primary.loc[shared_cells])
    if np.any(primary_cols < 0):
        bad = sorted(set(primary.iloc[np.where(primary_cols < 0)[0]]))
        raise ValueError(f"Primary cell types absent from weights/reference: {bad}")

    primary_type_weight = np.asarray(weights_mat[np.arange(len(shared_cells)), primary_cols]).ravel()
    n_celltypes_per_cell = np.asarray((weights_mat > 0).sum(axis=1)).ravel()
    n_celltypes_per_cell[n_celltypes_per_cell == 0] = 1
    ref_mat = ref.to_numpy(dtype=float)
    epsilon = 1e-10

    rows_all, cols_all, vals_all = [], [], []
    blocks = (
        [np.arange(len(shared_cells))]
        if not run_in_chunks
        else [np.arange(i, min(i + chunk_size, len(shared_cells))) for i in range(0, len(shared_cells), chunk_size)]
    )
    counts_csr = counts_aligned.tocsr()
    for idx in blocks:
        denominator = weights_mat[idx, :].toarray() @ ref_mat + epsilon
        numerator = (
            primary_type_weight[idx, None] * ref_mat[primary_cols[idx], :]
            + epsilon / n_celltypes_per_cell[idx, None]
        )
        ratio_t = (numerator / denominator).T
        corrected = counts_csr[:, idx].multiply(ratio_t)
        coo = corrected.tocoo()
        if coo.nnz:
            rows_all.append(coo.row)
            cols_all.append(idx[coo.col])
            vals_all.append(coo.data)

    if rows_all:
        purified = sparse.csc_matrix(
            (np.concatenate(vals_all), (np.concatenate(rows_all), np.concatenate(cols_all))),
            shape=counts_aligned.shape,
        )
    else:
        purified = sparse.csc_matrix(counts_aligned.shape)

    n_cell_types = np.asarray((weights_mat > 0).sum(axis=1)).ravel()
    cell_meta = pd.DataFrame(
        {
            "primary_cell_type": primary.loc[shared_cells].to_numpy(),
            "w1_primary": primary_type_weight,
            "cell_id": shared_cells,
            "first_type": primary.loc[shared_cells].to_numpy(),
            "purification_status": np.where(n_cell_types > 1, "purified", "raw"),
            "n_cell_types": n_cell_types,
        },
        index=shared_cells,
    )

    result = PurifyResult(purified, cell_meta, counts_aligned, shared_genes, shared_cells)
    return result if output == "result" else result.to_anndata()


def purify(counts, *, rctd=None, primary_cell_type=None, deconvolution_weights=None, reference=None, **kwargs):
    if rctd is not None:
        from .postprocess import convert_rctd_result_to_purify_input

        converted = convert_rctd_result_to_purify_input(rctd)
        primary_cell_type = converted["primary_cell_type"]
        deconvolution_weights = converted["deconvolution_weights"]
        reference = converted["reference"].T
    missing = [
        name
        for name, value in {
            "primary_cell_type": primary_cell_type,
            "deconvolution_weights": deconvolution_weights,
            "reference": reference,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError("For RCTD-free purification, missing: " + ", ".join(missing))
    return rctd_free_purify(
        counts,
        primary_cell_type=primary_cell_type,
        deconvolution_weights=deconvolution_weights,
        reference=reference,
        **kwargs,
    )


def split_cells(counts, *, output: str = "anndata", **purify_kwargs):
    purified = purify(counts, output="result", **purify_kwargs)
    residual = (purified.counts - purified.purified_counts).tocoo()
    if residual.nnz:
        residual.data[residual.data < 0] = 0
        residual.eliminate_zeros()
    residual = residual.tocsc()

    first = purified.purified_counts.copy()
    second = residual
    split_counts = sparse.hstack([first, second], format="csc")
    meta1 = purified.cell_meta.copy()
    meta2 = purified.cell_meta.copy()
    meta1["cell_type"] = meta1["first_type"]
    meta1["decomposition_order"] = "first"
    meta2["cell_type"] = meta2.get("second_type", pd.Series(index=meta2.index, dtype=object))
    meta2.loc[meta2["purification_status"] == "raw", "purification_status"] = "null"
    meta2["decomposition_order"] = "second"
    meta1.index = meta1.index.astype(str) + "_1"
    meta2.index = meta2.index.astype(str) + "_2"
    out = PurifyResult(
        split_counts,
        pd.concat([meta1, meta2], axis=0),
        split_counts,
        purified.genes,
        pd.Index(list(meta1.index) + list(meta2.index)),
    )
    return out if output == "result" else out.to_anndata()
