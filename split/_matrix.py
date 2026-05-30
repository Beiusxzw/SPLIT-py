from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def as_csc_matrix(x) -> sparse.csc_matrix:
    if sparse.issparse(x):
        return x.tocsc()
    return sparse.csc_matrix(np.asarray(x))


def as_csr_matrix(x) -> sparse.csr_matrix:
    if sparse.issparse(x):
        return x.tocsr()
    return sparse.csr_matrix(np.asarray(x))


def matrix_from_anndata(
    adata: ad.AnnData,
    *,
    layer: Optional[str] = None,
) -> Tuple[sparse.csc_matrix, pd.Index, pd.Index]:
    """Return counts as genes x cells plus matching gene and cell indexes."""
    x = adata.layers[layer] if layer is not None else adata.X
    return as_csc_matrix(x).T, adata.var_names.copy(), adata.obs_names.copy()


def dataframe_to_sparse(df: pd.DataFrame) -> sparse.csr_matrix:
    return sparse.csr_matrix(df.to_numpy(dtype=float))


def align_series(values, index: Sequence[str], name: str) -> pd.Series:
    if values is None:
        raise ValueError(f"`{name}` is required.")
    s = values if isinstance(values, pd.Series) else pd.Series(values)
    if s.index.equals(pd.RangeIndex(len(s))):
        if len(s) != len(index):
            raise ValueError(f"`{name}` must have length {len(index)}.")
        s.index = pd.Index(index)
    missing = pd.Index(index).difference(s.index)
    if len(missing):
        raise ValueError(f"`{name}` is missing {len(missing)} cell ids.")
    return s.loc[index]


def ensure_dataframe(x, *, index: Optional[Iterable[str]] = None, columns=None) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    return pd.DataFrame(x, index=index, columns=columns)

