from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


@dataclass
class Neighborhood:
    nn_idx: np.ndarray
    nn_dist: np.ndarray
    cell_id: pd.Index
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"nn_idx": self.nn_idx, "nn_dist": self.nn_dist, "cell_id": self.cell_id, **self.data}


def _representation(
    adata: ad.AnnData,
    *,
    reduction: str = "pca",
    dims: Optional[Sequence[int]] = None,
    features: Optional[Sequence[str]] = None,
) -> np.ndarray:
    if features is not None:
        return np.asarray(adata[:, features].X.toarray() if hasattr(adata[:, features].X, "toarray") else adata[:, features].X)
    if reduction in {"spatial", "X_spatial"}:
        key = "spatial" if "spatial" in adata.obsm else "X_spatial"
        x = np.asarray(adata.obsm[key])
    else:
        key = reduction if reduction.startswith("X_") else f"X_{reduction}"
        if key not in adata.obsm and reduction == "pca":
            try:
                import scanpy as sc

                sc.pp.pca(adata)
            except Exception:
                matrix = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
                n_components = min(50, matrix.shape[0] - 1, matrix.shape[1])
                adata.obsm[key] = PCA(n_components=n_components).fit_transform(matrix)
        x = np.asarray(adata.obsm[key])
    if dims is not None:
        dims0 = np.asarray(dims) - 1 if min(dims) == 1 else np.asarray(dims)
        x = x[:, dims0]
    return x


def compute_neighborhood(
    adata: ad.AnnData,
    *,
    k_knn: int = 20,
    reduction: str = "pca",
    dims: Optional[Sequence[int]] = None,
    features: Optional[Sequence[str]] = None,
    prune: bool = False,
    rad_pruning: float = np.inf,
) -> Neighborhood:
    x = _representation(adata, reduction=reduction, dims=dims, features=features)
    n_neighbors = min(k_knn + 1, x.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(x)
    dist, idx = nn.kneighbors(x)
    if prune and np.isfinite(rad_pruning):
        idx = idx.copy()
        dist = dist.copy()
        idx[dist > rad_pruning] = -1
        dist[dist > rad_pruning] = np.nan
    return Neighborhood(idx, dist, adata.obs_names.copy())


def build_spatial_network(adata: ad.AnnData, *, dims=(0, 1), rad_pruning=30, **kwargs) -> Neighborhood:
    return compute_neighborhood(adata, reduction="spatial", dims=dims, prune=True, rad_pruning=rad_pruning, **kwargs)


def build_transcriptomics_network(adata: ad.AnnData, *, reduction="pca", **kwargs) -> Neighborhood:
    return compute_neighborhood(adata, reduction=reduction, **kwargs)


def add_obs_to_neighborhood(neighborhood: Neighborhood, obs: pd.DataFrame) -> Neighborhood:
    aligned = obs.reindex(neighborhood.cell_id)
    flat = np.where(neighborhood.nn_idx >= 0, neighborhood.nn_idx, 0).ravel()
    valid = neighborhood.nn_idx >= 0
    for col in aligned.columns:
        vals = aligned.iloc[flat][col].to_numpy().reshape(neighborhood.nn_idx.shape)
        vals = vals.astype(object)
        vals[~valid] = np.nan
        neighborhood.data[col] = vals
    return neighborhood
