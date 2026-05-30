from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .networks import Neighborhood


def build_reassignment_operator(
    spatial_network: Neighborhood,
    cells_with_residual_transcripts: Sequence[str],
    *,
    n_count: Optional[pd.Series] = None,
    self_keep: float = 0.0,
    weight_power: float = 1.0,
    eps: float = 0.0,
    neighbor_key: str = "second_type_neighbors_no_reject",
) -> sparse.csr_matrix:
    cell_ids = pd.Index(spatial_network.cell_id)
    loc = {c: i for i, c in enumerate(cell_ids)}
    n = len(cell_ids)
    if n_count is not None:
        n_count = pd.Series(n_count).reindex(cell_ids).fillna(0).clip(lower=0)
        n_weight = (n_count + eps) ** weight_power
    rows, cols, vals = [], [], []
    neighbor_lists = spatial_network.data.get(neighbor_key)

    for cid in cells_with_residual_transcripts:
        if cid not in loc:
            continue
        i = loc[cid]
        if neighbor_lists is not None:
            recv_idx = np.asarray(neighbor_lists[i], dtype=int)
        else:
            recv_idx = spatial_network.nn_idx[i, 1:]
            recv_idx = recv_idx[recv_idx >= 0]
        if len(recv_idx) == 0:
            continue
        if self_keep > 0:
            rows.append(i); cols.append(i); vals.append(self_keep)
        if n_count is not None:
            raw = n_weight.iloc[recv_idx].to_numpy(dtype=float)
            total = raw.sum()
            w = raw / total if total > 0 else np.repeat(1 / len(recv_idx), len(recv_idx))
        else:
            w = np.repeat(1 / len(recv_idx), len(recv_idx))
        rows.extend([i] * len(recv_idx))
        cols.extend(recv_idx.tolist())
        vals.extend((w * (1 - self_keep)).tolist())
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def reassign_residual_counts(
    raw_counts,
    corrected_counts,
    spatial_network: Neighborhood,
    purification_status: pd.Series,
    *,
    mode: str = "uniform",
    return_reassignment_operator: bool = False,
    **kwargs,
):
    raw = raw_counts if sparse.issparse(raw_counts) else sparse.csc_matrix(raw_counts)
    corrected = corrected_counts if sparse.issparse(corrected_counts) else sparse.csc_matrix(corrected_counts)
    purification_status = pd.Series(purification_status)
    cells = purification_status.index
    senders = purification_status.index[purification_status == "purified"]
    n_count = None
    if mode == "count_proportional":
        n_count = pd.Series(np.asarray(raw.sum(axis=0)).ravel(), index=cells)
    elif mode != "uniform":
        raise ValueError("mode must be 'uniform' or 'count_proportional'.")
    operator = build_reassignment_operator(
        spatial_network,
        senders,
        n_count=n_count,
        **kwargs,
    )
    residual = raw - corrected
    reassigned = residual @ operator
    out = corrected + reassigned
    return (out, operator) if return_reassignment_operator else out

