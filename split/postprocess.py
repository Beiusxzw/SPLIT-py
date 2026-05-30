from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional
import warnings

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class RCTDResult:
    """Lightweight Python container for RCTD/rctd-py outputs used by SPLIT."""

    results_df: pd.DataFrame
    weights: pd.DataFrame
    weights_doublet: pd.DataFrame
    reference: pd.DataFrame  # genes x cell types, matching the R RCTD slot
    coords: pd.DataFrame = field(default_factory=pd.DataFrame)
    class_df: Optional[pd.DataFrame] = None
    results_df_old: Optional[pd.DataFrame] = None
    config: dict = field(default_factory=lambda: {"RCTDmode": "doublet"})


def _entropy_rows(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, None)
    sums = x.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    p = x / sums
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log(p), 0)
    return -(p * logp).sum(axis=1)


def run_post_process_rctd(
    rctd: RCTDResult,
    *,
    min_weight: float = 0.05,
    nfeature_doublet_threshold: float = 0.5,
    n_feature: Optional[pd.Series] = None,
    n_count: Optional[pd.Series] = None,
    lite: bool = True,
) -> RCTDResult:
    """Apply SPLIT's RCTD post-processing pipeline to a Python RCTDResult."""
    df = rctd.results_df.copy()
    if "scond_type" in df.columns and "second_type" not in df.columns:
        df["second_type"] = df["scond_type"]

    len_mat = (rctd.weights > min_weight).sum(axis=1).reindex(df.index).fillna(0).astype(int)
    confident = len_mat == 1
    no_type = len_mat == 0
    rejects = df.get("spot_class", pd.Series(index=df.index, dtype=object)).astype(str) == "reject"
    argmax_weight = rctd.weights.idxmax(axis=1).reindex(df.index)

    df.loc[confident, "first_type"] = argmax_weight.loc[confident]
    df.loc[no_type, "first_type"] = np.nan
    df.loc[confident | no_type, "second_type"] = np.nan
    df.loc[confident & ~rejects, "spot_class"] = "singlet"
    df.loc[no_type, "spot_class"] = "reject"
    df["spot_class"] = pd.Categorical(
        df["spot_class"],
        categories=["reject", "doublet_uncertain", "doublet_certain", "singlet"],
        ordered=True,
    )

    df["max_doublet_weight"] = rctd.weights_doublet.max(axis=1).reindex(df.index).to_numpy()
    df["n_candidates"] = len_mat.to_numpy()
    df["rctd_weights_entropy"] = _entropy_rows(rctd.weights.reindex(df.index).fillna(0).to_numpy())
    df["weight_first_type"] = rctd.weights_doublet.iloc[:, 0].reindex(df.index).to_numpy()
    df["weight_second_type"] = rctd.weights_doublet.iloc[:, 1].reindex(df.index).to_numpy()

    if rctd.class_df is not None and "class" in rctd.class_df:
        classes = rctd.class_df["class"]
        df["first_type_class"] = df["first_type"].map(classes)
        df["second_type_class"] = df["second_type"].map(classes)
        df["same_class"] = df["first_type_class"].eq(df["second_type_class"])

    if not rctd.coords.empty:
        shared = df.index.intersection(rctd.coords.index)
        for col in ["x", "y"]:
            if col in rctd.coords:
                df.loc[shared, col] = rctd.coords.loc[shared, col]

    if not lite:
        if n_feature is not None:
            df["nFeature"] = pd.Series(n_feature).reindex(df.index).to_numpy()
        if n_count is not None:
            df["nCount"] = pd.Series(n_count).reindex(df.index).to_numpy()
        if "score_diff" in df and "nFeature" in df:
            df["score_diff_normalized"] = df["score_diff"] / df["nFeature"].replace(0, np.nan)
            df["is_singlet_in_normalized_thresh"] = (
                df["score_diff_normalized"] < nfeature_doublet_threshold
            )

    df = compute_alternative_annotations(rctd, results_df=df)
    rctd.results_df_old = rctd.results_df.copy()
    rctd.results_df = df
    return rctd


def compute_alternative_annotations(rctd: RCTDResult, *, results_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = rctd.results_df.copy() if results_df is None else results_df.copy()
    df["annot_max_weight"] = rctd.weights.idxmax(axis=1).reindex(df.index).to_numpy()
    choose_first = df["weight_second_type"].isna() | (df["weight_first_type"] > df["weight_second_type"])
    df["annot_max_doublet_weight"] = np.where(choose_first, df["first_type"], df["second_type"])
    df["w1_larger_w2"] = df["first_type"].eq(df["annot_max_doublet_weight"])
    return df


def res_df_2_ijx(rctd: RCTDResult, cell_types=None) -> pd.DataFrame:
    df = rctd.results_df.copy()
    cell_types = pd.Index(cell_types if cell_types is not None else rctd.weights.columns)
    pieces = []
    certain = df["spot_class"].astype(str).isin(["singlet", "doublet_certain"])
    for col_type, col_weight in [("first_type", "weight_first_type"), ("second_type", "weight_second_type")]:
        tmp = pd.DataFrame({"cell": df.index, "cell_type": df[col_type], "weight": df[col_weight]})
        tmp = tmp[certain.to_numpy() & tmp["cell_type"].notna()]
        pieces.append(tmp)
    uncertain_cells = df.index[df["spot_class"].astype(str) == "doublet_uncertain"]
    if len(uncertain_cells):
        w = rctd.weights.loc[uncertain_cells, cell_types].stack().reset_index()
        w.columns = ["cell", "cell_type", "weight"]
        pieces.append(w)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["cell", "cell_type", "weight"])


def build_sparse_from_ijx(ijx: pd.DataFrame, *, row_order=None, col_order=None) -> pd.DataFrame:
    row_order = pd.Index(row_order if row_order is not None else pd.unique(ijx["cell"]))
    col_order = pd.Index(col_order if col_order is not None else pd.unique(ijx["cell_type"]))
    out = pd.DataFrame(0.0, index=row_order, columns=col_order)
    if len(ijx):
        out.values[
            row_order.get_indexer(ijx["cell"]),
            col_order.get_indexer(ijx["cell_type"]),
        ] = ijx["weight"].to_numpy(dtype=float)
    return out


def convert_rctd_result_to_purify_input(rctd: RCTDResult) -> dict:
    ijx = res_df_2_ijx(rctd, cell_types=rctd.weights.columns)
    weights = build_sparse_from_ijx(ijx, row_order=rctd.results_df.index, col_order=rctd.weights.columns)
    primary = rctd.results_df["first_type"].copy()
    primary.index = rctd.results_df.index
    return {
        "primary_cell_type": primary,
        "deconvolution_weights": weights,
        "reference": rctd.reference.copy(),  # genes x cell types, as in R conversion
    }


def reconstruct_rctd_from_rctdpy(
    save_dir,
    *,
    min_weight: float = 0.01,
    class_df: Optional[pd.DataFrame] = None,
    spot_class_map: Optional[Mapping[str, str]] = None,
    post_process: bool = True,
) -> RCTDResult:
    save_dir = Path(save_dir)
    if not save_dir.exists() or not save_dir.is_dir():
        raise ValueError("`save_dir` must be an existing directory.")
    if not isinstance(min_weight, (int, float)):
        raise TypeError("`min_weight` must be numeric.")
    spot_class_map = spot_class_map or {
        "0": "reject",
        "1": "singlet",
        "2": "doublet_certain",
        "3": "doublet_uncertain",
    }

    cell_ids = pd.read_parquet(save_dir / "cell_ids.parquet")["cell_id"].astype(str)
    weights = pd.read_parquet(save_dir / "weights.parquet")
    weights_doublet = pd.read_parquet(save_dir / "weights_doublet.parquet")
    spot_results = pd.read_parquet(save_dir / "spot_results.parquet")
    metadata = pd.read_parquet(save_dir / "metadata.parquet")
    cell_type_names = metadata["cell_type_names"].astype(str).tolist()

    mapped = spot_results["spot_class"].astype(str).map(spot_class_map)
    if mapped.isna().any():
        bad = sorted(spot_results.loc[mapped.isna(), "spot_class"].unique())
        raise ValueError("Unmapped spot_class integer(s): " + ", ".join(map(str, bad)))

    results_df = pd.DataFrame(
        {
            "spot_class": pd.Categorical(
                mapped,
                categories=["singlet", "doublet_certain", "doublet_uncertain", "reject"],
            ),
            "first_type": pd.Categorical(spot_results["first_type_name"], categories=cell_type_names),
            "second_type": pd.Categorical(spot_results["second_type_name"], categories=cell_type_names),
            "first_class": np.where(spot_results["first_class"], "doublet_certain", "singlet"),
            "second_class": np.where(spot_results["second_class"], "doublet_certain", "singlet"),
            "min_score": spot_results["min_score"].to_numpy(),
            "singlet_score": spot_results["singlet_score"].to_numpy(),
        },
        index=pd.Index(cell_ids, name=None),
    )

    weights = pd.DataFrame(weights.to_numpy(dtype=float), index=cell_ids, columns=cell_type_names)
    weights_doublet = pd.DataFrame(
        weights_doublet.loc[:, ["w_1", "w_2"]].to_numpy(dtype=float),
        index=cell_ids,
        columns=["first_type", "second_type"],
    )

    with h5py.File(save_dir / "reference_profiles.h5", "r") as h5:
        profiles = h5["profiles"][()]
        gene_names = [x.decode() if isinstance(x, bytes) else str(x) for x in h5["gene_names"][()]]
        ref_ct = [x.decode() if isinstance(x, bytes) else str(x) for x in h5["cell_type_names"][()]]
    reference = pd.DataFrame(profiles.T, index=gene_names, columns=ref_ct)

    if class_df is None:
        class_df = pd.DataFrame({"class": cell_type_names}, index=cell_type_names)
    else:
        missing = set(cell_type_names).difference(class_df.index)
        if missing:
            raise ValueError("class_df is missing rows for: " + ", ".join(sorted(missing)))

    out = RCTDResult(results_df, weights, weights_doublet, reference, class_df=class_df)
    if post_process:
        try:
            out = run_post_process_rctd(out, min_weight=min_weight, lite=True)
        except Exception as exc:
            warnings.warn(f"RCTD post-processing failed: {exc}", RuntimeWarning, stacklevel=2)
            raise
    return out

