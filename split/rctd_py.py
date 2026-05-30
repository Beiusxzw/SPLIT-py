from __future__ import annotations

from typing import Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd

from .core import purify
from .postprocess import RCTDResult, run_post_process_rctd


def _require_rctd_py():
    try:
        from rctd import Reference, run_rctd
    except ImportError as exc:
        raise ImportError(
            "rctd-py is required for this function. Install with "
            "`pip install 'split-st[rctd]'` or `pip install rctd-py`."
        ) from exc
    return Reference, run_rctd


def _require_rctd_config():
    try:
        from rctd._types import RCTDConfig
    except ImportError:
        return None
    return RCTDConfig


def _result_attr(result, name, default=None):
    if hasattr(result, name):
        return getattr(result, name)
    if isinstance(result, dict):
        return result.get(name, default)
    return default


def _cell_type_names(result, reference=None) -> pd.Index:
    names = _result_attr(result, "cell_type_names")
    if names is None and reference is not None:
        names = getattr(reference, "cell_type_names", None)
    if names is None:
        raise ValueError("Could not find `cell_type_names` on the rctd-py result or reference.")
    return pd.Index([x.decode() if isinstance(x, bytes) else str(x) for x in names])


def _reference_profiles(reference) -> pd.DataFrame:
    profiles = getattr(reference, "profiles", None)
    gene_names = getattr(reference, "gene_names", None)
    cell_type_names = getattr(reference, "cell_type_names", None)
    if profiles is None or gene_names is None or cell_type_names is None:
        raise ValueError("rctd-py Reference must expose profiles, gene_names, and cell_type_names.")
    genes = [x.decode() if isinstance(x, bytes) else str(x) for x in gene_names]
    cell_types = [x.decode() if isinstance(x, bytes) else str(x) for x in cell_type_names]
    arr = np.asarray(profiles)
    if arr.shape == (len(cell_types), len(genes)):
        arr = arr.T
    return pd.DataFrame(arr, index=genes, columns=cell_types)


def rctd_py_result_to_rctd_result(
    result,
    *,
    spatial: ad.AnnData,
    reference=None,
    class_df: Optional[pd.DataFrame] = None,
    min_weight: float = 0.01,
    post_process: bool = True,
) -> RCTDResult:
    """Convert an in-memory rctd-py result into SPLIT's lightweight RCTDResult."""
    pixel_mask = _result_attr(result, "pixel_mask")
    if pixel_mask is None:
        cell_ids = spatial.obs_names
    else:
        cell_ids = spatial.obs_names[np.asarray(pixel_mask, dtype=bool)]
    cell_ids = pd.Index(cell_ids.astype(str))
    cell_type_names = _cell_type_names(result, reference=reference)

    weights = pd.DataFrame(
        np.asarray(_result_attr(result, "weights"), dtype=float),
        index=cell_ids,
        columns=cell_type_names,
    )

    weights_doublet = _result_attr(result, "weights_doublet")
    if weights_doublet is None:
        weights_doublet = np.column_stack([weights.max(axis=1).to_numpy(), np.zeros(len(weights))])
    weights_doublet = pd.DataFrame(
        np.asarray(weights_doublet, dtype=float)[:, :2],
        index=cell_ids,
        columns=["first_type", "second_type"],
    )

    first_type = np.asarray(_result_attr(result, "first_type", weights.to_numpy().argmax(axis=1)))
    second_type = np.asarray(_result_attr(result, "second_type", np.full(len(cell_ids), -1)))
    first_names = np.where(first_type >= 0, cell_type_names.to_numpy()[first_type.astype(int)], None)
    second_names = np.where(second_type >= 0, cell_type_names.to_numpy()[second_type.astype(int)], None)

    spot_class = _result_attr(result, "spot_class", np.repeat(1, len(cell_ids)))
    spot_map = {
        0: "reject",
        1: "singlet",
        2: "doublet_certain",
        3: "doublet_uncertain",
        "reject": "reject",
        "singlet": "singlet",
        "doublet_certain": "doublet_certain",
        "doublet_uncertain": "doublet_uncertain",
    }
    spot_class = pd.Series(spot_class).map(spot_map).fillna(pd.Series(spot_class).astype(str)).to_numpy()

    results_df = pd.DataFrame(
        {
            "spot_class": spot_class,
            "first_type": first_names,
            "second_type": second_names,
            "first_class": _result_attr(result, "first_class", np.repeat(False, len(cell_ids))),
            "second_class": _result_attr(result, "second_class", np.repeat(False, len(cell_ids))),
            "min_score": _result_attr(result, "min_score", np.repeat(np.nan, len(cell_ids))),
            "singlet_score": _result_attr(result, "singlet_score", np.repeat(np.nan, len(cell_ids))),
        },
        index=cell_ids,
    )

    if reference is None:
        ref_profiles = pd.DataFrame(index=spatial.var_names, columns=cell_type_names, dtype=float)
    else:
        ref_profiles = _reference_profiles(reference)

    coords = pd.DataFrame(index=spatial.obs_names.astype(str))
    if "spatial" in spatial.obsm:
        coords = pd.DataFrame(spatial.obsm["spatial"], index=spatial.obs_names.astype(str))
        if coords.shape[1] >= 2:
            coords = coords.iloc[:, :2]
            coords.columns = ["x", "y"]

    if class_df is None:
        first_class_name = _result_attr(result, "first_class_name")
        if first_class_name is not None:
            class_df = pd.DataFrame({"class": cell_type_names}, index=cell_type_names)
        else:
            class_df = pd.DataFrame({"class": cell_type_names}, index=cell_type_names)

    out = RCTDResult(
        results_df=results_df,
        weights=weights,
        weights_doublet=weights_doublet,
        reference=ref_profiles,
        coords=coords,
        class_df=class_df,
    )
    return run_post_process_rctd(out, min_weight=min_weight, lite=True) if post_process else out


def run_rctd_py(
    spatial: ad.AnnData,
    reference_adata: ad.AnnData,
    *,
    cell_type_col: str = "cell_type",
    mode: str = "doublet",
    class_df: Optional[dict | pd.DataFrame] = None,
    config=None,
    batch_size: Optional[int] = None,
    sigma_override: Optional[float] = None,
    **reference_kwargs,
):
    """Run p-gueguen/rctd-py and return `(result, reference)`.

    This is a thin wrapper around `rctd.Reference` and `rctd.run_rctd`.
    """
    Reference, run_rctd = _require_rctd_py()
    reference = Reference(reference_adata, cell_type_col=cell_type_col, **reference_kwargs)

    if config is None and class_df is not None:
        RCTDConfig = _require_rctd_config()
        if RCTDConfig is not None:
            if isinstance(class_df, pd.DataFrame):
                class_map = class_df["class"].to_dict()
            else:
                class_map = dict(class_df)
            config = RCTDConfig(class_df=class_map)

    kwargs = {}
    if config is not None:
        kwargs["config"] = config
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if sigma_override is not None:
        kwargs["sigma_override"] = sigma_override
    result = run_rctd(spatial, reference, mode=mode, **kwargs)
    return result, reference


def run_rctd_py_and_split(
    spatial: ad.AnnData,
    reference_adata: ad.AnnData,
    *,
    cell_type_col: str = "cell_type",
    mode: str = "doublet",
    class_df: Optional[dict | pd.DataFrame] = None,
    min_weight: float = 0.01,
    purify_kwargs: Optional[dict] = None,
    **rctd_kwargs,
) -> tuple[ad.AnnData, RCTDResult, object]:
    """Run rctd-py followed by SPLIT purification.

    Returns `(purified_adata, split_rctd_result, rctd_py_result)`.
    """
    rctd_py_result, reference = run_rctd_py(
        spatial,
        reference_adata,
        cell_type_col=cell_type_col,
        mode=mode,
        class_df=class_df,
        **rctd_kwargs,
    )
    if isinstance(class_df, dict):
        class_frame = pd.DataFrame({"class": class_df})
    else:
        class_frame = class_df
    split_rctd = rctd_py_result_to_rctd_result(
        rctd_py_result,
        spatial=spatial,
        reference=reference,
        class_df=class_frame,
        min_weight=min_weight,
        post_process=True,
    )
    purified = purify(spatial, rctd=split_rctd, **(purify_kwargs or {}))
    return purified, split_rctd, rctd_py_result

