import numpy as np
import pandas as pd
import anndata as ad

from split import rctd_py_result_to_rctd_result, purify


class FakeRctdPyResult:
    pixel_mask = np.array([True, True, False])
    cell_type_names = np.array(["A", "B"])
    weights = np.array([[1.0, 0.0], [0.6, 0.4]])
    weights_doublet = np.array([[1.0, 0.0], [0.6, 0.4]])
    spot_class = np.array([1, 2])
    first_type = np.array([0, 0])
    second_type = np.array([-1, 1])
    first_class = np.array([False, False])
    second_class = np.array([False, False])
    min_score = np.array([0.0, 0.0])
    singlet_score = np.array([0.0, 0.0])


class FakeReference:
    profiles = np.array([[1.0, 0.0], [0.0, 1.0]])
    gene_names = np.array(["g1", "g2"])
    cell_type_names = np.array(["A", "B"])


def test_rctd_py_result_converts_and_purifies_anndata():
    spatial = ad.AnnData(
        X=np.array([[10, 0], [5, 5], [1, 1]], dtype=float),
        obs=pd.DataFrame(index=["c1", "c2", "c3"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )

    split_rctd = rctd_py_result_to_rctd_result(
        FakeRctdPyResult(),
        spatial=spatial,
        reference=FakeReference(),
        min_weight=0.05,
    )
    purified = purify(spatial, rctd=split_rctd)

    assert list(split_rctd.results_df.index) == ["c1", "c2"]
    assert split_rctd.results_df.loc["c2", "spot_class"] == "doublet_certain"
    assert purified.shape == (2, 2)
    assert "purified_counts" in purified.layers
    assert purified.obs.loc["c2", "purification_status"] == "purified"

