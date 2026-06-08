import numpy as np
import pandas as pd
import anndata as ad

from split import rctd_py_result_to_rctd_result, purify
from split.rctd_py import run_rctd_py


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


def test_run_rctd_py_forwards_device_default_and_override(monkeypatch):
    captured = {}

    def fake_run_rctd(spatial, reference, mode=None, **kwargs):
        captured["spatial_shape"] = spatial.shape
        captured["reference_cell_type_col"] = getattr(reference, "cell_type_col", None)
        captured["mode"] = mode
        captured["config_device"] = getattr(kwargs.get("config"), "device", None)
        return object()

    class FakeReference:
        def __init__(self, reference_adata, cell_type_col="cell_type", **kwargs):
            self.reference_adata = reference_adata
            self.cell_type_col = cell_type_col
            self.kwargs = kwargs

    class FakeRCTDConfig:
        def __init__(self, device="auto", class_df=None):
            self.device = device
            self.class_df = class_df

        def _replace(self, **changes):
            data = {"device": self.device, "class_df": self.class_df}
            data.update(changes)
            return FakeRCTDConfig(**data)

    monkeypatch.setattr("split.rctd_py._require_rctd_py", lambda: (FakeReference, fake_run_rctd))
    monkeypatch.setattr("split.rctd_py._require_rctd_config", lambda: FakeRCTDConfig)

    spatial = ad.AnnData(
        X=np.array([[1, 2]], dtype=float),
        obs=pd.DataFrame(index=["c1"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    reference = ad.AnnData(
        X=np.array([[3, 4]], dtype=float),
        obs=pd.DataFrame(index=["r1"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )

    run_rctd_py(spatial, reference)
    assert captured["config_device"] == "cuda:0"

    run_rctd_py(spatial, reference, device="cpu")
    assert captured["config_device"] == "cpu"

