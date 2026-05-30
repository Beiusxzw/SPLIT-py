import numpy as np
import pandas as pd

from split import (
    RCTDResult,
    convert_rctd_result_to_purify_input,
    rctd_free_purify,
    run_post_process_rctd,
)


def test_rctd_free_purify_matches_split_formula():
    counts = pd.DataFrame(
        [[10, 5, 0], [0, 5, 10]],
        index=["g1", "g2"],
        columns=["c1", "c2", "c3"],
    )
    weights = pd.DataFrame(
        [[1, 0], [0.5, 0.5], [0, 1]],
        index=["c1", "c2", "c3"],
        columns=["A", "B"],
    )
    reference = pd.DataFrame(
        [[1, 0], [0, 1]],
        index=["A", "B"],
        columns=["g1", "g2"],
    )
    primary = pd.Series(["A", "A", "B"], index=["c1", "c2", "c3"])

    result = rctd_free_purify(
        counts,
        weights,
        reference,
        primary,
        output="result",
        run_in_chunks=True,
        chunk_size=2,
    )

    assert result.purified_counts.shape == counts.shape
    np.testing.assert_allclose(
        result.purified_counts.toarray(),
        np.array([[10, 5, 0], [0, 5e-10, 10]]),
        rtol=1e-6,
        atol=1e-12,
    )
    assert result.cell_meta.loc["c2", "purification_status"] == "purified"


def test_rctd_conversion_preserves_pipeline_inputs():
    results_df = pd.DataFrame(
        {
            "spot_class": ["singlet", "doublet_certain"],
            "first_type": ["A", "A"],
            "second_type": [np.nan, "B"],
            "weight_first_type": [1.0, 0.7],
            "weight_second_type": [np.nan, 0.3],
        },
        index=["c1", "c2"],
    )
    rctd = RCTDResult(
        results_df=results_df,
        weights=pd.DataFrame([[1, 0], [0.7, 0.3]], index=["c1", "c2"], columns=["A", "B"]),
        weights_doublet=pd.DataFrame([[1, 0], [0.7, 0.3]], index=["c1", "c2"], columns=["first_type", "second_type"]),
        reference=pd.DataFrame([[1, 0], [0, 1]], index=["g1", "g2"], columns=["A", "B"]),
    )

    converted = convert_rctd_result_to_purify_input(rctd)

    assert list(converted["primary_cell_type"]) == ["A", "A"]
    np.testing.assert_allclose(converted["deconvolution_weights"].loc["c2"], [0.7, 0.3])
    assert list(converted["reference"].index) == ["g1", "g2"]


def test_run_post_process_rctd_corrects_confident_singlets():
    rctd = RCTDResult(
        results_df=pd.DataFrame(
            {
                "spot_class": ["doublet_certain", "doublet_certain"],
                "first_type": ["B", "A"],
                "second_type": ["A", "B"],
                "min_score": [0, 0],
                "singlet_score": [0, 0],
            },
            index=["c1", "c2"],
        ),
        weights=pd.DataFrame([[1, 0], [0.6, 0.4]], index=["c1", "c2"], columns=["A", "B"]),
        weights_doublet=pd.DataFrame([[1, 0], [0.6, 0.4]], index=["c1", "c2"], columns=["first_type", "second_type"]),
        reference=pd.DataFrame([[1, 0], [0, 1]], index=["g1", "g2"], columns=["A", "B"]),
    )

    out = run_post_process_rctd(rctd, min_weight=0.5)

    assert out.results_df.loc["c1", "first_type"] == "A"
    assert pd.isna(out.results_df.loc["c1", "second_type"])
    assert out.results_df.loc["c1", "spot_class"] == "singlet"

