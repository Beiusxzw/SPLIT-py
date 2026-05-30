"""Python implementation of the SPLIT spatial purification pipeline.

The public API mirrors the R package where practical, while using AnnData and
Scanpy conventions for input/output.
"""

from .core import PurifyResult, purify, rctd_free_purify, split_cells
from .networks import (
    Neighborhood,
    build_spatial_network,
    build_transcriptomics_network,
    compute_neighborhood,
)
from .postprocess import (
    RCTDResult,
    compute_alternative_annotations,
    convert_rctd_result_to_purify_input,
    reconstruct_rctd_from_rctdpy,
    run_post_process_rctd,
)
from .residuals import build_reassignment_operator, reassign_residual_counts
from .rctd_py import (
    rctd_py_result_to_rctd_result,
    run_rctd_py,
    run_rctd_py_and_split,
)

__all__ = [
    "Neighborhood",
    "PurifyResult",
    "RCTDResult",
    "build_reassignment_operator",
    "build_spatial_network",
    "build_transcriptomics_network",
    "compute_alternative_annotations",
    "compute_neighborhood",
    "convert_rctd_result_to_purify_input",
    "purify",
    "rctd_free_purify",
    "reassign_residual_counts",
    "reconstruct_rctd_from_rctdpy",
    "rctd_py_result_to_rctd_result",
    "run_post_process_rctd",
    "run_rctd_py",
    "run_rctd_py_and_split",
    "split_cells",
]
