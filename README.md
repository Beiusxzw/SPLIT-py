<!-- badges: start -->
[![R-CMD-check](https://github.com/BDSC-tds/SPLIT/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/BDSC-tds/SPLIT/actions/workflows/R-CMD-check.yaml)
[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41592--026--03089--8-brightgreen)](https://doi.org/10.1038/s41592-026-03089-8)
[![Version](https://img.shields.io/badge/version-0.2.0-blue)](https://github.com/bdsc-tds/SPLIT/releases/tag/v0.2.0)
<!-- badges: end -->

# SPLIT-py: Spatial Purification of Layered Intracellular Transcripts

**Python implementation note**
>
This repository contains an experimental Python/Scanpy rewrite in the SPLIT R package (https://github.com/bdsc-tds/SPLIT). The Python implementation was rewritten by AI using OpenAI Codex from the R package logic, with compatibility helpers for AnnData, Scanpy, and [p-gueguen/rctd-py](https://github.com/p-gueguen/rctd-py). Treat the R implementation as the canonical reference unless you have validated the Python rewrite for your own workflow.

## Overview

SPLIT-py provides a Python/Scanpy-oriented wrapper around the SPLIT
purification logic. It can run [rctd-py](https://github.com/p-gueguen/rctd-py)
on a spatial AnnData/10x matrix, convert the rctd-py output into SPLIT inputs,
and write a purified AnnData object plus metadata tables.

## Installation

From the repository root:

```bash
pip install ".[rctd]"
```

## Usage

Run the command-line pipeline with a single-cell reference, a spatial 10x HDF5
matrix, and an output directory:

```bash
python scripts/run_split_rctd_py_pipeline.py \
  --ref-path /path/to/reference.h5ad \
  --spatial-path /path/to/cell_feature_matrix.h5 \
  --out-dir /path/to/split_outputs \
  --cell-type-col Annotation
```

Useful options:

- `--ref-path`: single-cell reference `.h5ad`
- `--spatial-path`: spatial transcriptomics 10x `.h5`
- `--out-dir`: directory where results are written
- `--cell-type-col`: reference `.obs` column used as deconvolution cell types, default `Annotation`
- `--chunk-size`: SPLIT purification chunk size, default `50000`
- `--min-weight`: minimum RCTD weight for SPLIT post-processing, default `0.01`

The script writes these files to `--out-dir`:

- `xenium_split_purified.h5ad`: AnnData with raw counts in `X` and purified counts in `layers["purified_counts"]`
- `rctd_deconvolution_weights.parquet`: rctd-py cell-type weights
- `split_rctd_results.parquet`: SPLIT-compatible rctd-py metadata
- `split_cell_metadata.parquet`: purified AnnData cell metadata
- `run_summary.json`: paths, shapes, cell types, and purification status counts

To keep a run log, pipe stdout/stderr through `tee`:

```bash
python scripts/run_split_rctd_py_pipeline.py \
  --ref-path /path/to/reference.h5ad \
  --spatial-path /path/to/cell_feature_matrix.h5 \
  --out-dir /path/to/split_outputs \
  --cell-type-col Annotation \
  2>&1 | tee /path/to/split_outputs/run.log
```

Example using the server environment and paths from this project:

```bash
cd /mnt/volume2/snowxue/SPLIT_python

~/miniconda3/envs/pytorch/bin/python scripts/run_split_rctd_py_pipeline.py \
  --ref-path /mnt/volume2/snowxue/cell_segmentation_transcript_assignment/refscdata/Janesick_Chromium_FFPE_Human_Breast_Cancer_Chromium_FFPE_Human_Breast_Cancer_count_sample_filtered_feature_bc_matrix.h5ad \
  --spatial-path /mnt/volume2/snowxue/10x_Xenium_with_HE/Human_Breast_Biomarkers_S1_Bot/cell_feature_matrix.h5 \
  --out-dir /mnt/volume2/snowxue/SPLIT_python/run_outputs \
  --cell-type-col Annotation \
  2>&1 | tee /mnt/volume2/snowxue/SPLIT_python/run_outputs/run.log
```


# References

1. Bilous, M. et al. Resolving sensitivity, specificity and signal contamination in Xenium spatial transcriptomics. Nat Methods https://doi.org/10.1038/s41592-026-03089-8 (2026) doi:10.1038/s41592-026-03089-8.
