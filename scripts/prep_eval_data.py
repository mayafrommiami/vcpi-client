#!/usr/bin/env python3
"""
Download training data from GCS and merge into the format expected by
qnu_plate_holdout_eval.py.

Outputs (in repo root):
  - train_counts.parquet    (~1.5 GB, 78k genes × ~32k samples)
  - train_metadata.parquet  (~3 MB, ~32k samples × 26 columns)
  - train_chemistry.parquet (~1.5 MB, ~14k compounds × 13 columns)
  - weights.parquet         (~375 MB, 12,995 genes × ~14k compounds)
"""

import io
import sys
from pathlib import Path

import pandas as pd
from google.cloud import storage

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "vcpi-drugseq-2026"
JOBS = ["tvc-bhr-009", "tvc-kdl-010", "tvc-qnu-012"]


def read_parquet_gcs(blob_path: str) -> pd.DataFrame:
    client = storage.Client.create_anonymous_client()
    buf = io.BytesIO()
    client.bucket(BUCKET).blob(blob_path).download_to_file(buf)
    buf.seek(0)
    return pd.read_parquet(buf)


def main():
    counts_pieces = []
    metadata_pieces = []
    chemistry_pieces = []

    for job_id in JOBS:
        print(f"Downloading {job_id}...")
        counts = read_parquet_gcs(f"data/{job_id}/counts.parquet")
        meta = read_parquet_gcs(f"data/{job_id}/metadata.parquet")
        chem = read_parquet_gcs(f"data/{job_id}/chemistry.parquet")

        meta["job_id"] = job_id
        print(f"  counts: {counts.shape}, meta: {meta.shape}, chem: {chem.shape}")

        counts_pieces.append(counts.set_index("gene_id"))
        metadata_pieces.append(meta)
        chemistry_pieces.append(chem)

    print("\nMerging counts (horizontal join on gene_id)...")
    counts_all = (
        pd.concat(counts_pieces, axis=1, join="outer")
        .fillna(0)
        .astype("int32")
        .reset_index()
    )
    print(f"  train_counts: {counts_all.shape}")

    metadata_all = pd.concat(metadata_pieces, ignore_index=True)
    print(f"  train_metadata: {metadata_all.shape}")

    chemistry_all = (
        pd.concat(chemistry_pieces, ignore_index=True)
        .drop_duplicates(subset=["compound"])
        .reset_index(drop=True)
    )
    print(f"  train_chemistry: {chemistry_all.shape}")

    # Save
    counts_all.to_parquet(ROOT / "train_counts.parquet")
    print(f"  Wrote train_counts.parquet")
    metadata_all.to_parquet(ROOT / "train_metadata.parquet")
    print(f"  Wrote train_metadata.parquet")
    chemistry_all.to_parquet(ROOT / "train_chemistry.parquet")
    print(f"  Wrote train_chemistry.parquet")

    # Weights
    print("\nDownloading Mejia weights matrix...")
    from vcpi_prediction_contest import load_weights_matrix
    W = load_weights_matrix()
    W.to_parquet(ROOT / "weights.parquet")
    print(f"  Wrote weights.parquet ({W.shape})")

    print("\nDone. All 4 parquets ready.")


if __name__ == "__main__":
    main()
