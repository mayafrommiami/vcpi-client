#!/opt/homebrew/bin/python3.13
"""
Download training data from vcpi-client and upload to GCS.
Processes one dataset at a time to stay within memory limits.

Usage:
    TVC_TOKEN=<token> GCS_BUCKET=vcpi-drugseq-2026 python scripts/upload_data.py

Uploads per-dataset:
    gs://<bucket>/data/<job_id>/counts.parquet    (genes × samples, filtered to contest condition)
    gs://<bucket>/data/<job_id>/metadata.parquet  (sample metadata)
    gs://<bucket>/data/<job_id>/chemistry.parquet (compound properties + SMILES)
"""

import gc
import io
import os
import sys

import polars as pl
from google.cloud import storage
from tqdm import tqdm
import vcpi

# ── config ────────────────────────────────────────────────────────────────────
TVC_TOKEN = os.environ.get("TVC_TOKEN")
if not TVC_TOKEN:
    sys.exit("ERROR: TVC_TOKEN environment variable not set.")

BUCKET_NAME = os.environ.get("GCS_BUCKET", "vcpi-drugseq-2026")
DATASETS    = ["tvc-bhr-009", "tvc-kdl-010", "tvc-qnu-012"]

# Contest condition: THP-1, 24h, 10 µM (stored as 10000.0 nM)
FILTER_CELL_LINE    = "THP-1"
FILTER_TIMEPOINT    = "24h"
FILTER_CONCENTRATION = 10000.0  # nM


# ── helpers ───────────────────────────────────────────────────────────────────
def upload_df(df: pl.DataFrame, bucket, blob_path: str) -> None:
    """Write a Polars DataFrame as parquet directly to GCS (no local file)."""
    buf = io.BytesIO()
    df.write_parquet(buf)
    buf.seek(0)
    blob = bucket.blob(blob_path)
    blob.upload_from_file(buf, content_type="application/octet-stream")
    size_mb = buf.tell() / 1e6
    print(f"    uploaded gs://{bucket.name}/{blob_path}  ({size_mb:.1f} MB)")


def already_uploaded(bucket, job_id: str) -> bool:
    """Return True if all three parquets for this job_id already exist in GCS."""
    paths = [
        f"data/{job_id}/counts.parquet",
        f"data/{job_id}/metadata.parquet",
        f"data/{job_id}/chemistry.parquet",
    ]
    return all(storage.Blob(p, bucket).exists(bucket.client) for p in paths)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    gcs = storage.Client()
    bucket = gcs.bucket(BUCKET_NAME)

    for job_id in tqdm(DATASETS, desc="Datasets"):
        print(f"\n{'='*60}")
        print(f"Processing {job_id}")

        if already_uploaded(bucket, job_id):
            print(f"  Already in GCS — skipping.")
            continue

        # 1. Download full experiment
        print(f"  Downloading from vcpi (this may take a few minutes)...")
        exp = vcpi.load_experiment(job_id)
        data: pl.DataFrame     = exp["data"]
        meta: pl.DataFrame     = exp["metadata"]
        chem: pl.DataFrame     = exp["chemistry"]

        print(f"  Raw counts shape: {data.shape}")
        print(f"  Raw metadata rows: {meta.height}")

        # 2. Filter metadata to contest condition
        meta_filt = meta.filter(
            (pl.col("cell_line") == FILTER_CELL_LINE)
            & (pl.col("timepoint") == FILTER_TIMEPOINT)
            & (pl.col("compound_concentration") == FILTER_CONCENTRATION)
        )
        print(f"  After filter ({FILTER_CELL_LINE}, {FILTER_TIMEPOINT}, {FILTER_CONCENTRATION}nM): "
              f"{meta_filt.height} samples")

        if meta_filt.height == 0:
            print(f"  WARNING: No samples pass filter for {job_id}, skipping.")
            continue

        # 3. Filter counts to matching sample columns
        keep_ids = set(meta_filt["sequenced_id"].to_list())
        gene_col = "gene_id"
        keep_cols = [gene_col] + [c for c in data.columns if c in keep_ids]
        counts_filt = data.select(keep_cols)
        print(f"  Filtered counts shape: {counts_filt.shape}")

        # 4. Filter chemistry to compounds present in filtered metadata
        present_compounds = set(meta_filt["compound"].to_list())
        chem_filt = chem.filter(pl.col("compound").is_in(present_compounds))
        print(f"  Filtered chemistry rows: {chem_filt.height}")

        # 5. Upload to GCS
        print(f"  Uploading to gs://{BUCKET_NAME}/data/{job_id}/...")
        upload_df(counts_filt, bucket, f"data/{job_id}/counts.parquet")
        upload_df(meta_filt,   bucket, f"data/{job_id}/metadata.parquet")
        upload_df(chem_filt,   bucket, f"data/{job_id}/chemistry.parquet")

        # 6. Explicitly free memory before next dataset
        del exp, data, meta, chem, meta_filt, counts_filt, chem_filt, keep_ids
        gc.collect()
        print(f"  [{job_id}] complete.")

    print("\nAll datasets uploaded.")
    print(f"Verify with: gsutil ls -l gs://{BUCKET_NAME}/data/")


if __name__ == "__main__":
    main()
