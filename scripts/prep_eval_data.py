#!/usr/bin/env python3
"""
Download pre-merged training parquets from GCS.

Outputs (in repo root):
  - train_counts.parquet    (~1.5 GB, 78k genes × ~32k samples)
  - train_metadata.parquet  (~3 MB, ~32k samples × 26 columns)
  - train_chemistry.parquet (~1.5 MB, ~14k compounds × 13 columns)
  - weights.parquet         (~375 MB, 12,995 genes × ~14k compounds)
"""

import io
from pathlib import Path

from google.cloud import storage

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "vcpi-drugseq-2026"
FILES = [
    "train_counts.parquet",
    "train_metadata.parquet",
    "train_chemistry.parquet",
    "weights.parquet",
]


def main():
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(BUCKET)

    for fname in FILES:
        dest = ROOT / fname
        print(f"Downloading {fname}...")
        bucket.blob(f"merged/{fname}").download_to_filename(str(dest))
        size_mb = dest.stat().st_size / 1e6
        print(f"  Wrote {dest} ({size_mb:.1f} MB)")

    print("\nDone. All 4 parquets ready.")


if __name__ == "__main__":
    main()
