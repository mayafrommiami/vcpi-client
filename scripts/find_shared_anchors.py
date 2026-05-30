#!/opt/homebrew/bin/python3.13
"""
Find shared controls and shared perturbations between tvc-bhr-009 and tvc-qnu-012.
Uses vcpi.query() — does NOT download counts (fast, metadata-only).

Outputs: analysis/shared_anchors.csv with columns:
  inchi_key, compound_009, compound_012
"""

import os
import sys
from pathlib import Path

import polars as pl
import vcpi

# ── config ────────────────────────────────────────────────────────────────────
TOKEN = os.environ.get("TVC_TOKEN")
if not TOKEN:
    sys.exit("ERROR: TVC_TOKEN environment variable not set.")

JOBS = ["tvc-bhr-009", "tvc-qnu-012"]
FILTER_SQL = """
    SELECT compound, user_compound_id, is_control, compound_concentration,
           cell_line, timepoint, job_id
    FROM metadata
    WHERE cell_line = 'THP-1'
      AND timepoint = '24h'
      AND compound_concentration = 10000.0
"""
OUTPUT_DIR = Path(__file__).parent.parent / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 1. Pull metadata (no counts download) ─────────────────────────────────────
print("Querying metadata for tvc-bhr-009...")
meta_009 = vcpi.query(job="tvc-bhr-009", sql=FILTER_SQL)
print(f"  -009: {meta_009.height} samples")

print("Querying metadata for tvc-qnu-012...")
meta_012 = vcpi.query(job="tvc-qnu-012", sql=FILTER_SQL)
print(f"  -012: {meta_012.height} samples")

# ── 2. Shared controls ────────────────────────────────────────────────────────
ctrl_009 = set(meta_009.filter(pl.col("is_control") == True)["compound"].to_list())
ctrl_012 = set(meta_012.filter(pl.col("is_control") == True)["compound"].to_list())
shared_ctrl_compounds = ctrl_009 & ctrl_012

print(f"\n=== Controls ===")
print(f"  -009 controls: {len(ctrl_009)} unique compound IDs")
print(f"  -012 controls: {len(ctrl_012)} unique compound IDs")
print(f"  Shared by compound ID: {len(shared_ctrl_compounds)}")

# ── 3. Pull chemistry and match by inchi_key ──────────────────────────────────
CHEM_SQL = "SELECT compound, user_compound_id, inchi_key, smiles FROM chemistry"

print("\nQuerying chemistry for tvc-bhr-009...")
chem_009 = vcpi.query(job="tvc-bhr-009", sql=CHEM_SQL)

print("Querying chemistry for tvc-qnu-012...")
chem_012 = vcpi.query(job="tvc-qnu-012", sql=CHEM_SQL)

# Join on inchi_key — finds exact same molecules across batches
overlap = (
    chem_009
    .join(chem_012, on="inchi_key", how="inner", suffix="_012")
    .select([
        pl.col("inchi_key"),
        pl.col("compound").alias("compound_009"),
        pl.col("user_compound_id").alias("user_compound_id_009"),
        pl.col("compound_012"),
        pl.col("user_compound_id_012"),
    ])
)

print(f"\n=== Perturbation Overlap (by InChIKey) ===")
print(f"  Shared compounds across -009 and -012: {overlap.height}")

# Mark which shared compounds are controls in either batch
ctrl_in_009 = set(meta_009.filter(pl.col("is_control") == True)["compound"].to_list())
ctrl_in_012 = set(meta_012.filter(pl.col("is_control") == True)["compound"].to_list())

overlap = overlap.with_columns([
    pl.col("compound_009").is_in(list(ctrl_in_009)).alias("is_control_009"),
    pl.col("compound_012").is_in(list(ctrl_in_012)).alias("is_control_012"),
])

n_shared_ctrl   = overlap.filter(pl.col("is_control_009") | pl.col("is_control_012")).height
n_shared_drug   = overlap.filter(~pl.col("is_control_009") & ~pl.col("is_control_012")).height

print(f"  Of which are controls (either batch): {n_shared_ctrl}")
print(f"  Of which are non-control perturbations: {n_shared_drug}")

# ── 4. Save results ───────────────────────────────────────────────────────────
out_path = OUTPUT_DIR / "shared_anchors.csv"
overlap.write_csv(out_path)
print(f"\nSaved: {out_path}")
print("\nSample:")
print(overlap.head(10))

# ── 5. Summary for downstream analysis ───────────────────────────────────────
print("\n=== Summary for batch_effects.py ===")
print(f"  Use compounds in analysis/shared_anchors.csv as expression anchors.")
print(f"  Prefer shared controls for cleanest batch comparison")
print(f"  (same compound, different batch = pure batch signal).")
