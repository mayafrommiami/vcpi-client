#!/opt/homebrew/bin/python3.13
"""
Batch effect characterization: tvc-bhr-009 vs tvc-qnu-012.

Reads data from GCS (must run upload_data.py first).
Reads analysis/shared_anchors.csv (must run find_shared_anchors.py first).

Outputs:
  analysis/qc_comparison.png       — QC metric distributions per batch
  analysis/pca_batch.png           — PCA of expression for shared anchors
  analysis/gene_mean_scatter.png   — Per-gene mean: -009 vs -012
  analysis/batch_effect_genes.csv  — Ranked genes by mean abs diff
"""

import io
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from google.cloud import storage
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from vcpi_prediction_contest import counts_to_expression, load_gene_filter

# ── config ────────────────────────────────────────────────────────────────────
BUCKET_NAME  = os.environ.get("GCS_BUCKET", "vcpi-drugseq-2026")
ANALYSIS_DIR = Path(__file__).parent
ANCHORS_CSV  = ANALYSIS_DIR / "shared_anchors.csv"

BATCH_A = "tvc-bhr-009"
BATCH_B = "tvc-qnu-012"
COLORS  = {BATCH_A: "steelblue", BATCH_B: "tomato"}

QC_METRICS = ["total_umi_count", "percent_mitochondrial", "percent_duplicated", "ngenes3"]


# ── GCS I/O ───────────────────────────────────────────────────────────────────
_gcs_client: storage.Client | None = None

def gcs_client() -> storage.Client:
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client()
    return _gcs_client


def read_parquet_from_gcs(blob_path: str) -> pl.DataFrame:
    bucket = gcs_client().bucket(BUCKET_NAME)
    buf = io.BytesIO()
    bucket.blob(blob_path).download_to_file(buf)
    buf.seek(0)
    return pl.read_parquet(buf)


# ── Step 4b: QC metric comparison ─────────────────────────────────────────────
def load_filtered_metadata(anchor_compounds_009: set, anchor_compounds_012: set):
    """Load metadata from GCS, filtered to anchor compounds only."""
    meta_a = read_parquet_from_gcs(f"data/{BATCH_A}/metadata.parquet")
    meta_b = read_parquet_from_gcs(f"data/{BATCH_B}/metadata.parquet")

    meta_a = meta_a.filter(pl.col("compound").is_in(list(anchor_compounds_009)))
    meta_b = meta_b.filter(pl.col("compound").is_in(list(anchor_compounds_012)))

    meta_a = meta_a.with_columns(pl.lit(BATCH_A).alias("batch"))
    meta_b = meta_b.with_columns(pl.lit(BATCH_B).alias("batch"))

    print(f"Anchor samples — {BATCH_A}: {meta_a.height}, {BATCH_B}: {meta_b.height}")
    return meta_a, meta_b


def plot_qc_metrics(meta_a: pl.DataFrame, meta_b: pl.DataFrame) -> None:
    combined = pd.concat([meta_a.to_pandas(), meta_b.to_pandas()], ignore_index=True)
    available = [m for m in QC_METRICS if m in combined.columns]

    if not available:
        print("WARNING: No QC metric columns found, skipping QC plot.")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        sns.violinplot(
            data=combined, x="batch", y=metric, ax=ax,
            palette=COLORS, cut=0, inner="box"
        )
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle(f"QC Metrics: {BATCH_A} vs {BATCH_B}\n(shared anchor compounds only)")
    fig.tight_layout()
    out = ANALYSIS_DIR / "qc_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Step 4c: Expression PCA ────────────────────────────────────────────────────
def run_expression_pca(
    meta_a: pl.DataFrame,
    meta_b: pl.DataFrame,
    anchor_compounds_009: set,
    anchor_compounds_012: set,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_filter = load_gene_filter()  # 12,995 scored genes

    def load_and_express(batch: str, meta: pl.DataFrame) -> pd.DataFrame:
        print(f"  Loading counts for {batch}...")
        counts = read_parquet_from_gcs(f"data/{batch}/counts.parquet")

        # Keep only anchor sample columns
        anchor_ids = set(meta["sequenced_id"].to_list())
        keep_cols  = ["gene_id"] + [c for c in counts.columns if c in anchor_ids]
        counts = counts.select(keep_cols)
        print(f"  {batch} counts: {counts.shape}")

        # counts_to_expression expects pandas (or coercible)
        expr = counts_to_expression(counts.to_pandas(), meta.to_pandas())
        # expr: long pandas (compound, gene_id, expression)

        # Pivot to wide: index=gene_id, columns=compound
        wide = expr.pivot_table(
            index="gene_id", columns="compound", values="expression"
        )
        # Restrict to scored genes
        wide = wide.reindex([g for g in gene_filter if g in wide.index])
        print(f"  {batch} wide expr: {wide.shape}")
        return wide

    wide_a = load_and_express(BATCH_A, meta_a)
    wide_b = load_and_express(BATCH_B, meta_b)

    # Align on shared genes
    shared_genes = wide_a.index.intersection(wide_b.index)
    wide_a = wide_a.loc[shared_genes]
    wide_b = wide_b.loc[shared_genes]
    print(f"  Shared genes after alignment: {len(shared_genes)}")

    # ── PCA ──────────────────────────────────────────────────────────────────
    # Transpose: rows = samples (compounds), cols = genes
    X_a = wide_a.T.copy()
    X_b = wide_b.T.copy()
    X_a["batch"] = BATCH_A
    X_b["batch"] = BATCH_B
    X_all = pd.concat([X_a, X_b])

    batch_labels = X_all["batch"].values
    X_mat        = X_all.drop(columns=["batch"]).fillna(0).values

    print(f"  Running PCA on {X_mat.shape[0]} samples × {X_mat.shape[1]} genes...")
    X_scaled = StandardScaler().fit_transform(X_mat)
    pca      = PCA(n_components=min(50, X_mat.shape[0] - 1), random_state=42)
    coords   = pca.fit_transform(X_scaled)

    # Plot PC1 vs PC2
    fig, ax = plt.subplots(figsize=(8, 6))
    for batch in [BATCH_A, BATCH_B]:
        mask = batch_labels == batch
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=COLORS[batch], label=batch, alpha=0.75, s=70, edgecolors="none"
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var explained)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var explained)")
    ax.legend(framealpha=0.8)
    ax.set_title(f"PCA of shared-anchor expression\n{BATCH_A} vs {BATCH_B}")
    fig.tight_layout()
    out = ANALYSIS_DIR / "pca_batch.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Per-gene mean scatter ─────────────────────────────────────────────────
    mean_a = wide_a.mean(axis=1)
    mean_b = wide_b.mean(axis=1)
    means  = pd.DataFrame({"mean_009": mean_a, "mean_012": mean_b}).dropna()

    r, _  = pearsonr(means["mean_009"], means["mean_012"])
    r2    = r ** 2
    print(f"  Per-gene mean R² ({BATCH_A} vs {BATCH_B}): {r2:.4f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(means["mean_009"], means["mean_012"], alpha=0.25, s=8, color="slategray")
    ax.axline((0, 0), slope=1, color="crimson", linestyle="--", linewidth=1, label="y=x")
    ax.set_xlabel(f"Mean log₂(CPM+1) — {BATCH_A}")
    ax.set_ylabel(f"Mean log₂(CPM+1) — {BATCH_B}")
    ax.set_title(f"Per-gene mean expression: -009 vs -012\nR²={r2:.4f}")
    ax.legend()
    fig.tight_layout()
    out = ANALYSIS_DIR / "gene_mean_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    return wide_a, wide_b


# ── Step 4d: Batch effect magnitude ───────────────────────────────────────────
def summarize_batch_effects(wide_a: pd.DataFrame, wide_b: pd.DataFrame) -> None:
    # Both are (genes × compounds); compute per-gene mean across compounds
    mean_a = wide_a.mean(axis=1)
    mean_b = wide_b.mean(axis=1)
    abs_diff = (mean_a - mean_b).abs().rename("mean_abs_diff")

    results = abs_diff.reset_index().rename(columns={"index": "gene_id"})
    results = results.sort_values("mean_abs_diff", ascending=False).reset_index(drop=True)

    out = ANALYSIS_DIR / "batch_effect_genes.csv"
    results.to_csv(out, index=False)
    print(f"\nBatch effect summary saved: {out}")
    print(f"  {len(results)} genes scored")
    print(f"  Mean abs diff — median: {results['mean_abs_diff'].median():.4f}, "
          f"max: {results['mean_abs_diff'].max():.4f}")
    print("\nTop 10 most batch-affected genes:")
    print(results.head(10).to_string(index=False))


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if not ANCHORS_CSV.exists():
        sys.exit(
            f"ERROR: {ANCHORS_CSV} not found.\n"
            "Run scripts/find_shared_anchors.py first."
        )

    # Load shared anchors
    anchors = pd.read_csv(ANCHORS_CSV)
    print(f"Loaded {len(anchors)} shared anchors from {ANCHORS_CSV}")

    anchor_compounds_009 = set(anchors["compound_009"].tolist())
    anchor_compounds_012 = set(anchors["compound_012"].tolist())

    # Prefer controls for cleanest batch signal (if enough exist)
    if "is_control_009" in anchors.columns:
        ctrl_anchors = anchors[anchors["is_control_009"] | anchors["is_control_012"]]
        if len(ctrl_anchors) >= 3:
            print(f"Using {len(ctrl_anchors)} control anchors for QC/PCA "
                  f"(+ {len(anchors) - len(ctrl_anchors)} drug anchors)")
            # For QC, show all anchors; PCA uses all
        else:
            print(f"Only {len(ctrl_anchors)} control anchors — using all {len(anchors)} shared compounds")

    # Step 4b: QC metrics
    print(f"\n{'─'*50}")
    print("Step 4b: QC metric comparison")
    meta_a, meta_b = load_filtered_metadata(anchor_compounds_009, anchor_compounds_012)
    plot_qc_metrics(meta_a, meta_b)

    # Step 4c: PCA
    print(f"\n{'─'*50}")
    print("Step 4c: Expression PCA")
    wide_a, wide_b = run_expression_pca(
        meta_a, meta_b, anchor_compounds_009, anchor_compounds_012
    )

    # Step 4d: Batch effect genes
    print(f"\n{'─'*50}")
    print("Step 4d: Batch effect magnitude")
    summarize_batch_effects(wide_a, wide_b)

    print("\nAll batch effect analysis complete.")
    print(f"Outputs in: {ANALYSIS_DIR}/")


if __name__ == "__main__":
    main()
