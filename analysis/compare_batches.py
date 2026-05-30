#!/opt/homebrew/bin/python3.13
"""
Visualize batch differences across tvc-bhr-009, tvc-kdl-010, tvc-qnu-012.

Uses the 4 shared positive controls (Staurosporine, Brefeldin-A,
Trichostatin-A, Rigosertib) as anchors — same biology, any expression
difference is purely batch.

Outputs:
  analysis/batch_pca.png           — PCA of all control samples, colored by batch
  analysis/batch_pairwise.png      — Per-gene mean scatter: all 3 pairwise combos
  analysis/batch_r2_heatmap.png    — R² matrix across batches × controls
  analysis/batch_summary.csv       — Per-gene mean abs diff between each batch pair
"""

import io
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from vcpi_prediction_contest import counts_to_expression, load_gene_filter

# ── config ────────────────────────────────────────────────────────────────────
BUCKET       = os.environ.get("GCS_BUCKET", "vcpi-drugseq-2026")
ANALYSIS_DIR = Path(__file__).parent
GENE_FILTER  = load_gene_filter()

BATCHES = ["tvc-bhr-009", "tvc-kdl-010", "tvc-qnu-012"]
COLORS  = {
    "tvc-bhr-009": "#4477AA",
    "tvc-kdl-010": "#EE6677",
    "tvc-qnu-012": "#228833",
}
MARKERS = {"tvc-bhr-009": "o", "tvc-kdl-010": "s", "tvc-qnu-012": "^"}

# The 4 controls shared across all 3 batches (identified by InChIKey)
CONTROL_INCHIKEYS = {
    "HKSZLNNOFSGOKW-FYTWVXJKSA-N": "Staurosporine",
    "KQNZDYYTLMIZCT-KQPMLPITSA-N": "Brefeldin-A",
    "RTKIYFITIVXBLE-QEQCGCAPSA-N": "Trichostatin-A",
    "OWBFCJROIKNMGD-BQYQJAHWSA-N": "Rigosertib",
}


# ── GCS loader ────────────────────────────────────────────────────────────────
def read_parquet_gcs(blob_path: str) -> pl.DataFrame:
    from google.cloud import storage
    buf = io.BytesIO()
    storage.Client().bucket(BUCKET).blob(blob_path).download_to_file(buf)
    buf.seek(0)
    return pl.read_parquet(buf)


# ── load control expression per batch ────────────────────────────────────────
def load_control_expression(batch: str) -> pd.DataFrame:
    """
    Returns wide DataFrame: index=gene_id, columns=sequenced_id (sample).
    Restricted to control compounds and scored genes.
    Columns have a 'batch' prefix so we can distinguish after concat.
    """
    print(f"  Loading {batch}...")
    counts = read_parquet_gcs(f"data/{batch}/counts.parquet")
    meta   = read_parquet_gcs(f"data/{batch}/metadata.parquet")
    chem   = read_parquet_gcs(f"data/{batch}/chemistry.parquet")

    # Find control compound IDs in this batch by InChIKey
    ctrl_chem = chem.filter(pl.col("inchi_key").is_in(list(CONTROL_INCHIKEYS.keys())))
    ctrl_compounds = set(ctrl_chem["compound"].to_list())

    # Filter metadata to control samples only
    meta_ctrl = meta.filter(pl.col("compound").is_in(ctrl_compounds))

    # Add control name and batch label to metadata
    inchi_map    = dict(zip(ctrl_chem["compound"].to_list(), ctrl_chem["inchi_key"].to_list()))
    meta_pd      = meta_ctrl.to_pandas()
    meta_pd["inchi_key"]    = meta_pd["compound"].map(inchi_map)
    meta_pd["control_name"] = meta_pd["inchi_key"].map(CONTROL_INCHIKEYS)
    meta_pd["batch"]        = batch

    # Compute expression (long format)
    expr = counts_to_expression(counts.to_pandas(), meta_ctrl.to_pandas())

    # Pivot to wide: index=gene_id, columns=compound (mean across replicates)
    wide = expr.pivot_table(index="gene_id", columns="compound", values="expression")

    # Map compound → control name
    comp_to_ctrl = dict(zip(meta_pd["compound"], meta_pd["control_name"]))
    wide.columns = [f"{batch}|{comp_to_ctrl.get(c, c)}" for c in wide.columns]

    # Restrict to scored genes
    wide = wide.reindex([g for g in GENE_FILTER if g in wide.index])

    n_controls = len(ctrl_compounds)
    print(f"    {n_controls} control compounds, {wide.shape[0]} scored genes")
    return wide


# ── plots ─────────────────────────────────────────────────────────────────────
def plot_pca(all_wide: pd.DataFrame) -> None:
    """
    PCA across all control samples from all batches.
    Each point = one (batch, control compound) combination.
    """
    # all_wide: genes × (batch|control) columns
    # Transpose: rows = samples, cols = genes
    X    = all_wide.T.fillna(0).values.astype(np.float32)
    cols = all_wide.columns.tolist()   # "batch|ControlName"

    X_scaled = StandardScaler().fit_transform(X)
    n_comp   = min(10, X.shape[0] - 1)
    pca      = PCA(n_components=n_comp, random_state=42)
    coords   = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (pc_x, pc_y) in enumerate([(0, 1), (0, 2)]):
        ax = axes[ax_idx]
        for col, xy in zip(cols, coords):
            batch, ctrl_name = col.split("|", 1)
            ax.scatter(
                xy[pc_x], xy[pc_y],
                c=COLORS[batch], marker=MARKERS[batch],
                s=120, edgecolors="white", linewidths=0.8,
                label=f"{batch.split('-')[1]} | {ctrl_name}",
                zorder=3,
            )
            ax.annotate(ctrl_name[:6], (xy[pc_x], xy[pc_y]),
                        fontsize=7, ha="left", va="bottom", alpha=0.7)
        ax.set_xlabel(f"PC{pc_x+1} ({pca.explained_variance_ratio_[pc_x]:.1%})")
        ax.set_ylabel(f"PC{pc_y+1} ({pca.explained_variance_ratio_[pc_y]:.1%})")
        ax.set_title(f"PC{pc_x+1} vs PC{pc_y+1}")
        ax.grid(True, alpha=0.3)

    # Deduplicate legend entries (one per batch)
    handles, labels = axes[0].get_legend_handles_labels()
    seen, unique_h, unique_l = set(), [], []
    for h, l in zip(handles, labels):
        key = l.split("|")[0].strip()
        if key not in seen:
            seen.add(key)
            unique_h.append(h)
            unique_l.append(key)
    axes[0].legend(unique_h, unique_l, title="Batch", loc="best", fontsize=9)

    fig.suptitle(
        "PCA of Shared Controls Across All 3 Batches\n"
        "(Points = batch × control compound; separation = batch effect)",
        fontsize=12
    )
    fig.tight_layout()
    out = ANALYSIS_DIR / "batch_pca.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_pairwise_scatter(batch_means: dict[str, pd.Series]) -> None:
    """
    3 pairwise scatter plots of per-gene means.
    Each point = one gene. Diagonal = perfect agreement.
    """
    pairs = [
        ("tvc-bhr-009", "tvc-kdl-010"),
        ("tvc-bhr-009", "tvc-qnu-012"),
        ("tvc-kdl-010", "tvc-qnu-012"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (b1, b2) in zip(axes, pairs):
        df = pd.DataFrame({"x": batch_means[b1], "y": batch_means[b2]}).dropna()
        r, _  = pearsonr(df["x"], df["y"])
        r2    = r ** 2
        mad   = (df["x"] - df["y"]).abs().mean()

        ax.scatter(df["x"], df["y"], alpha=0.15, s=5, color="slategray", rasterized=True)
        mn = min(df["x"].min(), df["y"].min())
        mx = max(df["x"].max(), df["y"].max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1, alpha=0.8, label="y=x")

        short = lambda b: b.split("-")[1]   # e.g. "bhr" from "tvc-bhr-009"
        ax.set_xlabel(f"Mean log₂(CPM+1) — {short(b1)}-009" if "009" in b1
                      else f"Mean log₂(CPM+1) — {short(b1)}-010" if "010" in b1
                      else f"Mean log₂(CPM+1) — {short(b1)}-012")
        ax.set_ylabel(f"Mean log₂(CPM+1) — {short(b2)}-009" if "009" in b2
                      else f"Mean log₂(CPM+1) — {short(b2)}-010" if "010" in b2
                      else f"Mean log₂(CPM+1) — {short(b2)}-012")
        ax.set_title(f"R²={r2:.4f}  |  MAD={mad:.4f}")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Per-Gene Mean Expression: Pairwise Batch Comparison\n"
        "(Controls only — pure batch signal)",
        fontsize=12
    )
    fig.tight_layout()
    out = ANALYSIS_DIR / "batch_pairwise.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_r2_heatmap(batch_means: dict[str, pd.Series]) -> None:
    """R² matrix across all batch pairs — one panel per control compound."""
    # Also compute per-control-compound R²
    # batch_means is keyed by batch, values are mean across all 4 controls
    # For per-control, we need the wide matrix
    labels = [b.replace("tvc-", "") for b in BATCHES]
    mat = np.zeros((3, 3))
    for i, b1 in enumerate(BATCHES):
        for j, b2 in enumerate(BATCHES):
            df = pd.DataFrame({"x": batch_means[b1], "y": batch_means[b2]}).dropna()
            r, _ = pearsonr(df["x"], df["y"])
            mat[i, j] = r ** 2

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, vmin=0.9, vmax=1.0, cmap="RdYlGn")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels(labels)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat[i,j]:.4f}", ha="center", va="center",
                    fontsize=10, color="black" if mat[i,j] > 0.95 else "white")
    plt.colorbar(im, ax=ax, label="R²")
    ax.set_title("Batch R² Matrix\n(per-gene mean expression, controls only)")
    fig.tight_layout()
    out = ANALYSIS_DIR / "batch_r2_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def save_summary(batch_means: dict[str, pd.Series]) -> None:
    pairs = [
        ("tvc-bhr-009", "tvc-kdl-010"),
        ("tvc-bhr-009", "tvc-qnu-012"),
        ("tvc-kdl-010", "tvc-qnu-012"),
    ]
    frames = []
    for b1, b2 in pairs:
        df = pd.DataFrame({"mean_b1": batch_means[b1], "mean_b2": batch_means[b2]}).dropna()
        df["abs_diff"] = (df["mean_b1"] - df["mean_b2"]).abs()
        df["pair"] = f"{b1.split('-')[1]} vs {b2.split('-')[1]}"
        frames.append(df.reset_index().rename(columns={"gene_id": "gene_id"}))
    summary = pd.concat(frames)
    out = ANALYSIS_DIR / "batch_summary.csv"
    summary.to_csv(out, index=False)
    print(f"Saved: {out}")

    print("\n=== Batch Effect Summary ===")
    for b1, b2 in pairs:
        df = pd.DataFrame({"x": batch_means[b1], "y": batch_means[b2]}).dropna()
        mad = (df["x"] - df["y"]).abs().mean()
        r, _ = pearsonr(df["x"], df["y"])
        print(f"  {b1.split('-')[1]} vs {b2.split('-')[1]}:  "
              f"R²={r**2:.4f}  MAD={mad:.4f}  max_diff={((df['x']-df['y']).abs().max()):.4f}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading control expression from all 3 batches...")
    batch_wide = {}
    for batch in BATCHES:
        batch_wide[batch] = load_control_expression(batch)

    # Align all on same gene set
    shared_genes = batch_wide[BATCHES[0]].index
    for b in BATCHES[1:]:
        shared_genes = shared_genes.intersection(batch_wide[b].index)
    print(f"\nShared scored genes across all batches: {len(shared_genes)}")

    for b in BATCHES:
        batch_wide[b] = batch_wide[b].loc[shared_genes]

    # Concatenate all samples into one matrix for PCA
    all_wide = pd.concat([batch_wide[b] for b in BATCHES], axis=1)

    # Per-batch mean across all 4 controls (for pairwise scatter)
    batch_means = {b: batch_wide[b].mean(axis=1) for b in BATCHES}

    print("\nGenerating plots...")
    plot_pca(all_wide)
    plot_pairwise_scatter(batch_means)
    plot_r2_heatmap(batch_means)
    save_summary(batch_means)

    print("\nDone. Check analysis/*.png for figures.")


if __name__ == "__main__":
    main()
