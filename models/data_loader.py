#!/opt/homebrew/bin/python3.13
"""
Load training data from GCS, compute Morgan fingerprints, build PyTorch datasets.

Key design:
  - One sample = one compound: input=fingerprint (2048), output=expression vector (12,995 genes)
  - Predicts RESIDUALS from per-gene training mean (easier signal for MLP to learn)
  - Per-gene mean is stored separately and added back at inference time
"""

import io
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from vcpi_prediction_contest import counts_to_expression, load_gene_filter

# rdkit import with graceful error
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    raise ImportError("rdkit not installed. Run: pip install rdkit")

# ── constants ─────────────────────────────────────────────────────────────────
BUCKET       = os.environ.get("GCS_BUCKET", "vcpi-drugseq-2026")
DATASETS     = ["tvc-bhr-009", "tvc-kdl-010", "tvc-qnu-012"]
FP_RADIUS    = 2
FP_NBITS     = 2048
GENE_FILTER  = load_gene_filter()   # 12,995 scored genes


# ── GCS helpers ───────────────────────────────────────────────────────────────
def _gcs_client():
    from google.cloud import storage
    return storage.Client()


def read_parquet_gcs(blob_path: str) -> pl.DataFrame:
    buf = io.BytesIO()
    _gcs_client().bucket(BUCKET).blob(blob_path).download_to_file(buf)
    buf.seek(0)
    return pl.read_parquet(buf)


# ── fingerprint computation ───────────────────────────────────────────────────
def smiles_to_fingerprint(smiles: str) -> np.ndarray | None:
    """Morgan fingerprint (ECFP4) → float32 numpy array of shape (FP_NBITS,)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_NBITS)
    return np.array(fp, dtype=np.float32)


def batch_smiles_to_fingerprints(
    smiles_series: pd.Series,
    compound_ids: pd.Series,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute fingerprints for a Series of SMILES strings.
    Returns (fp_matrix, valid_compound_ids) — drops compounds with invalid SMILES.
    """
    fps, valid_ids = [], []
    n_invalid = 0
    for cid, smi in zip(compound_ids, smiles_series):
        fp = smiles_to_fingerprint(smi)
        if fp is not None:
            fps.append(fp)
            valid_ids.append(cid)
        else:
            n_invalid += 1
    if n_invalid:
        print(f"  WARNING: {n_invalid} compounds had invalid SMILES and were dropped.")
    return np.stack(fps, axis=0), valid_ids  # (N, FP_NBITS)


# ── data loading ──────────────────────────────────────────────────────────────
def load_expression_from_gcs(job_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load counts + metadata from GCS, run counts_to_expression().
    Returns (expr_long, chem) where:
      expr_long: pandas DataFrame with columns [compound, gene_id, expression]
      chem:      pandas DataFrame with columns [compound, smiles, ...]
    """
    print(f"  Loading {job_id} from GCS...")
    counts = read_parquet_gcs(f"data/{job_id}/counts.parquet")
    meta   = read_parquet_gcs(f"data/{job_id}/metadata.parquet")
    chem   = read_parquet_gcs(f"data/{job_id}/chemistry.parquet")

    expr = counts_to_expression(counts.to_pandas(), meta.to_pandas())
    return expr, chem.to_pandas()


def build_training_matrix(
    datasets: list[str] = DATASETS,
    gene_filter: list[str] = GENE_FILTER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all datasets, merge expression, return:
      - expr_wide: (genes × compounds) DataFrame, restricted to gene_filter
      - chem:      DataFrame with compound → smiles mapping (deduplicated)
    """
    all_expr, all_chem = [], []

    for job_id in datasets:
        expr, chem = load_expression_from_gcs(job_id)
        all_expr.append(expr)
        all_chem.append(chem[["compound", "user_compound_id", "smiles"]].drop_duplicates())
        print(f"  {job_id}: {expr['compound'].nunique()} compounds, "
              f"{expr['gene_id'].nunique()} genes")

    expr_all = pd.concat(all_expr, ignore_index=True)

    # Average duplicates (same compound appearing in multiple datasets)
    expr_all = (
        expr_all
        .groupby(["compound", "gene_id"], as_index=False)["expression"]
        .mean()
    )

    # Pivot to wide: rows=genes, cols=compounds
    expr_wide = expr_all.pivot_table(
        index="gene_id", columns="compound", values="expression"
    )

    # Restrict to scored genes
    expr_wide = expr_wide.reindex([g for g in gene_filter if g in expr_wide.index])
    expr_wide = expr_wide.dropna(how="all", axis=1)  # drop compounds with all-NaN

    # Fill remaining NaN with per-gene mean (rare missing values)
    gene_means = expr_wide.mean(axis=1)
    expr_wide  = expr_wide.T.fillna(gene_means).T

    chem_all = pd.concat(all_chem, ignore_index=True).drop_duplicates("compound")

    print(f"\nFinal training matrix: {expr_wide.shape[0]} genes × {expr_wide.shape[1]} compounds")
    return expr_wide, chem_all


# ── Dataset ───────────────────────────────────────────────────────────────────
@dataclass
class DrugExpressionDataset(Dataset):
    """
    PyTorch Dataset: one sample = one compound.
    Input:  Morgan fingerprint (FP_NBITS,)
    Target: residual expression vector (n_genes,) = expression - per_gene_mean
    """
    fingerprints: np.ndarray     # (N, FP_NBITS)
    expressions:  np.ndarray     # (N, n_genes)
    compound_ids: list[str]
    gene_ids:     list[str]
    per_gene_mean: np.ndarray    # (n_genes,) — added back at inference

    def __len__(self) -> int:
        return len(self.compound_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fp  = torch.from_numpy(self.fingerprints[idx])
        # Residual: deviation from per-gene mean (easier for MLP to learn)
        res = torch.from_numpy(self.expressions[idx] - self.per_gene_mean)
        return fp, res


def build_dataset(
    datasets: list[str] = DATASETS,
    gene_filter: list[str] = GENE_FILTER,
) -> DrugExpressionDataset:
    """Full pipeline: GCS → expression → fingerprints → DrugExpressionDataset."""
    print("Building training dataset...")
    expr_wide, chem = build_training_matrix(datasets, gene_filter)

    # Align chemistry to expression compounds
    expr_compounds  = list(expr_wide.columns)
    chem_indexed    = chem.set_index("compound")
    chem_aligned    = chem_indexed.reindex(expr_compounds)

    # Drop compounds with missing SMILES
    valid_mask      = chem_aligned["smiles"].notna()
    chem_aligned    = chem_aligned[valid_mask]
    expr_aligned    = expr_wide.loc[:, valid_mask.values]

    print(f"\nComputing Morgan fingerprints (radius={FP_RADIUS}, bits={FP_NBITS})...")
    fps, valid_ids  = batch_smiles_to_fingerprints(
        chem_aligned["smiles"],
        chem_aligned.index.tolist(),
    )

    # Filter expression to valid compounds (those with good SMILES)
    valid_id_set    = set(valid_ids)
    keep_cols       = [c for c in expr_aligned.columns if c in valid_id_set]
    expr_valid      = expr_aligned[keep_cols]

    # Expression matrix: (N_compounds, N_genes)
    expr_mat        = expr_valid.values.T.astype(np.float32)  # (N, n_genes)
    per_gene_mean   = expr_mat.mean(axis=0)                   # (n_genes,)

    print(f"Dataset: {len(valid_ids)} compounds, {expr_mat.shape[1]} genes")
    print(f"Fingerprint matrix: {fps.shape}")
    print(f"Expression range: [{expr_mat.min():.3f}, {expr_mat.max():.3f}]")

    return DrugExpressionDataset(
        fingerprints  = fps,
        expressions   = expr_mat,
        compound_ids  = valid_ids,
        gene_ids      = list(expr_wide.index),
        per_gene_mean = per_gene_mean,
    )


def get_dataloaders(
    dataset: DrugExpressionDataset,
    val_frac: float = 0.1,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Split dataset into train/val DataLoaders."""
    n_val   = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"Train: {n_train} compounds | Val: {n_val} compounds | Batch: {batch_size}")
    return train_dl, val_dl
