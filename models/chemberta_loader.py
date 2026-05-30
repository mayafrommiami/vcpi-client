#!/usr/bin/env python3
"""
Compute frozen ChemBERTa embeddings for all training compounds.

Uses seyonec/ChemBERTa-zinc-base-v1 (84M params, 384-dim CLS token).
Embeddings are pre-computed once and cached to GCS, then the MLP
head is trained on top — much faster than end-to-end fine-tuning.
"""

import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from google.cloud import storage

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.data_loader import (
    read_parquet_gcs, build_training_matrix, DrugExpressionDataset,
    BUCKET, DATASETS, GENE_FILTER
)

CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
EMBED_DIM       = 384
EMBED_BATCH     = 64   # compounds per forward pass through ChemBERTa
EMBED_CACHE_PATH = f"embeddings/chemberta/{'-'.join(DATASETS)}.parquet"


def compute_chemberta_embeddings(
    smiles_list: list[str],
    compound_ids: list[str],
    device: str = "cpu",
) -> tuple[np.ndarray, list[str]]:
    """
    Compute CLS-token embeddings for a list of SMILES strings.
    Returns (embed_matrix, valid_compound_ids) — drops invalid SMILES.
    """
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise ImportError("Run: pip install transformers")

    print(f"  Loading ChemBERTa: {CHEMBERTA_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL)
    model     = AutoModel.from_pretrained(CHEMBERTA_MODEL).to(device)
    model.eval()

    embeddings, valid_ids = [], []
    n_invalid = 0

    for i in range(0, len(smiles_list), EMBED_BATCH):
        batch_smiles = smiles_list[i : i + EMBED_BATCH]
        batch_ids    = compound_ids[i : i + EMBED_BATCH]

        # Filter out None/empty SMILES
        valid_mask   = [s is not None and len(s) > 0 for s in batch_smiles]
        good_smiles  = [s for s, v in zip(batch_smiles, valid_mask) if v]
        good_ids     = [c for c, v in zip(batch_ids,   valid_mask) if v]
        n_invalid   += sum(1 for v in valid_mask if not v)

        if not good_smiles:
            continue

        inputs = tokenizer(
            good_smiles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            out = model(**inputs)
            # CLS token (position 0) as compound embedding
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()

        embeddings.extend(cls)
        valid_ids.extend(good_ids)

        if (i // EMBED_BATCH) % 10 == 0:
            print(f"    {i+len(good_smiles)}/{len(smiles_list)} compounds embedded")

    if n_invalid:
        print(f"  WARNING: {n_invalid} compounds had invalid/empty SMILES and were dropped.")

    return np.stack(embeddings, axis=0).astype(np.float32), valid_ids


def load_or_compute_embeddings(
    chem: pd.DataFrame,
    device: str = "cpu",
    force_recompute: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Load cached ChemBERTa embeddings from GCS if available, else compute + cache.
    chem: DataFrame with columns [compound, smiles]
    Returns (embed_matrix, compound_ids)
    """
    gcs = storage.Client.create_anonymous_client()
    bucket = gcs.bucket(BUCKET)
    blob   = bucket.blob(EMBED_CACHE_PATH)

    if not force_recompute and blob.exists(gcs):
        print(f"  Loading cached embeddings from gs://{BUCKET}/{EMBED_CACHE_PATH}")
        buf = io.BytesIO()
        blob.download_to_file(buf)
        buf.seek(0)
        cached = pd.read_parquet(buf)
        compound_ids = cached["compound"].tolist()
        embed_mat    = cached.drop(columns=["compound"]).values.astype(np.float32)
        print(f"  Loaded {embed_mat.shape[0]} embeddings (dim={embed_mat.shape[1]})")
        return embed_mat, compound_ids

    # Compute embeddings
    print(f"  Computing ChemBERTa embeddings for {len(chem)} compounds...")
    smiles_list  = chem["smiles"].tolist()
    compound_ids_all = chem["compound"].tolist()
    embed_mat, valid_ids = compute_chemberta_embeddings(smiles_list, compound_ids_all, device)

    # Cache to GCS
    print(f"  Caching embeddings to gs://{BUCKET}/{EMBED_CACHE_PATH}")
    df_cache = pd.DataFrame(
        embed_mat,
        columns=[f"d{i}" for i in range(embed_mat.shape[1])]
    )
    df_cache.insert(0, "compound", valid_ids)
    buf = io.BytesIO()
    df_cache.to_parquet(buf, index=False)
    buf.seek(0)
    # Need write credentials for upload
    write_client = storage.Client()
    write_client.bucket(BUCKET).blob(EMBED_CACHE_PATH).upload_from_file(
        buf, rewind=True, content_type="application/octet-stream"
    )
    print(f"  Cached {embed_mat.shape[0]} embeddings.")
    return embed_mat, valid_ids


def build_chemberta_dataset(
    datasets: list[str] = DATASETS,
    gene_filter: list[str] = GENE_FILTER,
    device: str = "cpu",
) -> DrugExpressionDataset:
    """
    Full pipeline: GCS → expression → ChemBERTa embeddings → DrugExpressionDataset.
    Reuses DrugExpressionDataset — just swaps fingerprints for ChemBERTa embeddings.
    """
    print("Building ChemBERTa dataset...")
    expr_wide, chem = build_training_matrix(datasets, gene_filter)

    # Align chemistry to expression compounds
    expr_compounds  = list(expr_wide.columns)
    chem_indexed    = chem.set_index("compound")
    chem_aligned    = chem_indexed.reindex(expr_compounds)

    # Drop compounds with missing SMILES
    valid_mask      = chem_aligned["smiles"].notna()
    chem_aligned    = chem_aligned[valid_mask].reset_index()
    expr_aligned    = expr_wide.loc[:, valid_mask.values]

    # Load or compute embeddings
    embed_mat, valid_ids = load_or_compute_embeddings(
        chem_aligned[["compound", "smiles"]], device=device
    )

    # Filter expression to valid embedded compounds
    valid_id_set = set(valid_ids)
    keep_cols    = [c for c in expr_aligned.columns if c in valid_id_set]
    expr_valid   = expr_aligned[keep_cols]

    expr_mat     = expr_valid.values.T.astype(np.float32)   # (N, n_genes)
    per_gene_mean = expr_mat.mean(axis=0)

    print(f"ChemBERTa dataset: {len(valid_ids)} compounds, "
          f"{expr_mat.shape[1]} genes, embed_dim={embed_mat.shape[1]}")

    return DrugExpressionDataset(
        fingerprints  = embed_mat,      # shape (N, 384) — same field, different content
        expressions   = expr_mat,
        compound_ids  = valid_ids,
        gene_ids      = list(expr_wide.index),
        per_gene_mean = per_gene_mean,
    )
