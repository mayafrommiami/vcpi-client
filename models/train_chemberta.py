#!/usr/bin/env python3
"""
Training script for ChemBERTa frozen-embedding MLP on Vertex AI (or local).

Same loop as train.py but swaps Morgan fingerprints for 384-dim ChemBERTa
CLS-token embeddings (pre-computed once, cached to GCS).

Usage (local):
    python models/train_chemberta.py

Usage (Vertex AI):
    Submitted via scripts/submit_vertex_job.py --model chemberta
"""

import gc
import io
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from google.cloud import storage

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mlp import FingerprintMLP, WeightedMSELoss, build_loss_weights
from models.chemberta_loader import build_chemberta_dataset, EMBED_DIM
from models.data_loader import get_dataloaders, GENE_FILTER
from vcpi_prediction_contest import load_test_compounds

# ── config ────────────────────────────────────────────────────────────────────
BUCKET      = os.environ.get("GCS_BUCKET",     "vcpi-drugseq-2026")
OUTPUT_DIR  = os.environ.get("AIP_MODEL_DIR",  f"gs://{BUCKET}/runs/chemberta-latest")
EPOCHS      = int(os.environ.get("EPOCHS",     "80"))
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE", "64"))
LR          = float(os.environ.get("LR",       "1e-3"))
VAL_FRAC    = float(os.environ.get("VAL_FRAC", "0.1"))
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Bucket: gs://{BUCKET}")
print(f"Output: {OUTPUT_DIR}")
print(f"Epochs: {EPOCHS}  BS: {BATCH_SIZE}  LR: {LR}")


# ── GCS helpers ───────────────────────────────────────────────────────────────
def upload_bytes(buf: io.BytesIO, gcs_path: str) -> None:
    assert gcs_path.startswith("gs://"), f"Expected gs:// path, got {gcs_path}"
    parts = gcs_path[5:].split("/", 1)
    bucket_name, blob_path = parts[0], parts[1]
    write_client = storage.Client()
    write_client.bucket(bucket_name).blob(blob_path).upload_from_file(
        buf, rewind=True, content_type="application/octet-stream"
    )
    print(f"  Uploaded → {gcs_path}")


def save_to_gcs(obj, gcs_path: str, fmt: str = "parquet") -> None:
    buf = io.BytesIO()
    if fmt == "parquet":
        obj.to_parquet(buf, index=False)
    elif fmt == "pt":
        torch.save(obj, buf)
    elif fmt == "json":
        buf.write(json.dumps(obj).encode())
    buf.seek(0)
    upload_bytes(buf, gcs_path)


# ── ChemBERTa embedding at inference ──────────────────────────────────────────
def smiles_to_chemberta_embedding(
    smiles: str,
    tokenizer,
    model,
    device: str = "cpu",
) -> np.ndarray | None:
    """Single SMILES → 384-dim CLS embedding (or None for invalid SMILES)."""
    if not smiles or smiles != smiles:  # empty / NaN
        return None
    try:
        inputs = tokenizer(
            [smiles],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)
        return out.last_hidden_state[0, 0, :].cpu().numpy().astype(np.float32)
    except Exception:
        return None


# ── training ──────────────────────────────────────────────────────────────────
def train():
    # 1. Build dataset (loads/computes ChemBERTa embeddings, cached to GCS)
    print("\n=== Building ChemBERTa dataset ===")
    dataset = build_chemberta_dataset(device=DEVICE)
    train_dl, val_dl = get_dataloaders(
        dataset, val_frac=VAL_FRAC, batch_size=BATCH_SIZE
    )

    n_genes = len(dataset.gene_ids)
    print(f"Genes: {n_genes}  |  Train: {len(train_dl.dataset)}  |  Val: {len(val_dl.dataset)}")

    # 2. Model: same MLP, just fp_dim=384 (ChemBERTa) instead of 2048 (Morgan)
    model     = FingerprintMLP(fp_dim=EMBED_DIM, n_genes=n_genes).to(DEVICE)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    weights   = build_loss_weights(dataset.expressions).to(DEVICE)
    loss_fn   = WeightedMSELoss(weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # 3. Training loop
    print("\n=== Training ===")
    best_val   = float("inf")
    best_state = None
    history    = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0.0
        for emb, target in train_dl:
            emb, target = emb.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(emb), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        scheduler.step()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for emb, target in val_dl:
                emb, target = emb.to(DEVICE), target.to(DEVICE)
                v_loss += loss_fn(model(emb), target).item()

        t_loss /= len(train_dl)
        v_loss /= len(val_dl)
        history.append({"epoch": epoch, "train_loss": t_loss, "val_loss": v_loss})

        if v_loss < best_val:
            best_val   = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  train={t_loss:.4f}  val={v_loss:.4f}  "
                  f"best={best_val:.4f}  lr={scheduler.get_last_lr()[0]:.1e}")

    model.load_state_dict(best_state)
    model.eval()
    print(f"\nBest val loss: {best_val:.4f}")

    # 4. Generate submission using ChemBERTa embeddings for test compounds
    print("\n=== Generating submission ===")
    try:
        from transformers import AutoTokenizer, AutoModel as HFAutoModel
        from models.chemberta_loader import CHEMBERTA_MODEL
        chem_tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL)
        chem_model     = HFAutoModel.from_pretrained(CHEMBERTA_MODEL).to(DEVICE)
        chem_model.eval()
        use_chemberta_inference = True
        print("  Using ChemBERTa for test compound embeddings")
    except Exception as e:
        print(f"  WARNING: Could not load ChemBERTa for inference ({e}); falling back to per-gene mean")
        use_chemberta_inference = False

    test_compounds = load_test_compounds()
    per_gene_mean  = torch.tensor(dataset.per_gene_mean, device=DEVICE)

    rows = []
    with torch.no_grad():
        for _, row in test_compounds.iterrows():
            emb = None
            if use_chemberta_inference:
                emb = smiles_to_chemberta_embedding(
                    row.get("smiles", ""), chem_tokenizer, chem_model, DEVICE
                )

            if emb is None:
                pred = per_gene_mean.cpu().numpy()
            else:
                emb_t    = torch.tensor(emb).unsqueeze(0).to(DEVICE)
                residual = model(emb_t).squeeze(0)
                pred     = (per_gene_mean + residual).clamp(min=0).cpu().numpy()

            for gene, val in zip(dataset.gene_ids, pred):
                rows.append({
                    "compound":             str(row["compound"]),
                    "gene_id":              gene,
                    "predicted_expression": float(val),
                })

    submission  = pd.DataFrame(rows)
    n_compounds = submission["compound"].nunique()
    n_genes_out = submission["gene_id"].nunique()
    print(f"Submission: {len(submission):,} rows  ({n_compounds} compounds × {n_genes_out} genes)")

    # 5. Save to GCS
    print("\n=== Saving outputs ===")
    save_to_gcs(submission,                          f"{OUTPUT_DIR}/submission.parquet")
    save_to_gcs({"state_dict": best_state,
                 "gene_ids":   dataset.gene_ids,
                 "per_gene_mean": dataset.per_gene_mean.tolist(),
                 "embed_dim": EMBED_DIM},
                f"{OUTPUT_DIR}/model.pt", fmt="pt")
    save_to_gcs(pd.DataFrame(history),               f"{OUTPUT_DIR}/history.parquet")

    print(f"\nDone. Outputs at: {OUTPUT_DIR}")
    return submission


if __name__ == "__main__":
    train()
