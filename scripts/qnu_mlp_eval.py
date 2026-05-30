#!/usr/bin/env python3
"""
MLP holdout evaluation using the same plate-holdout framework as
qnu_plate_holdout_eval.py but scoring our FingerprintMLP architecture.

For each held-out plate:
  1. Compute Morgan FPs (2048-bit) for reference compounds
  2. Train FingerprintMLP for N epochs on CPU
  3. Predict held-out validation compounds
  4. Score with wMSE using official Mejia weights

Usage:
    python scripts/qnu_mlp_eval.py
    python scripts/qnu_mlp_eval.py --epochs 50 --lr 5e-4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add parent dir to path so we can import from models/ and scripts/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models"))

import qnu_plate_holdout_eval as qnu
from mlp import FingerprintMLP, WeightedMSELoss, build_loss_weights


FP_NBITS = 2048
FP_RADIUS = 2


def smiles_to_fp(smiles: str) -> np.ndarray | None:
    """Morgan fingerprint via RDKit."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_NBITS)
    return np.array(fp, dtype=np.float32)


def build_fp_matrix(train_chem: pd.DataFrame, compound_ids: list[str]) -> np.ndarray:
    """Build (N, 2048) fingerprint matrix for given compound IDs."""
    chem = train_chem.copy()
    chem["user_id"] = chem["user_compound_id"].astype(str)
    chem = chem.drop_duplicates("user_id").set_index("user_id")

    fps = []
    for uid in compound_ids:
        smi = str(chem.loc[uid, "smiles"])
        fp = smiles_to_fp(smi)
        if fp is None:
            fp = np.zeros(FP_NBITS, dtype=np.float32)
        fps.append(fp)
    return np.vstack(fps)


def train_mlp_on_plate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_genes: int,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cpu",
) -> FingerprintMLP:
    """Train a FingerprintMLP on the reference compounds for one holdout split."""
    model = FingerprintMLP(fp_dim=FP_NBITS, n_genes=n_genes)
    model = model.to(device)

    # Build loss weights from training expression variance
    loss_weights = build_loss_weights(y_train)
    criterion = WeightedMSELoss(loss_weights).to(device)

    # Per-gene mean for residual prediction
    per_gene_mean = y_train.mean(axis=0)

    # Residuals
    y_residual = y_train - per_gene_mean[None, :]

    x_t = torch.from_numpy(x_train).float()
    y_t = torch.from_numpy(y_residual).float()
    dataset = TensorDataset(x_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(dataset)
            print(f"    Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}", flush=True)

    model.eval()
    return model, per_gene_mean


def predict_mlp(
    model: FingerprintMLP,
    x_val: np.ndarray,
    per_gene_mean: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Predict expression for validation compounds. Returns (N_val, n_genes)."""
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x_val).float().to(device)
        residuals = model(x_t).cpu().numpy()
    # Add back per-gene mean
    return np.clip(residuals + per_gene_mean[None, :], 0.0, None).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-prefix", default="eval_mlp")
    parser.add_argument("--num-plates", type=int, default=qnu.DEFAULT_NUM_PLATES)
    parser.add_argument("--plates", nargs="*", type=int, default=None)
    parser.add_argument("--include-ensemble", action="store_true",
                        help="Also score MLP+Ridge ensemble (requires Ridge predictions)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    train_chem, train_meta, query, _qpath, gene_filter, weight_cols = qnu.load_inputs()
    train_meta = train_meta.copy()
    train_meta["user_compound_id"] = train_meta["user_compound_id"].astype(str)
    n_genes = len(gene_filter)

    active_mask = qnu.target_active_mask(train_meta)
    active_ids = set(train_meta.loc[active_mask, "user_compound_id"])
    all_ids, all_fps = qnu.build_chemistry_maps(train_chem, active_ids, weight_cols)
    all_id_index = {uid: i for i, uid in enumerate(all_ids)}

    qnu_active = train_meta[
        (train_meta["job_id"] == qnu.QNU_JOB_ID)
        & active_mask
        & train_meta["user_compound_id"].isin(all_ids)
    ].copy()

    selection_args = SimpleNamespace(
        plates=args.plates,
        num_plates=args.num_plates,
        max_plates=None,
        plate_selection="test-similar",
    )
    plates, plate_stats = qnu.select_plates(
        args=selection_args,
        qnu_active=qnu_active,
        all_fps=all_fps,
        query=query,
    )

    print(f"Selected {len(plates)} plates for holdout eval", flush=True)

    # Aggregate expression once for all compounds
    print("Aggregating all active-compound expression...", flush=True)
    expr_all = qnu.expression_wide(train_meta, all_ids, gene_filter)
    y_all = expr_all[all_ids].T.to_numpy(dtype=np.float32)

    # Build Morgan FP matrix once for all compounds
    print(f"Computing {FP_NBITS}-bit Morgan fingerprints for {len(all_ids)} compounds...", flush=True)
    fp_all = build_fp_matrix(train_chem, all_ids)
    print(f"  FP matrix: {fp_all.shape}", flush=True)

    # Also build Ridge features for ensemble
    if args.include_ensemble:
        print("Building RDKit regression features for Ridge ensemble...", flush=True)
        ridge_features = qnu.regression_features(train_chem, all_ids)

    plate_rows: list[dict[str, object]] = []
    per_compound_pieces: list[pd.DataFrame] = []

    for plate_idx, plate_id in enumerate(plates, start=1):
        print(f"\n[{plate_idx}/{len(plates)}] Plate {plate_id}", flush=True)
        plate_meta = qnu_active[qnu_active["container_id"] == plate_id].copy()
        val_ids = sorted(set(plate_meta["user_compound_id"].astype(str)) & set(all_ids))
        if not val_ids:
            continue

        val_positions = {all_id_index[uid] for uid in val_ids}
        ref_mask = np.array([i not in val_positions for i in range(len(all_ids))], dtype=bool)
        ref_ids = [uid for uid, keep in zip(all_ids, ref_mask, strict=True) if keep]

        # Ground truth from plate-level expression
        truth = qnu.plate_expression_wide(
            plate_meta[plate_meta["user_compound_id"].astype(str).isin(val_ids)],
            gene_filter,
        )
        truth = truth.reindex(index=gene_filter, columns=val_ids)
        weights = qnu.load_weights(gene_filter, val_ids)

        # Train/val split for MLP
        val_idx = np.array([all_id_index[uid] for uid in val_ids], dtype=np.int64)
        x_ref = fp_all[ref_mask]
        x_val = fp_all[val_idx]
        y_ref = y_all[ref_mask]

        print(f"  Training MLP: {len(ref_ids)} ref compounds → {len(val_ids)} val compounds", flush=True)
        print(f"  x_ref: {x_ref.shape}, y_ref: {y_ref.shape}", flush=True)

        model, per_gene_mean = train_mlp_on_plate(
            x_ref, y_ref, n_genes,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            device=device,
        )

        # Predict
        pred_arr = predict_mlp(model, x_val, per_gene_mean, device=device)
        pred_df = pd.DataFrame(pred_arr.T, index=gene_filter, columns=val_ids)

        scores: dict[str, pd.Series] = {}

        # MLP score
        mlp_name = f"morgan_fp{FP_NBITS}_mlp_{args.epochs}ep"
        scores[mlp_name] = qnu.score_wmse(truth, pred_df, weights)

        # Mean baseline for comparison
        mean_expr = y_ref.mean(axis=0)
        mean_pred = pd.DataFrame(
            np.broadcast_to(mean_expr[:, None], (n_genes, len(val_ids))),
            index=gene_filter, columns=val_ids,
        )
        scores["all_active_mean"] = qnu.score_wmse(truth, mean_pred, weights)

        # Ridge baseline for comparison
        ridge_pred_arr = qnu.target_pca_ridge_prediction(
            x_ref=qnu.regression_features(train_chem, all_ids)[ref_mask],
            x_val=qnu.regression_features(train_chem, all_ids)[val_idx],
            y_ref=y_ref,
            n_target_components=128,
            alpha=1000.0,
        )
        ridge_pred_df = pd.DataFrame(ridge_pred_arr.T, index=gene_filter, columns=val_ids)
        scores["ridge_pca128_alpha1000"] = qnu.score_wmse(truth, ridge_pred_df, weights)

        # Ensemble: simple average of MLP + Ridge
        if args.include_ensemble:
            ensemble_arr = (pred_arr + ridge_pred_arr) / 2.0
            ensemble_df = pd.DataFrame(ensemble_arr.T, index=gene_filter, columns=val_ids)
            scores["mlp_ridge_ensemble"] = qnu.score_wmse(truth, ensemble_df, weights)

        # Record results
        per_compound = pd.DataFrame({"plate_id": plate_id, "compound": val_ids})
        for model_name, series in scores.items():
            values = series.reindex(val_ids).to_numpy()
            per_compound[model_name] = values
            plate_rows.append({
                "plate_id": plate_id,
                "model": model_name,
                "n_compounds": len(val_ids),
                "wmse_mean": float(np.mean(values)),
            })
            print(f"  {model_name}: wmse_mean={np.mean(values):.6f}", flush=True)
        per_compound_pieces.append(per_compound)

    # Summarize
    per_compound_all = pd.concat(per_compound_pieces, ignore_index=True)
    model_cols = [c for c in per_compound_all.columns if c not in {"plate_id", "compound"}]
    summary_rows = []
    for model_name in model_cols:
        values = per_compound_all[model_name].to_numpy()
        summary_rows.append({
            "model": model_name,
            "n_plates": int(per_compound_all["plate_id"].nunique()),
            "n_compounds": int(per_compound_all["compound"].nunique()),
            "wmse_mean": float(np.mean(values)),
        })
    summary = pd.DataFrame(summary_rows).sort_values("wmse_mean").reset_index(drop=True)

    out_prefix = ROOT / args.output_prefix
    summary.to_csv(f"{out_prefix}_summary.csv", index=False)
    pd.DataFrame(plate_rows).to_csv(f"{out_prefix}_per_plate.csv", index=False)
    per_compound_all.to_csv(f"{out_prefix}_per_compound.csv", index=False)

    print("\n" + "=" * 60)
    print("MLP Holdout Evaluation Results:")
    print("=" * 60)
    print(summary.to_string(index=False))
    print(f"\nWrote {out_prefix}_summary.csv")
    print(f"Wrote {out_prefix}_per_plate.csv")
    print(f"Wrote {out_prefix}_per_compound.csv")


if __name__ == "__main__":
    main()
