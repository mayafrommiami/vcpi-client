#!/usr/bin/env python3
"""
Generate an ensemble submission.parquet by averaging predictions from
multiple Ridge+PCA models trained on ALL available training data.

Ensemble strategy: train K models with different (n_components, alpha, fp_config)
settings, then average their test-set predictions. Diversity across FP configs
provides the most benefit; diversity within the same FP config is marginal.

Usage:
    # Default: top configs confirmed by holdout sweep
    python scripts/generate_ensemble_submission.py

    # Submit after generating
    python scripts/generate_ensemble_submission.py --submit

    # Custom weights (e.g. inverse-wmse from sweep CSVs)
    python scripts/generate_ensemble_submission.py --weighted
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import Ridge
from sklearn.utils.extmath import randomized_svd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import qnu_plate_holdout_eval as qnu

SEED = 13
DESC_COLS = qnu.DESC_COLS


# ── Ensemble member definitions ───────────────────────────────────────────────
# Each entry: (label, fp_radius, fp_bits, use_desc, n_components, alpha, holdout_wmse)
# holdout_wmse from sweep results — used for inverse-wmse weighting
ENSEMBLE_MEMBERS = [
    # Best confirmed config (improvements + extended sweeps)
    ("r2_512+desc_pca128_a1000",   2, 512,  True,  128, 1000.0, 0.47866),
    # Near-best with more PCA components
    ("r2_512+desc_pca192_a1000",   2, 512,  True,  192, 1000.0, 0.47873),
    # Same PCA, lower alpha
    ("r2_512+desc_pca128_a100",    2, 512,  True,  128,  100.0, 0.47873),
    # More PCA components
    ("r2_512+desc_pca256_a1000",   2, 512,  True,  256, 1000.0, 0.47883),
    # No descriptors — different feature view
    ("r2_512_pca128_a1000",        2, 512,  False, 128, 1000.0, 0.47900),
    # Wider fingerprint — different structural view
    ("r2_2048+desc_pca128_a1000",  2, 2048, True,  128, 1000.0, 0.47900),
]


def build_fp_matrix(
    ids: list[str],
    chem_idx: pd.DataFrame,
    fp_radius: int,
    fp_bits: int,
    use_desc: bool,
) -> np.ndarray:
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_bits)
    rows = []
    for uid in ids:
        if uid not in chem_idx.index:
            dim = fp_bits + (len(DESC_COLS) if use_desc else 0)
            rows.append(np.zeros(dim, dtype=np.float32))
            continue
        row = chem_idx.loc[uid]
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        fp = qnu.bitvect_to_array(fp_gen.GetFingerprint(mol), fp_bits) if mol else \
             np.zeros(fp_bits, dtype=np.float32)
        if use_desc:
            desc = pd.to_numeric(row[DESC_COLS], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
            rows.append(np.concatenate([fp, desc]))
        else:
            rows.append(fp)
    return np.vstack(rows).astype(np.float32)


def train_and_predict(
    scope_ids: list[str],
    y_train: np.ndarray,
    test_query: pd.DataFrame,
    chem_idx: pd.DataFrame,
    fp_radius: int,
    fp_bits: int,
    use_desc: bool,
    n_components: int,
    alpha: float,
    label: str,
) -> np.ndarray:
    print(f"  [{label}] Building features...", flush=True)
    x_train = build_fp_matrix(scope_ids, chem_idx, fp_radius, fp_bits, use_desc)

    # Standardise
    x_mean  = x_train.mean(axis=0, keepdims=True)
    x_scale = x_train.std(axis=0, keepdims=True)
    x_scale[x_scale < 1e-6] = 1.0
    x_train_s = ((x_train - x_mean) / x_scale).astype(np.float32)

    # Target PCA
    y_mean = y_train.mean(axis=0, keepdims=True).astype(np.float32)
    y_c    = y_train - y_mean
    n_comp = min(n_components, y_c.shape[0] - 1, y_c.shape[1] - 1)
    u, s, vt = randomized_svd(y_c, n_components=n_comp, n_iter=5, random_state=SEED)
    pc_train = (u * s[None, :]).astype(np.float32)

    # Ridge
    print(f"  [{label}] Fitting Ridge (n_comp={n_comp}, alpha={alpha})...", flush=True)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(x_train_s, pc_train)

    # Test features
    test_ids = []
    x_test_rows = []
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_bits)
    for _, row in test_query.iterrows():
        smi   = str(row.get("smiles", ""))
        canon = qnu.canonical_smiles(smi) if smi else None
        mol   = Chem.MolFromSmiles(canon) if canon else None
        if mol is None:
            dim = fp_bits + (len(DESC_COLS) if use_desc else 0)
            x_test_rows.append(np.zeros(dim, dtype=np.float32))
        else:
            fp = qnu.bitvect_to_array(fp_gen.GetFingerprint(mol), fp_bits)
            if use_desc:
                desc_vals = []
                for col in DESC_COLS:
                    try:
                        desc_vals.append(float(row[col]) if col in row.index else 0.0)
                    except (ValueError, TypeError):
                        desc_vals.append(0.0)
                x_test_rows.append(np.concatenate([fp, np.array(desc_vals, dtype=np.float32)]))
            else:
                x_test_rows.append(fp)

    x_test   = np.vstack(x_test_rows).astype(np.float32)
    x_test_s = ((x_test - x_mean) / x_scale).astype(np.float32)

    pred_pc   = ridge.predict(x_test_s).astype(np.float32)
    pred_expr = np.clip(pred_pc @ vt[:n_comp] + y_mean, 0.0, None).astype(np.float32)
    print(f"  [{label}] Done. pred shape={pred_expr.shape}", flush=True)
    return pred_expr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output",   default="submission_ensemble.parquet")
    p.add_argument("--submit",   action="store_true")
    p.add_argument("--weighted", action="store_true",
                   help="Use inverse-wmse weighting instead of simple mean")
    p.add_argument("--members",  nargs="*",
                   help="Subset of member labels to include (default: all)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    members = ENSEMBLE_MEMBERS
    if args.members:
        members = [m for m in ENSEMBLE_MEMBERS if m[0] in args.members]
    print(f"Ensemble: {len(members)} members", flush=True)

    print("Loading inputs...", flush=True)
    train_chem, train_meta, query, _qpath, gene_filter, weight_cols = qnu.load_inputs()
    train_meta["user_compound_id"] = train_meta["user_compound_id"].astype(str)

    # Build training scope (qnu-ref active compounds)
    active_mask = qnu.target_active_mask(train_meta)
    scope_mask  = active_mask & (train_meta["job_id"] == qnu.QNU_JOB_ID)
    active_ids  = set(train_meta.loc[active_mask, "user_compound_id"])

    all_ids, _ = qnu.build_chemistry_maps(train_chem, active_ids, weight_cols)
    all_id_set = set(all_ids)

    scope_ids = sorted(
        set(train_meta.loc[scope_mask, "user_compound_id"].astype(str)) & all_id_set
    )
    print(f"Training compounds (qnu-ref scope): {len(scope_ids)}", flush=True)

    # Expression matrix
    print("Building expression matrix...", flush=True)
    expr_all = qnu.expression_wide(train_meta, all_ids, gene_filter)
    y_train  = expr_all[scope_ids].T.to_numpy(dtype=np.float32)
    gene_ids = list(gene_filter)

    # Chemistry index
    chem_idx = (
        train_chem.copy()
        .assign(user_id=lambda d: d["user_compound_id"].astype(str))
        .drop_duplicates("user_id")
        .set_index("user_id")
    )

    compound_ids = (
        query["compound"].astype(str).tolist()
        if "compound" in query.columns
        else [str(query.index[i]) for i in range(len(query))]
    )

    # ── Train each ensemble member ────────────────────────────────────────────
    all_preds = []
    all_weights = []

    for label, fp_radius, fp_bits, use_desc, n_comp, alpha, holdout_wmse in members:
        print(f"\nMember: {label}", flush=True)
        pred = train_and_predict(
            scope_ids=scope_ids,
            y_train=y_train,
            test_query=query,
            chem_idx=chem_idx,
            fp_radius=fp_radius,
            fp_bits=fp_bits,
            use_desc=use_desc,
            n_components=n_comp,
            alpha=alpha,
            label=label,
        )
        all_preds.append(pred)
        # Inverse-wmse weight: lower wmse → higher weight
        all_weights.append(1.0 / holdout_wmse)

    # ── Combine ───────────────────────────────────────────────────────────────
    preds_stack = np.stack(all_preds, axis=0)  # (n_members, n_test, n_genes)

    if args.weighted:
        w = np.array(all_weights, dtype=np.float32)
        w = w / w.sum()
        print(f"\nWeighted ensemble weights: {dict(zip([m[0] for m in members], w.round(4)))}", flush=True)
        pred_final = (preds_stack * w[:, None, None]).sum(axis=0)
    else:
        pred_final = preds_stack.mean(axis=0)
        print(f"\nSimple mean ensemble of {len(members)} members", flush=True)

    pred_final = np.clip(pred_final, 0.0, None).astype(np.float32)

    # ── Build submission ──────────────────────────────────────────────────────
    print(f"\nBuilding submission: {len(compound_ids)} compounds × {len(gene_ids)} genes", flush=True)
    submission = pd.DataFrame(pred_final, index=compound_ids, columns=gene_ids)
    submission.index.name = "compound"

    out_path = ROOT / args.output
    submission.to_parquet(out_path)
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    if args.submit:
        print("\nSubmitting to contest API...", flush=True)
        try:
            from vcpi_prediction_contest import submit
            result = submit(str(out_path))
            print(f"Submission result: {result}", flush=True)
        except Exception as e:
            print(f"Submission failed: {e}", flush=True)
            print(f"  python -c \"from vcpi_prediction_contest import submit; submit('{out_path}')\"",
                  flush=True)


if __name__ == "__main__":
    main()
