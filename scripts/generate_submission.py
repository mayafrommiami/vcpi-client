#!/usr/bin/env python3
"""
Generate submission.parquet for the VCPI Drug-seq contest using the best
Ridge + target-PCA model found in the sweep.

Trains on ALL available training data (no holdout), then predicts the
1,064 test compounds.

Usage:
    # Default: best known config (qnu-ref scope, morgan512+desc, pca128, alpha=1000)
    python scripts/generate_submission.py

    # Override config
    python scripts/generate_submission.py \\
        --alpha 1000 --n-components 128 --reference-scope qnu \\
        --output submission.parquet

    # Submit after generating
    python scripts/generate_submission.py --submit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import Ridge
from sklearn.utils.extmath import randomized_svd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import qnu_plate_holdout_eval as qnu


SEED = 13
DESC_COLS = qnu.DESC_COLS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha",           type=float, default=1000.0,
                   help="Ridge alpha (default: 1000)")
    p.add_argument("--n-components",    type=int,   default=128,
                   help="PCA target components (default: 128)")
    p.add_argument("--fp-bits",         type=int,   default=512,
                   help="Morgan FP bits (default: 512)")
    p.add_argument("--fp-radius",       type=int,   default=2,
                   help="Morgan FP radius (default: 2)")
    p.add_argument("--reference-scope", default="qnu",
                   choices=["qnu", "all"],
                   help="Training reference: qnu=tvc-qnu-012 only, all=all datasets")
    p.add_argument("--output",          default="submission.parquet",
                   help="Output file path")
    p.add_argument("--submit",          action="store_true",
                   help="Submit to contest API after generating")
    return p.parse_args()


def compute_features(smiles_series: pd.Series, fp_bits: int, fp_radius: int) -> tuple[np.ndarray, list[bool]]:
    """Compute Morgan FP + 8 descriptors for a series of SMILES. Returns (matrix, valid_mask)."""
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_bits)
    rows = []
    valid = []
    for smi in smiles_series:
        canon = qnu.canonical_smiles(str(smi)) if smi else None
        mol   = Chem.MolFromSmiles(canon) if canon else None
        if mol is None:
            rows.append(np.zeros(fp_bits + len(DESC_COLS), dtype=np.float32))
            valid.append(False)
        else:
            fp   = qnu.bitvect_to_array(fp_gen.GetFingerprint(mol), fp_bits)
            rows.append(fp)
            valid.append(True)
    return np.vstack(rows).astype(np.float32), valid


def main() -> None:
    args = parse_args()

    print("Loading inputs...", flush=True)
    train_chem, train_meta, query, _qpath, gene_filter, weight_cols = qnu.load_inputs()
    train_meta["user_compound_id"] = train_meta["user_compound_id"].astype(str)

    # ── Build training set ────────────────────────────────────────────────────
    active_mask = qnu.target_active_mask(train_meta)
    active_ids  = set(train_meta.loc[active_mask, "user_compound_id"])

    if args.reference_scope == "qnu":
        scope_mask = active_mask & (train_meta["job_id"] == qnu.QNU_JOB_ID)
        print("Reference scope: tvc-qnu-012 only", flush=True)
    else:
        scope_mask = active_mask
        print("Reference scope: all datasets", flush=True)

    all_ids, all_fps = qnu.build_chemistry_maps(train_chem, active_ids, weight_cols)
    all_id_set = set(all_ids)

    scope_ids = sorted(set(
        train_meta.loc[scope_mask, "user_compound_id"].astype(str)
    ) & all_id_set)
    print(f"Training compounds: {len(scope_ids)}", flush=True)

    # Expression matrix for training compounds
    print("Building expression matrix...", flush=True)
    expr_all  = qnu.expression_wide(train_meta, all_ids, gene_filter)
    y_train   = expr_all[scope_ids].T.to_numpy(dtype=np.float32)   # (n_train, n_genes)

    # ── Training features ─────────────────────────────────────────────────────
    print(f"Computing Morgan FP (r={args.fp_radius}, {args.fp_bits}bit) + descriptors...", flush=True)
    chem_idx = (
        train_chem.copy()
        .assign(user_id=lambda d: d["user_compound_id"].astype(str))
        .drop_duplicates("user_id")
        .set_index("user_id")
    )

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=args.fp_radius, fpSize=args.fp_bits)
    x_train_rows = []
    for uid in scope_ids:
        row  = chem_idx.loc[uid]
        mol  = Chem.MolFromSmiles(str(row["smiles"]))
        fp   = qnu.bitvect_to_array(fp_gen.GetFingerprint(mol), args.fp_bits) if mol else \
               np.zeros(args.fp_bits, dtype=np.float32)
        desc = pd.to_numeric(row[DESC_COLS], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        x_train_rows.append(np.concatenate([fp, desc]))
    x_train = np.vstack(x_train_rows).astype(np.float32)

    # ── Target PCA ───────────────────────────────────────────────────────────
    print(f"Fitting PCA (n_components={args.n_components})...", flush=True)
    y_mean    = y_train.mean(axis=0, keepdims=True).astype(np.float32)
    y_centered = y_train - y_mean
    n_comp    = min(args.n_components, y_centered.shape[0] - 1, y_centered.shape[1] - 1)
    u, s, vt  = randomized_svd(y_centered, n_components=n_comp, n_iter=5, random_state=SEED)
    pc_train  = (u * s[None, :]).astype(np.float32)

    # ── Standardise features ─────────────────────────────────────────────────
    x_mean  = x_train.mean(axis=0, keepdims=True)
    x_scale = x_train.std(axis=0, keepdims=True)
    x_scale[x_scale < 1e-6] = 1.0
    x_train_s = ((x_train - x_mean) / x_scale).astype(np.float32)

    # ── Fit Ridge ────────────────────────────────────────────────────────────
    print(f"Fitting Ridge (alpha={args.alpha})...", flush=True)
    ridge = Ridge(alpha=args.alpha, fit_intercept=True)
    ridge.fit(x_train_s, pc_train[:, :n_comp])

    # ── Test compound features ────────────────────────────────────────────────
    print(f"Computing test compound features ({len(query)} compounds)...", flush=True)
    test_smiles = query["smiles"].astype(str)
    x_test_rows = []
    fallback_ids = []
    for i, (_, row) in enumerate(query.iterrows()):
        smi  = str(row.get("smiles", ""))
        canon = qnu.canonical_smiles(smi) if smi else None
        mol  = Chem.MolFromSmiles(canon) if canon else None
        if mol is None:
            fallback_ids.append(i)
            x_test_rows.append(np.zeros(args.fp_bits + len(DESC_COLS), dtype=np.float32))
        else:
            fp   = qnu.bitvect_to_array(fp_gen.GetFingerprint(mol), args.fp_bits)
            # Descriptors from query df if available, else zeros
            desc_vals = []
            for col in DESC_COLS:
                try:
                    desc_vals.append(float(row[col]) if col in row.index else 0.0)
                except (ValueError, TypeError):
                    desc_vals.append(0.0)
            x_test_rows.append(np.concatenate([fp, np.array(desc_vals, dtype=np.float32)]))

    x_test   = np.vstack(x_test_rows).astype(np.float32)
    x_test_s = ((x_test - x_mean) / x_scale).astype(np.float32)

    if fallback_ids:
        print(f"  WARNING: {len(fallback_ids)} test compounds had invalid SMILES — "
              f"using training mean as fallback", flush=True)

    # ── Predict ───────────────────────────────────────────────────────────────
    print("Predicting test compounds...", flush=True)
    pred_pc   = ridge.predict(x_test_s).astype(np.float32)
    pred_expr = np.clip(pred_pc @ vt[:n_comp] + y_mean, 0.0, None).astype(np.float32)

    # Fallback to training mean for compounds with invalid SMILES
    if fallback_ids:
        for idx in fallback_ids:
            pred_expr[idx] = y_mean[0]

    # ── Build submission DataFrame ────────────────────────────────────────────
    gene_ids    = list(gene_filter)
    compound_ids = query["compound"].astype(str).tolist() if "compound" in query.columns \
                   else [str(query.index[i]) for i in range(len(query))]

    print(f"Building submission: {len(compound_ids)} compounds × {len(gene_ids)} genes", flush=True)
    submission = pd.DataFrame(
        pred_expr,
        index=compound_ids,
        columns=gene_ids,
    )
    submission.index.name = "compound"

    out_path = ROOT / args.output
    submission.to_parquet(out_path)
    print(f"\nWrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    print(f"Config: scope={args.reference_scope}, fp_bits={args.fp_bits}, "
          f"fp_radius={args.fp_radius}, pca={n_comp}, alpha={args.alpha}", flush=True)

    # ── Submit ────────────────────────────────────────────────────────────────
    if args.submit:
        print("\nSubmitting to contest API...", flush=True)
        try:
            from vcpi_prediction_contest import submit
            result = submit(str(out_path))
            print(f"Submission result: {result}", flush=True)
        except Exception as e:
            print(f"Submission failed: {e}", flush=True)
            print("You can submit manually with:", flush=True)
            print(f"  python -c \"from vcpi_prediction_contest import submit; submit('{out_path}')\"",
                  flush=True)


if __name__ == "__main__":
    main()
