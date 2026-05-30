#!/usr/bin/env python3
"""
Generate submission_torsion.parquet using Topological Torsion FP (2048-bit) + Morgan FP (512-bit) + 8 descriptors.
Total feature dimensions: 2048 + 512 + 8 = 2568.

Model: Ridge + target-PCA (128 components, alpha=1000), trained on qnu-ref scope.
"""

from __future__ import annotations

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

TT_BITS     = 2048
MORGAN_BITS = 512
TOTAL_BITS  = TT_BITS + MORGAN_BITS + len(DESC_COLS)  # 2568

N_COMPONENTS = 128
ALPHA        = 1000.0


def main() -> None:
    print("Loading inputs...", flush=True)
    train_chem, train_meta, query, _qpath, gene_filter, weight_cols = qnu.load_inputs()
    train_meta["user_compound_id"] = train_meta["user_compound_id"].astype(str)

    # ── Build training set ────────────────────────────────────────────────────
    active_mask = qnu.target_active_mask(train_meta)
    scope_mask  = active_mask & (train_meta["job_id"] == qnu.QNU_JOB_ID)
    print("Reference scope: tvc-qnu-012 only", flush=True)

    active_ids  = set(train_meta.loc[active_mask, "user_compound_id"])
    all_ids, _  = qnu.build_chemistry_maps(train_chem, active_ids, weight_cols)
    all_id_set  = set(all_ids)

    scope_ids = sorted(
        set(train_meta.loc[scope_mask, "user_compound_id"].astype(str)) & all_id_set
    )
    print(f"Training compounds: {len(scope_ids)}", flush=True)

    # ── Expression matrix ─────────────────────────────────────────────────────
    print("Building expression matrix...", flush=True)
    expr_all = qnu.expression_wide(train_meta, all_ids, gene_filter)
    y_train  = expr_all[scope_ids].T.to_numpy(dtype=np.float32)   # (n_train, n_genes)

    # ── Fingerprint generators ────────────────────────────────────────────────
    tt_gen     = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=TT_BITS)
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=MORGAN_BITS)

    chem_idx = (
        train_chem.copy()
        .assign(user_id=lambda d: d["user_compound_id"].astype(str))
        .drop_duplicates("user_id")
        .set_index("user_id")
    )

    print(
        f"Computing TopologicalTorsion({TT_BITS}bit) + Morgan({MORGAN_BITS}bit) + {len(DESC_COLS)} descriptors "
        f"for {len(scope_ids)} training compounds...",
        flush=True,
    )
    x_train_rows = []
    for uid in scope_ids:
        row  = chem_idx.loc[uid]
        mol  = Chem.MolFromSmiles(str(row["smiles"]))
        if mol:
            tt_fp     = qnu.bitvect_to_array(tt_gen.GetFingerprint(mol), TT_BITS)
            morgan_fp = qnu.bitvect_to_array(morgan_gen.GetFingerprint(mol), MORGAN_BITS)
        else:
            tt_fp     = np.zeros(TT_BITS,    dtype=np.float32)
            morgan_fp = np.zeros(MORGAN_BITS, dtype=np.float32)
        desc = pd.to_numeric(row[DESC_COLS], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        x_train_rows.append(np.concatenate([tt_fp, morgan_fp, desc]))
    x_train = np.vstack(x_train_rows).astype(np.float32)
    print(f"x_train shape: {x_train.shape}", flush=True)

    # ── Target PCA ───────────────────────────────────────────────────────────
    print(f"Fitting PCA (n_components={N_COMPONENTS})...", flush=True)
    y_mean    = y_train.mean(axis=0, keepdims=True).astype(np.float32)
    y_centered = y_train - y_mean
    n_comp    = min(N_COMPONENTS, y_centered.shape[0] - 1, y_centered.shape[1] - 1)
    u, s, vt  = randomized_svd(y_centered, n_components=n_comp, n_iter=5, random_state=SEED)
    pc_train  = (u * s[None, :]).astype(np.float32)

    # ── Standardise features ─────────────────────────────────────────────────
    x_mean  = x_train.mean(axis=0, keepdims=True)
    x_scale = x_train.std(axis=0, keepdims=True)
    x_scale[x_scale < 1e-6] = 1.0
    x_train_s = ((x_train - x_mean) / x_scale).astype(np.float32)

    # ── Fit Ridge ────────────────────────────────────────────────────────────
    print(f"Fitting Ridge (alpha={ALPHA})...", flush=True)
    ridge = Ridge(alpha=ALPHA, fit_intercept=True)
    ridge.fit(x_train_s, pc_train[:, :n_comp])

    # ── Test compound features ────────────────────────────────────────────────
    print(f"Computing test compound features ({len(query)} compounds)...", flush=True)
    x_test_rows = []
    fallback_ids = []
    for i, (_, row) in enumerate(query.iterrows()):
        smi   = str(row.get("smiles", ""))
        canon = qnu.canonical_smiles(smi) if smi else None
        mol   = Chem.MolFromSmiles(canon) if canon else None
        if mol is None:
            fallback_ids.append(i)
            x_test_rows.append(np.zeros(TOTAL_BITS, dtype=np.float32))
        else:
            tt_fp     = qnu.bitvect_to_array(tt_gen.GetFingerprint(mol), TT_BITS)
            morgan_fp = qnu.bitvect_to_array(morgan_gen.GetFingerprint(mol), MORGAN_BITS)
            desc_vals = []
            for col in DESC_COLS:
                try:
                    desc_vals.append(float(row[col]) if col in row.index else 0.0)
                except (ValueError, TypeError):
                    desc_vals.append(0.0)
            x_test_rows.append(np.concatenate([tt_fp, morgan_fp, np.array(desc_vals, dtype=np.float32)]))

    x_test   = np.vstack(x_test_rows).astype(np.float32)
    x_test_s = ((x_test - x_mean) / x_scale).astype(np.float32)

    if fallback_ids:
        print(
            f"  WARNING: {len(fallback_ids)} test compounds had invalid SMILES — using zeros",
            flush=True,
        )

    # ── Predict ───────────────────────────────────────────────────────────────
    print("Predicting test compounds...", flush=True)
    pred_pc   = ridge.predict(x_test_s).astype(np.float32)
    pred_expr = np.clip(pred_pc @ vt[:n_comp] + y_mean, 0.0, None).astype(np.float32)

    if fallback_ids:
        for idx in fallback_ids:
            pred_expr[idx] = y_mean[0]

    # ── Build submission DataFrame ────────────────────────────────────────────
    gene_ids     = list(gene_filter)
    compound_ids = (
        query["compound"].astype(str).tolist()
        if "compound" in query.columns
        else [str(query.index[i]) for i in range(len(query))]
    )

    print(f"Building submission: {len(compound_ids)} compounds x {len(gene_ids)} genes", flush=True)
    submission = pd.DataFrame(pred_expr, index=compound_ids, columns=gene_ids)
    submission.index.name = "compound"

    out_path = ROOT / "submission_torsion.parquet"
    submission.to_parquet(out_path)
    print(f"\nWrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)", flush=True)
    print(
        f"Config: scope=qnu, tt_bits={TT_BITS}, morgan_bits={MORGAN_BITS}, "
        f"n_desc={len(DESC_COLS)}, total_dims={TOTAL_BITS}, pca={n_comp}, alpha={ALPHA}",
        flush=True,
    )


if __name__ == "__main__":
    main()
