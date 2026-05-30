#!/usr/bin/env python3
"""
Generate submission parquet files using extended fingerprint feature vectors.

Variant 1 (submission_maccs.parquet):
    MACCS 167-bit + Morgan512 + 8 descriptors = 687-dim

Variant 2 (submission_rdkfp.parquet):
    RDKit topological FP 2048-bit + Morgan512 + 8 descriptors = 2568-dim

Both use Ridge + PCA128 + alpha=1000, qnu-ref scope, trained on all training
data, predicting 1064 test compounds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from sklearn.linear_model import Ridge
from sklearn.utils.extmath import randomized_svd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import qnu_plate_holdout_eval as qnu

SEED = 13
DESC_COLS = qnu.DESC_COLS
N_COMP = 128
ALPHA = 1000.0
MORGAN_BITS = 512
MORGAN_RADIUS = 2
MACCS_BITS = 167
RDKFP_BITS = 2048


def get_mol(smiles: str) -> Chem.Mol | None:
    canon = qnu.canonical_smiles(smiles) if smiles else None
    if canon is None:
        return None
    return Chem.MolFromSmiles(canon)


def compute_maccs_morgan_desc(
    mol: Chem.Mol | None,
    morgan_gen: rdFingerprintGenerator.FingeprintGenerator64,
    desc_vals: np.ndarray,
) -> np.ndarray:
    """Return 167 + 512 + 8 = 687-dim vector."""
    if mol is None:
        return np.zeros(MACCS_BITS + MORGAN_BITS + len(DESC_COLS), dtype=np.float32)
    maccs = np.zeros(MACCS_BITS, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), maccs)
    morgan = qnu.bitvect_to_array(morgan_gen.GetFingerprint(mol), MORGAN_BITS)
    return np.concatenate([maccs, morgan, desc_vals])


def compute_rdkfp_morgan_desc(
    mol: Chem.Mol | None,
    rdk_gen: rdFingerprintGenerator.FingeprintGenerator64,
    morgan_gen: rdFingerprintGenerator.FingeprintGenerator64,
    desc_vals: np.ndarray,
) -> np.ndarray:
    """Return 2048 + 512 + 8 = 2568-dim vector."""
    if mol is None:
        return np.zeros(RDKFP_BITS + MORGAN_BITS + len(DESC_COLS), dtype=np.float32)
    rdk = np.zeros(RDKFP_BITS, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(rdk_gen.GetFingerprint(mol), rdk)
    morgan = qnu.bitvect_to_array(morgan_gen.GetFingerprint(mol), MORGAN_BITS)
    return np.concatenate([rdk, morgan, desc_vals])


def build_train_features(scope_ids: list[str], chem_idx: pd.DataFrame, variant: str,
                         morgan_gen, rdk_gen=None) -> np.ndarray:
    rows = []
    for uid in scope_ids:
        row = chem_idx.loc[uid]
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        desc = pd.to_numeric(row[DESC_COLS], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        if variant == "maccs":
            rows.append(compute_maccs_morgan_desc(mol, morgan_gen, desc))
        else:
            rows.append(compute_rdkfp_morgan_desc(mol, rdk_gen, morgan_gen, desc))
    return np.vstack(rows).astype(np.float32)


def build_test_features(query: pd.DataFrame, variant: str,
                        morgan_gen, rdk_gen=None) -> tuple[np.ndarray, list[int]]:
    rows = []
    fallback_ids = []
    for i, (_, row) in enumerate(query.iterrows()):
        smi = str(row.get("smiles", ""))
        mol = get_mol(smi)
        if mol is None:
            fallback_ids.append(i)
        desc_vals = []
        for col in DESC_COLS:
            try:
                desc_vals.append(float(row[col]) if col in row.index else 0.0)
            except (ValueError, TypeError):
                desc_vals.append(0.0)
        desc = np.array(desc_vals, dtype=np.float32)
        if variant == "maccs":
            rows.append(compute_maccs_morgan_desc(mol, morgan_gen, desc))
        else:
            rows.append(compute_rdkfp_morgan_desc(mol, rdk_gen, morgan_gen, desc))
    return np.vstack(rows).astype(np.float32), fallback_ids


def run_variant(
    variant: str,
    scope_ids: list[str],
    chem_idx: pd.DataFrame,
    y_train: np.ndarray,
    query: pd.DataFrame,
    gene_filter: list[str],
    out_path: Path,
) -> None:
    print(f"\n{'='*60}", flush=True)
    print(f"Variant: {variant.upper()}", flush=True)

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
    )
    rdk_gen = None
    if variant == "rdkfp":
        rdk_gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=RDKFP_BITS)

    # --- Training features ---
    feat_dim = (MACCS_BITS + MORGAN_BITS + len(DESC_COLS)) if variant == "maccs" \
               else (RDKFP_BITS + MORGAN_BITS + len(DESC_COLS))
    print(f"Computing {variant} training features ({len(scope_ids)} compounds, {feat_dim}-dim)...",
          flush=True)
    x_train = build_train_features(scope_ids, chem_idx, variant, morgan_gen, rdk_gen)

    # --- Target PCA ---
    print(f"Fitting PCA (n_components={N_COMP})...", flush=True)
    y_mean = y_train.mean(axis=0, keepdims=True).astype(np.float32)
    y_centered = (y_train - y_mean).astype(np.float32)
    n_comp = min(N_COMP, y_centered.shape[0] - 1, y_centered.shape[1] - 1)
    u, s, vt = randomized_svd(y_centered, n_components=n_comp, n_iter=5, random_state=SEED)
    pc_train = (u * s[None, :]).astype(np.float32)

    # --- Standardise features ---
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_scale = x_train.std(axis=0, keepdims=True)
    x_scale[x_scale < 1e-6] = 1.0
    x_train_s = ((x_train - x_mean) / x_scale).astype(np.float32)

    # --- Fit Ridge ---
    print(f"Fitting Ridge (alpha={ALPHA})...", flush=True)
    ridge = Ridge(alpha=ALPHA, fit_intercept=True)
    ridge.fit(x_train_s, pc_train[:, :n_comp])

    # --- Test features ---
    print(f"Computing {variant} test features ({len(query)} compounds)...", flush=True)
    x_test, fallback_ids = build_test_features(query, variant, morgan_gen, rdk_gen)
    x_test_s = ((x_test - x_mean) / x_scale).astype(np.float32)

    if fallback_ids:
        print(f"  WARNING: {len(fallback_ids)} test compounds had invalid SMILES — "
              f"using training mean as fallback", flush=True)

    # --- Predict ---
    print("Predicting test compounds...", flush=True)
    pred_pc = ridge.predict(x_test_s).astype(np.float32)
    pred_expr = np.clip(pred_pc @ vt[:n_comp] + y_mean, 0.0, None).astype(np.float32)

    for idx in fallback_ids:
        pred_expr[idx] = y_mean[0]

    # --- Build submission ---
    compound_ids = query["compound"].astype(str).tolist() if "compound" in query.columns \
                   else [str(query.index[i]) for i in range(len(query))]

    print(f"Building submission: {len(compound_ids)} compounds x {len(gene_filter)} genes",
          flush=True)
    submission = pd.DataFrame(pred_expr, index=compound_ids, columns=gene_filter)
    submission.index.name = "compound"

    submission.to_parquet(out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {out_path} ({size_mb:.1f} MB)", flush=True)
    print(f"Config: variant={variant}, feat_dim={feat_dim}, pca={n_comp}, alpha={ALPHA}",
          flush=True)


def main() -> None:
    print("Loading inputs...", flush=True)
    train_chem, train_meta, query, _qpath, gene_filter, weight_cols = qnu.load_inputs()
    train_meta["user_compound_id"] = train_meta["user_compound_id"].astype(str)

    # --- Build training set (qnu-ref scope, same as generate_submission.py) ---
    active_mask = qnu.target_active_mask(train_meta)
    active_ids = set(train_meta.loc[active_mask, "user_compound_id"])

    scope_mask = active_mask & (train_meta["job_id"] == qnu.QNU_JOB_ID)
    print("Reference scope: tvc-qnu-012 only", flush=True)

    all_ids, all_fps = qnu.build_chemistry_maps(train_chem, active_ids, weight_cols)
    all_id_set = set(all_ids)

    scope_ids = sorted(
        set(train_meta.loc[scope_mask, "user_compound_id"].astype(str)) & all_id_set
    )
    print(f"Training compounds: {len(scope_ids)}", flush=True)

    # --- Expression matrix ---
    print("Building expression matrix...", flush=True)
    expr_all = qnu.expression_wide(train_meta, all_ids, gene_filter)
    y_train = expr_all[scope_ids].T.to_numpy(dtype=np.float32)   # (n_train, n_genes)

    # --- Chemistry index ---
    chem_idx = (
        train_chem.copy()
        .assign(user_id=lambda d: d["user_compound_id"].astype(str))
        .drop_duplicates("user_id")
        .set_index("user_id")
    )

    # --- Run both variants ---
    run_variant(
        variant="maccs",
        scope_ids=scope_ids,
        chem_idx=chem_idx,
        y_train=y_train,
        query=query,
        gene_filter=gene_filter,
        out_path=ROOT / "submission_maccs.parquet",
    )

    run_variant(
        variant="rdkfp",
        scope_ids=scope_ids,
        chem_idx=chem_idx,
        y_train=y_train,
        query=query,
        gene_filter=gene_filter,
        out_path=ROOT / "submission_rdkfp.parquet",
    )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
