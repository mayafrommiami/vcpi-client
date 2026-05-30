#!/usr/bin/env python3
"""
Local head-to-head evaluation of all fingerprint variants using the
3-plate qnu holdout framework.

Loads data once, scores all FP variants at PCA128 + Ridge alpha=1000,
reports wMSE per plate and mean.

Run:
    /opt/homebrew/bin/python3.13 scripts/eval_fp_comparison.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from sklearn.linear_model import Ridge
from sklearn.utils.extmath import randomized_svd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import qnu_plate_holdout_eval as qnu
import qnu_plate_extended_model_sweep as sweep

SEED = 13
N_COMP = 128
ALPHA  = 1000.0


# ── Feature builders ──────────────────────────────────────────────────────────

def _morgan_gen(radius=2, nbits=512):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)

def _rdkit_gen(nbits=2048):
    return rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=nbits)

def _atompair_gen(nbits=2048):
    return rdFingerprintGenerator.GetAtomPairGenerator(fpSize=nbits)

def _torsion_gen(nbits=2048):
    return rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=nbits)


def build_fp_dict(chem_df: pd.DataFrame, all_ids: list[str], generators: list, desc: bool = True):
    """Build {uid: feature_vector} for each compound."""
    chem = chem_df.copy()
    chem["user_id"] = chem["user_compound_id"].astype(str)
    chem = chem[chem["user_id"].isin(all_ids)].drop_duplicates("user_id").set_index("user_id")
    desc_frame = chem[qnu.DESC_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)

    result = {}
    for uid, row in chem.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None:
            continue
        parts = []
        for gen, nbits in generators:
            parts.append(qnu.bitvect_to_array(gen.GetFingerprint(mol), nbits))
        if desc:
            parts.append(desc_frame.loc[uid].to_numpy(dtype=np.float32))
        result[uid] = np.concatenate(parts).astype(np.float32)
    return result


# ── FP variant definitions: (name, generators_list, use_desc) ─────────────────
FP_VARIANTS = [
    ("Morgan512+desc",           [(_morgan_gen(2, 512), 512)],                             True),
    ("MACCS167+Morgan512+desc",  [("maccs", 167), (_morgan_gen(2, 512), 512)],              True),
    ("RDKitFP2048+Morgan512+desc",[(_rdkit_gen(2048), 2048), (_morgan_gen(2, 512), 512)],  True),
    ("AtomPair2048+Morgan512+desc",[(_atompair_gen(2048), 2048), (_morgan_gen(2, 512), 512)], True),
    ("Torsion2048+Morgan512+desc",[(_torsion_gen(2048), 2048), (_morgan_gen(2, 512), 512)], True),
    ("Morgan2048+desc",          [(_morgan_gen(2, 2048), 2048)],                            True),
]


def build_fp_dict_v2(chem_df, all_ids, variant_generators, use_desc):
    """Handle the MACCS special case (not a generator-style API)."""
    chem = chem_df.copy()
    chem["user_id"] = chem["user_compound_id"].astype(str)
    chem = chem[chem["user_id"].isin(all_ids)].drop_duplicates("user_id").set_index("user_id")
    desc_frame = chem[qnu.DESC_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)

    result = {}
    for uid, row in chem.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None:
            continue
        parts = []
        for gen_or_str, nbits in variant_generators:
            if gen_or_str == "maccs":
                bv = MACCSkeys.GenMACCSKeys(mol)
                arr = np.zeros(nbits, dtype=np.float32)
                for bit in bv.GetOnBits():
                    if bit < nbits:
                        arr[bit] = 1.0
                parts.append(arr)
            else:
                parts.append(qnu.bitvect_to_array(gen_or_str.GetFingerprint(mol), nbits))
        if use_desc:
            parts.append(desc_frame.loc[uid].to_numpy(dtype=np.float32))
        result[uid] = np.concatenate(parts).astype(np.float32)
    return result


def robust_std(x_train, x_val):
    mu  = x_train.mean(axis=0, keepdims=True)
    sig = x_train.std(axis=0, keepdims=True)
    sig[sig < 1e-6] = 1.0
    return ((x_train - mu) / sig).astype(np.float32), ((x_val - mu) / sig).astype(np.float32)


def eval_variant(name, fp_dict, ref_ids, val_ids, y_all, all_id_index, fallback_mean,
                 truth, weights, gene_filter):
    ref_cov = [uid for uid in ref_ids if uid in fp_dict]
    val_cov = [uid for uid in val_ids if uid in fp_dict]
    if len(ref_cov) < 20 or not val_cov:
        return None

    x_ref = np.vstack([fp_dict[uid] for uid in ref_cov])
    x_val = np.vstack([fp_dict[uid] for uid in val_cov])
    x_ref_s, x_val_s = robust_std(x_ref, x_val)

    ref_pos = np.array([all_id_index[uid] for uid in ref_cov])
    y_ref   = y_all[ref_pos]
    y_mean  = y_ref.mean(axis=0, keepdims=True).astype(np.float32)
    y_c     = y_ref - y_mean
    n_comp  = min(N_COMP, y_c.shape[0] - 1, y_c.shape[1] - 1)

    u, s, vt = randomized_svd(y_c, n_components=n_comp, n_iter=5, random_state=SEED)
    pc_ref   = (u * s[None, :]).astype(np.float32)

    ridge = Ridge(alpha=ALPHA, fit_intercept=True)
    ridge.fit(x_ref_s, pc_ref)
    pred_pc = ridge.predict(x_val_s).astype(np.float32)
    pred_cov = np.clip(pred_pc @ vt[:n_comp] + y_mean, 0.0, None)  # (n_val_cov, n_genes)

    # Build full pred DataFrame (genes × compounds) using fallback for missing
    pred_df = pd.DataFrame(
        np.broadcast_to(fallback_mean[None, :], (len(val_ids), len(fallback_mean))).copy(),
        index=val_ids, columns=gene_filter,
    )
    for i, uid in enumerate(val_cov):
        pred_df.loc[uid] = pred_cov[i]
    pred_df = pred_df.T  # genes × compounds

    return float(qnu.score_wmse(truth, pred_df, weights).mean())


def main():
    print("Loading inputs...", flush=True)
    train_chem, train_meta, query, _qpath, gene_filter, weight_cols = qnu.load_inputs()
    train_meta["user_compound_id"] = train_meta["user_compound_id"].astype(str)

    active_mask = qnu.target_active_mask(train_meta)
    active_ids  = set(train_meta.loc[active_mask, "user_compound_id"])
    all_ids, all_fps = qnu.build_chemistry_maps(train_chem, active_ids, weight_cols)
    all_id_index = {uid: i for i, uid in enumerate(all_ids)}

    qnu_active = train_meta[
        (train_meta["job_id"] == qnu.QNU_JOB_ID) & active_mask
        & train_meta["user_compound_id"].isin(set(all_ids))
    ]

    print("Aggregating expression matrix...", flush=True)
    expr_all = qnu.expression_wide(train_meta, all_ids, gene_filter)
    y_all    = expr_all[all_ids].T.to_numpy(dtype=np.float32)

    # Load weights matrix
    w_path = ROOT / "weights.parquet"
    w_df   = pd.read_parquet(w_path)
    if "gene_id" in w_df.columns:
        w_df = w_df.set_index("gene_id")
    gene_list = list(gene_filter)

    # Select holdout plates
    plates, _ = qnu.select_plates(args=type("A", (), {"plates": None,
                                                       "plate_selection": "similarity",
                                                       "num_plates": 3,
                                                       "max_plates": None,
                                                       "target_components": 64,
                                                       "pls_components": 16,
                                                       "min_plate_size": 20,
                                                       "plate_similarity_top_k": 10})(),
                                  qnu_active=qnu_active, all_fps=all_fps, query=query)

    # Pre-build all fp_dicts once
    print("\nBuilding fingerprint dictionaries...", flush=True)
    fp_dicts = {}
    for vname, generators, use_desc in FP_VARIANTS:
        print(f"  {vname}...", flush=True)
        fp_dicts[vname] = build_fp_dict_v2(train_chem, all_ids, generators, use_desc)

    results = {vname: [] for vname, *_ in FP_VARIANTS}

    plate_meta_all = train_meta[train_meta["user_compound_id"].isin(set(all_ids))]

    for plate_id in plates:
        val_ids = sorted(
            qnu_active.loc[qnu_active["container_id"] == plate_id, "user_compound_id"]
            .astype(str).unique()
        )
        ref_ids = sorted(
            set(train_meta.loc[
                active_mask & (train_meta["job_id"] == qnu.QNU_JOB_ID)
                & (train_meta["container_id"] != plate_id),
                "user_compound_id"
            ].astype(str)) & set(all_ids)
        )
        fallback_mean = y_all[[all_id_index[uid] for uid in ref_ids if uid in all_id_index]].mean(axis=0)

        plate_meta = plate_meta_all[
            plate_meta_all["user_compound_id"].isin(val_ids)
            & (plate_meta_all["container_id"] == plate_id)
        ]
        truth   = qnu.plate_expression_wide(plate_meta, gene_list)
        truth   = truth.reindex(index=gene_list, columns=val_ids)
        weights = qnu.load_weights(gene_list, val_ids)

        print(f"\nPlate {plate_id}: {len(val_ids)} val, {len(ref_ids)} ref", flush=True)
        for vname, generators, use_desc in FP_VARIANTS:
            score = eval_variant(vname, fp_dicts[vname], ref_ids, val_ids,
                                 y_all, all_id_index, fallback_mean, truth, weights, gene_list)
            if score is not None:
                results[vname].append(score)
                print(f"  {vname}: {score:.5f}", flush=True)

    print("\n" + "="*60)
    print("FINAL RESULTS (mean wMSE across 3 plates, lower=better):")
    print("="*60)
    rows = []
    for vname, scores in results.items():
        mean_score = np.mean(scores) if scores else float("nan")
        rows.append({"variant": vname, "n_plates": len(scores), "wmse_mean": mean_score})
    df = pd.DataFrame(rows).sort_values("wmse_mean")
    print(df.to_string(index=False))
    df.to_csv(ROOT / "eval_fp_comparison_results.csv", index=False)
    print(f"\nSaved to {ROOT}/eval_fp_comparison_results.csv")


if __name__ == "__main__":
    main()
