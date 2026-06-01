#!/usr/bin/env python3
"""
Evaluate whether excluding edge wells from training expression improves wMSE.

Edge wells have ~16% lower UMI counts (less RNA captured). When a compound
has one edge and one interior replicate, the current pipeline averages them
equally. This script uses only interior-well replicates for training targets,
giving the Ridge model a cleaner signal.

Three conditions on the same 3-plate holdout:
  all_wells        — baseline, should reproduce 0.47753
  no_edge          — exclude is_edge=True replicates; all-edge compounds dropped
  no_edge_fallback — prefer interior replicates; fall back to edge if both are edge

Run:
    /opt/homebrew/bin/python3.13 scripts/qnu_no_edge_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from sklearn.linear_model import Ridge
from sklearn.utils.extmath import randomized_svd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import qnu_plate_holdout_eval as qnu

SEED   = 13
N_COMP = 128
ALPHA  = 1000.0


# ── Feature building (MACCS167 + Morgan512 + 8 desc = 687-dim) ────────────────

def build_fp_dict(train_chem: pd.DataFrame, all_ids: list[str]) -> dict[str, np.ndarray]:
    chem = train_chem.copy()
    chem["user_id"] = chem["user_compound_id"].astype(str)
    chem = chem[chem["user_id"].isin(all_ids)].drop_duplicates("user_id").set_index("user_id")
    desc_frame = chem[qnu.DESC_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    result: dict[str, np.ndarray] = {}
    for uid, row in chem.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None:
            continue
        bv = MACCSkeys.GenMACCSKeys(mol)
        maccs = np.zeros(167, dtype=np.float32)
        for bit in bv.GetOnBits():
            if bit < 167:
                maccs[bit] = 1.0
        morgan = qnu.bitvect_to_array(morgan_gen.GetFingerprint(mol), 512)
        desc   = desc_frame.loc[uid].to_numpy(dtype=np.float32) if uid in desc_frame.index \
                 else np.zeros(len(qnu.DESC_COLS), dtype=np.float32)
        result[uid] = np.concatenate([maccs, morgan, desc])
    return result


def _robust_std(x_tr, x_val):
    mu  = x_tr.mean(axis=0, keepdims=True)
    sig = x_tr.std(axis=0, keepdims=True)
    sig[sig < 1e-6] = 1.0
    return ((x_tr - mu) / sig).astype(np.float32), ((x_val - mu) / sig).astype(np.float32)


# ── Per-plate evaluation ───────────────────────────────────────────────────────

def eval_condition(
    *,
    label: str,
    plates: list[int],
    qnu_active: pd.DataFrame,
    expr: pd.DataFrame,          # genes × compounds
    all_ids: list[str],
    fp_dict: dict[str, np.ndarray],
    train_meta: pd.DataFrame,
    gene_filter: list[str],
) -> list[float]:
    all_id_index = {uid: i for i, uid in enumerate(all_ids)}
    y_all = expr[all_ids].T.to_numpy(dtype=np.float32)
    scores = []

    for plate_id in plates:
        val_ids = sorted(
            qnu_active.loc[qnu_active["container_id"] == plate_id, "user_compound_id"]
            .astype(str).unique()
        )
        val_set = set(val_ids)
        ref_ids = [uid for uid in all_ids if uid not in val_set]
        ref_cov = [uid for uid in ref_ids if uid in fp_dict]
        val_cov = [uid for uid in val_ids if uid in fp_dict]
        if len(ref_cov) < 20 or not val_cov:
            print(f"  [{label}] plate {plate_id}: insufficient coverage, skipping", flush=True)
            continue

        x_ref = np.vstack([fp_dict[uid] for uid in ref_cov])
        x_val = np.vstack([fp_dict[uid] for uid in val_cov])
        x_ref_s, x_val_s = _robust_std(x_ref, x_val)

        ref_pos = np.array([all_id_index[uid] for uid in ref_cov])
        y_ref   = y_all[ref_pos]
        y_mean  = y_ref.mean(axis=0, keepdims=True).astype(np.float32)
        y_c     = y_ref - y_mean
        n_comp  = min(N_COMP, y_c.shape[0] - 1, y_c.shape[1] - 1)
        u, s, vt = randomized_svd(y_c, n_components=n_comp, n_iter=5, random_state=SEED)
        pc_ref   = (u * s[None, :]).astype(np.float32)

        ridge    = Ridge(alpha=ALPHA, fit_intercept=True)
        ridge.fit(x_ref_s, pc_ref)
        pred_pc  = ridge.predict(x_val_s).astype(np.float32)
        pred_cov = np.clip(pred_pc @ vt[:n_comp] + y_mean, 0.0, None)

        # Truth always from all wells (matches contest scoring)
        plate_meta_val = train_meta[
            train_meta["user_compound_id"].astype(str).isin(val_ids)
            & (train_meta["container_id"] == plate_id)
        ]
        truth   = qnu.plate_expression_wide(plate_meta_val, gene_filter)
        truth   = truth.reindex(index=gene_filter, columns=val_ids)
        weights = qnu.load_weights(gene_filter, val_ids)

        fallback = y_all[[all_id_index[uid] for uid in ref_ids if uid in all_id_index]].mean(axis=0)
        pred_df  = pd.DataFrame(
            np.broadcast_to(fallback[None, :], (len(val_ids), len(fallback))).copy(),
            index=val_ids, columns=gene_filter,
        )
        for i, uid in enumerate(val_cov):
            pred_df.loc[uid] = pred_cov[i]
        pred_df = pred_df.T

        score = float(qnu.score_wmse(truth, pred_df, weights).mean())
        scores.append(score)
        print(f"  [{label}] plate {plate_id}: wMSE={score:.5f}  (ref={len(ref_cov)}, val={len(val_cov)})", flush=True)

    return scores


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading inputs...", flush=True)
    train_chem, train_meta, query, _qpath, gene_filter, weight_cols = qnu.load_inputs()
    train_meta["user_compound_id"] = train_meta["user_compound_id"].astype(str)
    train_meta["is_edge"] = train_meta["is_edge"].astype(bool)

    active_mask   = qnu.target_active_mask(train_meta)
    active_ids    = set(train_meta.loc[active_mask, "user_compound_id"])
    all_ids, all_fps = qnu.build_chemistry_maps(train_chem, active_ids, weight_cols)
    all_ids_set   = set(all_ids)

    qnu_active = train_meta[
        (train_meta["job_id"] == qnu.QNU_JOB_ID) & active_mask
        & train_meta["user_compound_id"].isin(all_ids_set)
    ].copy()

    mock_args = SimpleNamespace(
        plates=None, plate_selection="similarity", num_plates=3,
        max_plates=None, target_components=64, pls_components=16,
        min_plate_size=20, plate_similarity_top_k=10,
    )
    plates, _ = qnu.select_plates(args=mock_args, qnu_active=qnu_active, all_fps=all_fps, query=query)
    print(f"Holdout plates: {plates}", flush=True)

    # Build per-condition metadata filters
    # no_edge: drop edge replicates entirely (compounds with both edge → excluded from training)
    meta_no_edge = train_meta[~train_meta["is_edge"]]

    # no_edge_fallback: prefer interior; include edge only for all-edge compounds
    # Find compounds where ALL replicates are edge
    qnu_active_meta = train_meta[
        (train_meta["job_id"] == qnu.QNU_JOB_ID) & active_mask
        & train_meta["user_compound_id"].isin(all_ids_set)
    ]
    reps_per = qnu_active_meta.groupby("user_compound_id")["is_edge"].agg(["sum", "count"])
    all_edge_ids = set(reps_per[reps_per["sum"] == reps_per["count"]].index)
    print(f"Compounds with ALL replicates edge: {len(all_edge_ids)}", flush=True)
    # Fallback: non-edge rows PLUS all-edge compounds' edge rows
    meta_no_edge_fb = pd.concat([
        train_meta[~train_meta["is_edge"]],
        train_meta[train_meta["is_edge"] & train_meta["user_compound_id"].isin(all_edge_ids)],
    ])

    CONDITIONS = {
        "all_wells":        train_meta,
        "no_edge":          meta_no_edge,
        "no_edge_fallback": meta_no_edge_fb,
    }

    results = {}
    for label, meta_filter in CONDITIONS.items():
        print(f"\n{'='*55}", flush=True)
        print(f"CONDITION: {label}", flush=True)

        print("Computing expression matrix...", flush=True)
        expr = qnu.expression_wide(
            meta_filter[
                (meta_filter["job_id"] == qnu.QNU_JOB_ID) & active_mask
                & meta_filter["user_compound_id"].isin(all_ids_set)
            ],
            all_ids,
            gene_filter,
        )
        # Some compounds may be missing (all-edge dropped in no_edge)
        present = [uid for uid in all_ids if uid in expr.columns]
        expr    = expr[present]
        print(f"  Compounds with expression: {len(present)}/{len(all_ids)}", flush=True)

        print("Building feature dict...", flush=True)
        fp_dict = build_fp_dict(train_chem, present)

        scores = eval_condition(
            label=label,
            plates=plates,
            qnu_active=qnu_active,
            expr=expr,
            all_ids=present,
            fp_dict=fp_dict,
            train_meta=train_meta,   # always use full meta for truth
            gene_filter=gene_filter,
        )
        results[label] = scores

    print("\n" + "=" * 60, flush=True)
    print("FINAL RESULTS (mean wMSE, lower=better):", flush=True)
    print("=" * 60, flush=True)
    rows = []
    for label, scores in results.items():
        mean = float(np.mean(scores)) if scores else float("nan")
        rows.append({"condition": label, "n_plates": len(scores), "wmse_mean": round(mean, 5)})
    df = pd.DataFrame(rows).sort_values("wmse_mean")
    print(df.to_string(index=False), flush=True)
    df.to_csv(ROOT / "eval_no_edge_summary.csv", index=False)
    print("\nPrevious best (all wells): 0.47753", flush=True)


if __name__ == "__main__":
    main()
