#!/usr/bin/env python3
"""
Query-aware kNN Ridge + alpha grid search on the 3-plate qnu holdout.

Experiment A — Alpha grid search
  Tests α ∈ {1, 10, 100, 500, 1000, 3000, 5000, 10000} with MACCS+Morgan512+desc.
  Baseline α=1000 should reproduce ~0.47752.

Experiment B — kNN-weighted Ridge (α=1000)
  For each validation compound, weight training compounds by Tanimoto similarity
  (MACCS+Morgan binary portion) before fitting Ridge — localized regression.
  Top-K variants: K = 30, 50, 100, 200, all.

Run:
    /opt/homebrew/bin/python3.13 scripts/qnu_knn_alpha_sweep.py
"""
from __future__ import annotations

import argparse
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

SEED       = 13
N_COMP     = 128
MACCS_DIM  = 167
MORGAN_DIM = 512
BINARY_DIM = MACCS_DIM + MORGAN_DIM   # 679 bits — used for Tanimoto

ALPHA_GRID = [1, 10, 100, 500, 1000, 3000, 5000, 10000]
KNN_CONFIGS = [
    {"K": 30,   "label": "knn_k30"},
    {"K": 50,   "label": "knn_k50"},
    {"K": 100,  "label": "knn_k100"},
    {"K": 200,  "label": "knn_k200"},
    {"K": None, "label": "knn_all"},   # all training compounds, weighted by similarity
]


# ── Shared utilities ───────────────────────────────────────────────────────────

def build_fp_dict(train_chem: pd.DataFrame, all_ids: list[str]) -> dict[str, np.ndarray]:
    """MACCS167 + Morgan512 (r=2) + 8 RDKit descriptors = 687-dim."""
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
        desc = (
            desc_frame.loc[uid].to_numpy(dtype=np.float32)
            if uid in desc_frame.index
            else np.zeros(len(qnu.DESC_COLS), dtype=np.float32)
        )
        result[uid] = np.concatenate([maccs, morgan, desc])
    return result


def robust_std(x_tr: np.ndarray, x_val: np.ndarray):
    mu  = x_tr.mean(axis=0, keepdims=True)
    sig = x_tr.std(axis=0, keepdims=True)
    sig[sig < 1e-6] = 1.0
    return ((x_tr - mu) / sig).astype(np.float32), ((x_val - mu) / sig).astype(np.float32)


def tanimoto_row(query_bin: np.ndarray, ref_bin: np.ndarray) -> np.ndarray:
    """Tanimoto similarity: one query (d,) vs ref matrix (n, d). Both binary {0,1}."""
    dot   = ref_bin @ query_bin                        # (n,)
    q_sq  = float(query_bin @ query_bin)
    r_sq  = (ref_bin * ref_bin).sum(axis=1)            # (n,)
    denom = q_sq + r_sq - dot
    denom[denom < 1e-8] = 1e-8
    return (dot / denom).astype(np.float32)


def build_pred_df(
    val_ids: list[str],
    val_cov: list[str],
    pred_cov: np.ndarray,
    ref_ids: list[str],
    all_id_index: dict[str, int],
    y_all: np.ndarray,
    gene_filter: list[str],
) -> pd.DataFrame:
    """Assemble genes×val_ids DataFrame; fills non-covered val_ids with ref mean."""
    fallback = y_all[[all_id_index[uid] for uid in ref_ids if uid in all_id_index]].mean(axis=0)
    pred_df  = pd.DataFrame(
        np.broadcast_to(fallback[None, :], (len(val_ids), len(fallback))).copy(),
        index=val_ids, columns=gene_filter,
    )
    for i, uid in enumerate(val_cov):
        pred_df.loc[uid] = pred_cov[i]
    return pred_df.T   # genes × val_ids


# ── Experiment A: standard Ridge, alpha sweep ──────────────────────────────────

def eval_alpha(
    *,
    plates: list[int],
    qnu_active: pd.DataFrame,
    all_ids: list[str],
    all_id_index: dict[str, int],
    y_all: np.ndarray,
    fp_dict: dict[str, np.ndarray],
    train_meta: pd.DataFrame,
    gene_filter: list[str],
    alpha: float,
) -> list[float]:
    label  = f"α={alpha}"
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
            continue

        x_ref = np.vstack([fp_dict[uid] for uid in ref_cov]).astype(np.float32)
        x_val = np.vstack([fp_dict[uid] for uid in val_cov]).astype(np.float32)
        x_ref_s, x_val_s = robust_std(x_ref, x_val)

        y_ref  = y_all[[all_id_index[uid] for uid in ref_cov]]
        y_mean = y_ref.mean(axis=0, keepdims=True)
        y_c    = y_ref - y_mean
        n_comp = min(N_COMP, y_c.shape[0] - 1, y_c.shape[1] - 1)
        u, s, vt = randomized_svd(y_c, n_components=n_comp, n_iter=5, random_state=SEED)
        pc_ref   = (u * s[None, :]).astype(np.float32)

        ridge    = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(x_ref_s, pc_ref)
        pred_pc  = ridge.predict(x_val_s).astype(np.float32)
        pred_cov = np.clip(pred_pc @ vt[:n_comp] + y_mean, 0.0, None)

        plate_meta_val = train_meta[
            train_meta["user_compound_id"].astype(str).isin(val_ids)
            & (train_meta["container_id"] == plate_id)
        ]
        truth   = qnu.plate_expression_wide(plate_meta_val, gene_filter)
        truth   = truth.reindex(index=gene_filter, columns=val_ids)
        weights = qnu.load_weights(gene_filter, val_ids)
        pred_df = build_pred_df(val_ids, val_cov, pred_cov, ref_ids, all_id_index, y_all, gene_filter)

        score = float(qnu.score_wmse(truth, pred_df, weights).mean())
        scores.append(score)
        print(f"  [{label}] plate {plate_id}: wMSE={score:.5f}", flush=True)

    return scores


# ── Experiment B: per-compound kNN-weighted Ridge ─────────────────────────────

def eval_knn(
    *,
    plates: list[int],
    qnu_active: pd.DataFrame,
    all_ids: list[str],
    all_id_index: dict[str, int],
    y_all: np.ndarray,
    fp_dict: dict[str, np.ndarray],
    train_meta: pd.DataFrame,
    gene_filter: list[str],
    alpha: float,
    K: int | None,
    label: str,
) -> list[float]:
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
            continue

        x_ref = np.vstack([fp_dict[uid] for uid in ref_cov]).astype(np.float32)
        x_val = np.vstack([fp_dict[uid] for uid in val_cov]).astype(np.float32)
        x_ref_s, x_val_s = robust_std(x_ref, x_val)

        # Global PCA basis (shared across all val compounds for this plate)
        y_ref  = y_all[[all_id_index[uid] for uid in ref_cov]]
        y_mean = y_ref.mean(axis=0, keepdims=True)
        y_c    = y_ref - y_mean
        n_comp = min(N_COMP, y_c.shape[0] - 1, y_c.shape[1] - 1)
        u, s, vt = randomized_svd(y_c, n_components=n_comp, n_iter=5, random_state=SEED)
        pc_ref   = (u * s[None, :]).astype(np.float32)

        # Binary portion for Tanimoto
        x_ref_bin = x_ref[:, :BINARY_DIM]

        pred_cov = np.zeros((len(val_cov), len(gene_filter)), dtype=np.float32)

        for i, uid in enumerate(val_cov):
            query_bin = fp_dict[uid][:BINARY_DIM]
            sim = tanimoto_row(query_bin, x_ref_bin)   # (n_ref_cov,)

            if K is not None and K < len(ref_cov):
                # Zero out all but top-K by similarity
                top_k_idx = np.argpartition(sim, -K)[-K:]
                w = np.zeros(len(ref_cov), dtype=np.float32)
                w[top_k_idx] = sim[top_k_idx]
            else:
                w = sim.copy()

            w = np.clip(w, 0.0, None)
            w_sum = w.sum()
            if w_sum < 1e-8:
                # Degenerate: fall back to uniform weights
                w = np.ones(len(ref_cov), dtype=np.float32)
            else:
                # Normalize so mean weight = 1 (preserves effective sample size)
                w = w / w_sum * len(ref_cov)

            ridge_local = Ridge(alpha=alpha, fit_intercept=True)
            ridge_local.fit(x_ref_s, pc_ref, sample_weight=w)
            pred_pc_i    = ridge_local.predict(x_val_s[[i]]).astype(np.float32)
            pred_cov[i]  = np.clip(pred_pc_i @ vt[:n_comp] + y_mean, 0.0, None)

        plate_meta_val = train_meta[
            train_meta["user_compound_id"].astype(str).isin(val_ids)
            & (train_meta["container_id"] == plate_id)
        ]
        truth   = qnu.plate_expression_wide(plate_meta_val, gene_filter)
        truth   = truth.reindex(index=gene_filter, columns=val_ids)
        weights = qnu.load_weights(gene_filter, val_ids)
        pred_df = build_pred_df(val_ids, val_cov, pred_cov, ref_ids, all_id_index, y_all, gene_filter)

        score = float(qnu.score_wmse(truth, pred_df, weights).mean())
        scores.append(score)
        print(f"  [{label}] plate {plate_id}: wMSE={score:.5f}  ({len(val_cov)} val fits)", flush=True)

    return scores


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", default="eval_knn_alpha")
    args = parser.parse_args()

    print("Loading inputs...", flush=True)
    train_chem, train_meta, query, _qpath, gene_filter, weight_cols = qnu.load_inputs()
    train_meta["user_compound_id"] = train_meta["user_compound_id"].astype(str)

    active_mask  = qnu.target_active_mask(train_meta)
    active_ids   = set(train_meta.loc[active_mask, "user_compound_id"])
    all_ids_list, all_fps = qnu.build_chemistry_maps(train_chem, active_ids, weight_cols)
    all_ids_set  = set(all_ids_list)

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

    print("Computing expression matrix...", flush=True)
    expr = qnu.expression_wide(
        train_meta[
            (train_meta["job_id"] == qnu.QNU_JOB_ID) & active_mask
            & train_meta["user_compound_id"].isin(all_ids_set)
        ],
        all_ids_list, gene_filter,
    )
    all_ids_list = [uid for uid in all_ids_list if uid in expr.columns]
    y_all        = expr[all_ids_list].T.to_numpy(dtype=np.float32)
    all_id_index = {uid: i for i, uid in enumerate(all_ids_list)}

    print("Building features...", flush=True)
    fp_dict = build_fp_dict(train_chem, all_ids_list)
    print(f"  Coverage: {len(fp_dict)}/{len(all_ids_list)}", flush=True)

    rows = []

    # ── A: Alpha grid ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("EXPERIMENT A — Alpha grid search", flush=True)
    print(f"{'='*60}", flush=True)
    for alpha in ALPHA_GRID:
        print(f"\n  α = {alpha}", flush=True)
        scores = eval_alpha(
            plates=plates, qnu_active=qnu_active,
            all_ids=all_ids_list, all_id_index=all_id_index,
            y_all=y_all, fp_dict=fp_dict, train_meta=train_meta,
            gene_filter=gene_filter, alpha=alpha,
        )
        mean = float(np.mean(scores)) if scores else float("nan")
        rows.append({
            "experiment": "alpha_sweep",
            "config": f"alpha_{alpha}",
            "alpha": alpha, "K": None,
            "n_plates": len(scores), "wmse_mean": round(mean, 5),
        })

    # ── B: kNN-weighted Ridge ──────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("EXPERIMENT B — kNN-weighted Ridge (α=1000)", flush=True)
    print(f"{'='*60}", flush=True)
    for cfg in KNN_CONFIGS:
        print(f"\n  K = {cfg['K']}", flush=True)
        scores = eval_knn(
            plates=plates, qnu_active=qnu_active,
            all_ids=all_ids_list, all_id_index=all_id_index,
            y_all=y_all, fp_dict=fp_dict, train_meta=train_meta,
            gene_filter=gene_filter, alpha=1000,
            K=cfg["K"], label=cfg["label"],
        )
        mean = float(np.mean(scores)) if scores else float("nan")
        rows.append({
            "experiment": "knn_ridge",
            "config": cfg["label"],
            "alpha": 1000, "K": cfg["K"],
            "n_plates": len(scores), "wmse_mean": round(mean, 5),
        })

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("FINAL RESULTS (mean wMSE, lower=better):", flush=True)
    print("=" * 60, flush=True)
    df = pd.DataFrame(rows).sort_values("wmse_mean")
    print(df[["experiment", "config", "wmse_mean"]].to_string(index=False), flush=True)

    out_path = ROOT / f"{args.output_prefix}_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}", flush=True)
    print("\nBaseline (α=1000, standard Ridge): 0.47752", flush=True)


if __name__ == "__main__":
    main()
