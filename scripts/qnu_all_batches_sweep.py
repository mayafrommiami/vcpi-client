#!/usr/bin/env python3
"""
Batch-corrected multi-job Ridge evaluation.

Tests whether per-gene mean-shift correction of tvc-bhr-009 and tvc-kdl-010
into tvc-qnu-012's expression space, followed by all-job training, beats
the qnu-only MACCS+Morgan baseline (wMSE 0.47753).

Three conditions compared on the same 3-plate holdout:
  qnu_only      — train on tvc-qnu-012 only            (should reproduce 0.47753)
  no_correction — train on all 3 jobs, raw             (expect ~0.4815)
  mean_shift    — train on all 3 jobs, bhr/kdl shifted to qnu space

Per-gene shift is estimated from the 4 shared positive controls
(Staurosporin, Trichostatin-A, Brefeldin-A, Rigosertib) present in all
3 jobs. This is the standard ComBat-style mean-shift step without the
scale correction (which is unreliable with only 4 anchor compounds).

Run:
    /opt/homebrew/bin/python3.13 scripts/qnu_all_batches_sweep.py
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

SEED   = 13
N_COMP = 128
ALPHA  = 1000.0

# Shared positive controls present in all 3 jobs (user_compound_id)
CTRL_IDS = ["Staurosporin", "Trichostatin-A", "Brefeldin-A", "Rigosertib"]


# ── Expression loading ─────────────────────────────────────────────────────────

def load_job_expr(
    metadata: pd.DataFrame,
    job_id: str,
    active_ids: set[str],
    gene_filter: list[str],
) -> pd.DataFrame:
    """log₂(CPM+1) wide matrix for one job, including the 4 shared controls."""
    job_meta = metadata[metadata["job_id"] == job_id]
    all_ids  = sorted(active_ids | set(CTRL_IDS))
    return qnu.expression_wide(job_meta, all_ids, gene_filter)


# ── Batch correction ───────────────────────────────────────────────────────────

def compute_shift(ref_expr: pd.DataFrame, batch_expr: pd.DataFrame) -> pd.Series:
    """Per-gene additive shift to align batch to reference using shared controls.

    Returns a Series indexed by gene_id. Positive shift means the batch is
    lower than the reference and the expression will be increased.
    """
    shared = [c for c in CTRL_IDS if c in ref_expr.columns and c in batch_expr.columns]
    if len(shared) < 2:
        raise RuntimeError(
            f"Need ≥2 shared controls, found {len(shared)}: {shared}. "
            "Check that CTRL_IDS match user_compound_id values in metadata."
        )
    shift = ref_expr[shared].mean(axis=1) - batch_expr[shared].mean(axis=1)
    abs_s = shift.abs()
    print(
        f"    shift ({len(shared)} controls): "
        f"mean={abs_s.mean():.4f}  p95={abs_s.quantile(0.95):.4f}  max={abs_s.max():.4f}",
        flush=True,
    )
    return shift


def build_combined_expr(
    metadata: pd.DataFrame,
    active_ids_by_job: dict[str, set[str]],
    gene_filter: list[str],
    correction: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Build combined expression matrix.

    Returns (genes × compounds DataFrame, sorted compound id list).
    All values in log₂(CPM+1) space; bhr/kdl shifted into qnu-012's
    space when correction="mean_shift". Control columns are dropped.

    correction: "mean_shift" | "no_correction" | "qnu_only"
    """
    if correction == "qnu_only":
        print("  Loading qnu-012 expression...", flush=True)
        expr = load_job_expr(
            metadata, qnu.QNU_JOB_ID, active_ids_by_job[qnu.QNU_JOB_ID], gene_filter
        )
        expr = expr.drop(columns=[c for c in CTRL_IDS if c in expr.columns])
        compound_ids = sorted(expr.columns)
        return expr[compound_ids].astype(np.float32), compound_ids

    print("  Loading qnu-012 expression...", flush=True)
    ref_expr = load_job_expr(
        metadata, qnu.QNU_JOB_ID, active_ids_by_job[qnu.QNU_JOB_ID], gene_filter
    )
    parts = [ref_expr.drop(columns=[c for c in CTRL_IDS if c in ref_expr.columns])]

    for job_id in [qnu.BHR_JOB_ID, qnu.KDL_JOB_ID]:
        print(f"  Loading {job_id} expression...", flush=True)
        batch_expr = load_job_expr(
            metadata, job_id, active_ids_by_job[job_id], gene_filter
        )
        if correction == "mean_shift":
            print(f"  Computing mean-shift for {job_id}...", flush=True)
            shift = compute_shift(ref_expr, batch_expr)
            batch_expr = batch_expr.add(shift, axis=0).clip(lower=0.0)
        batch_expr = batch_expr.drop(columns=[c for c in CTRL_IDS if c in batch_expr.columns])
        parts.append(batch_expr)

    combined = pd.concat(parts, axis=1).astype(np.float32)
    compound_ids = sorted(combined.columns)
    return combined[compound_ids], compound_ids


# ── Feature building ───────────────────────────────────────────────────────────

def build_maccs_morgan_fp_dict(
    train_chem: pd.DataFrame,
    all_ids: list[str],
) -> dict[str, np.ndarray]:
    """MACCS167 + Morgan512 (r=2) + 8 RDKit descriptors = 687-dim."""
    chem = train_chem.copy()
    chem["user_id"] = chem["user_compound_id"].astype(str)
    chem = (
        chem[chem["user_id"].isin(all_ids)]
        .drop_duplicates("user_id")
        .set_index("user_id")
    )
    desc_frame = chem[qnu.DESC_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)

    result: dict[str, np.ndarray] = {}
    for uid, row in chem.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None:
            continue
        # MACCS 167-bit
        bv   = MACCSkeys.GenMACCSKeys(mol)
        maccs = np.zeros(167, dtype=np.float32)
        for bit in bv.GetOnBits():
            if bit < 167:
                maccs[bit] = 1.0
        # Morgan 512-bit
        morgan = qnu.bitvect_to_array(morgan_gen.GetFingerprint(mol), 512)
        # 8 descriptors
        desc = (
            desc_frame.loc[uid].to_numpy(dtype=np.float32)
            if uid in desc_frame.index
            else np.zeros(len(qnu.DESC_COLS), dtype=np.float32)
        )
        result[uid] = np.concatenate([maccs, morgan, desc])
    return result


def _robust_std(
    x_train: np.ndarray, x_val: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mu  = x_train.mean(axis=0, keepdims=True)
    sig = x_train.std(axis=0, keepdims=True)
    sig[sig < 1e-6] = 1.0
    return ((x_train - mu) / sig).astype(np.float32), ((x_val - mu) / sig).astype(np.float32)


# ── Per-plate evaluation ───────────────────────────────────────────────────────

def eval_condition(
    *,
    condition: str,
    plates: list[int],
    qnu_active: pd.DataFrame,
    expr_combined: pd.DataFrame,
    compound_ids: list[str],
    fp_dict: dict[str, np.ndarray],
    train_meta: pd.DataFrame,
    gene_filter: list[str],
) -> list[float]:
    """Evaluate one correction condition over all holdout plates."""
    all_id_index = {uid: i for i, uid in enumerate(compound_ids)}
    y_all = expr_combined[compound_ids].T.to_numpy(dtype=np.float32)
    id_set = set(compound_ids)

    scores = []
    for plate_id in plates:
        val_ids = sorted(
            qnu_active.loc[qnu_active["container_id"] == plate_id, "user_compound_id"]
            .astype(str).unique()
        )
        # Reference: all compounds in expression matrix except this plate's compounds
        val_set = set(val_ids)
        ref_ids = [uid for uid in compound_ids if uid not in val_set]

        ref_cov = [uid for uid in ref_ids if uid in fp_dict]
        val_cov = [uid for uid in val_ids if uid in fp_dict]
        if len(ref_cov) < 20 or not val_cov:
            print(
                f"  [{condition}] plate {plate_id}: insufficient coverage "
                f"(ref_cov={len(ref_cov)}, val_cov={len(val_cov)}), skipping",
                flush=True,
            )
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
        pred_cov = np.clip(pred_pc @ vt[:n_comp] + y_mean, 0.0, None)  # (n_val_cov, n_genes)

        # Truth comes directly from raw counts — unaffected by any correction
        plate_meta_val = train_meta[
            train_meta["user_compound_id"].astype(str).isin(val_ids)
            & (train_meta["container_id"] == plate_id)
        ]
        truth   = qnu.plate_expression_wide(plate_meta_val, gene_filter)
        truth   = truth.reindex(index=gene_filter, columns=val_ids)
        weights = qnu.load_weights(gene_filter, val_ids)

        # Build full pred DataFrame (fallback = ref mean for val compounds not in fp_dict)
        fallback_mean = y_all[
            [all_id_index[uid] for uid in ref_ids if uid in all_id_index]
        ].mean(axis=0)
        pred_df = pd.DataFrame(
            np.broadcast_to(fallback_mean[None, :], (len(val_ids), len(fallback_mean))).copy(),
            index=val_ids,
            columns=gene_filter,
        )
        for i, uid in enumerate(val_cov):
            pred_df.loc[uid] = pred_cov[i]
        pred_df = pred_df.T  # genes × val_ids

        plate_score = float(qnu.score_wmse(truth, pred_df, weights).mean())
        scores.append(plate_score)
        print(
            f"  [{condition}] plate {plate_id}: wMSE={plate_score:.5f}  "
            f"(ref={len(ref_cov)} [{len(compound_ids)-len(val_set)} total-val], "
            f"val={len(val_cov)})",
            flush=True,
        )

    return scores


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", default="eval_all_batches")
    args = parser.parse_args()

    print("Loading inputs...", flush=True)
    train_chem, train_meta, query, _qpath, gene_filter, weight_cols = qnu.load_inputs()
    train_meta["user_compound_id"] = train_meta["user_compound_id"].astype(str)

    active_mask      = qnu.target_active_mask(train_meta)
    active_ids       = set(train_meta.loc[active_mask, "user_compound_id"])
    all_ids_list, all_fps = qnu.build_chemistry_maps(train_chem, active_ids, weight_cols)
    all_ids_set      = set(all_ids_list)

    active_ids_by_job = {
        job_id: set(
            train_meta.loc[
                (train_meta["job_id"] == job_id)
                & active_mask
                & train_meta["user_compound_id"].isin(all_ids_set),
                "user_compound_id",
            ].astype(str).unique()
        )
        for job_id in [qnu.BHR_JOB_ID, qnu.KDL_JOB_ID, qnu.QNU_JOB_ID]
    }
    for job_id, ids in active_ids_by_job.items():
        print(f"  {job_id}: {len(ids)} active compounds", flush=True)

    qnu_active = train_meta[
        (train_meta["job_id"] == qnu.QNU_JOB_ID)
        & active_mask
        & train_meta["user_compound_id"].isin(all_ids_set)
    ].copy()

    # Same 3-plate holdout as the MACCS+Morgan best model
    mock_args = SimpleNamespace(
        plates=None,
        plate_selection="similarity",
        num_plates=3,
        max_plates=None,
        target_components=64,
        pls_components=16,
        min_plate_size=20,
        plate_similarity_top_k=10,
    )
    plates, _ = qnu.select_plates(
        args=mock_args, qnu_active=qnu_active, all_fps=all_fps, query=query
    )
    print(f"Holdout plates: {plates}", flush=True)

    # ── Evaluate all conditions ────────────────────────────────────────────────
    CONDITIONS = ["qnu_only", "no_correction", "mean_shift"]
    all_results: dict[str, list[float]] = {}

    for condition in CONDITIONS:
        print(f"\n{'='*60}", flush=True)
        print(f"CONDITION: {condition}", flush=True)
        print(f"{'='*60}", flush=True)

        print("Building expression matrix...", flush=True)
        expr_combined, compound_ids = build_combined_expr(
            train_meta, active_ids_by_job, gene_filter, condition
        )
        print(f"  Matrix: {expr_combined.shape[1]} compounds × {len(gene_filter)} genes", flush=True)

        print("Building MACCS+Morgan feature dict...", flush=True)
        fp_dict = build_maccs_morgan_fp_dict(train_chem, compound_ids)
        print(f"  Feature coverage: {len(fp_dict)}/{len(compound_ids)} compounds", flush=True)

        scores = eval_condition(
            condition=condition,
            plates=plates,
            qnu_active=qnu_active,
            expr_combined=expr_combined,
            compound_ids=compound_ids,
            fp_dict=fp_dict,
            train_meta=train_meta,
            gene_filter=gene_filter,
        )
        all_results[condition] = scores

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("FINAL RESULTS (mean wMSE across 3 plates, lower=better):", flush=True)
    print("=" * 60, flush=True)
    rows = []
    for condition, scores in all_results.items():
        mean_score = float(np.mean(scores)) if scores else float("nan")
        rows.append({
            "condition": condition,
            "n_plates": len(scores),
            "wmse_mean": round(mean_score, 5),
        })
    df = pd.DataFrame(rows).sort_values("wmse_mean")
    print(df.to_string(index=False), flush=True)

    out_path = ROOT / f"{args.output_prefix}_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}", flush=True)
    print("\nPrevious best (MACCS+Morgan, qnu-only): 0.47753", flush=True)


if __name__ == "__main__":
    main()
