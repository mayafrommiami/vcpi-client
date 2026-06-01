#!/usr/bin/env python3
"""
Per-plate positive-control normalization sweep.

Each of the 57 tvc-qnu-012 plates contains 4 positive controls
(Staurosporin, Trichostatin-A, Brefeldin-A, Rigosertib) measured
in quadruplicate (16 reference wells per plate). Since every active
compound lives on exactly one plate, any plate-level technical offset
enters the training data as a compound-specific artifact.

Per-plate shift (per gene, log2CPM space):

  For each control c on plate p:
    dev[p][c] = plate_ctrl_mean[p][c] − global_ctrl_mean[c]   (genes-dim vector)

  plate_bias[p] = mean over controls { dev[p][c] }
  corrected[compound] = raw[compound] − plate_bias[plate(compound)]

The subtraction removes the plate's average deviation; clipped at 0.

Conditions:
  no_correction  — baseline, should reproduce ~0.47753
  plate_shift    — per-plate correction using 4 positive controls

Also outputs eval_plate_correction_plate_stats.csv with per-plate
shift magnitudes (identifies outlier plates / which plates drift most).

Run:
    python scripts/qnu_plate_correction_sweep.py
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

CTRL_IDS = ["Staurosporin", "Trichostatin-A", "Brefeldin-A", "Rigosertib"]


# ── Per-plate shift estimation ────────────────────────────────────────────────

def compute_plate_shifts(
    train_meta: pd.DataFrame,
    gene_filter: list[str],
) -> tuple[dict[int, pd.Series], pd.DataFrame]:
    """Estimate per-plate technical bias from positive controls.

    For each plate:
      plate_bias[p][gene] = mean over controls { plate_ctrl_mean[p][c][gene]
                                                 − global_ctrl_mean[c][gene] }

    Returns:
      plate_bias   — dict[plate_id → Series[gene_id → bias_value]]
      plate_stats  — DataFrame with per-plate quality metrics
    """
    ctrl_meta = train_meta[
        (train_meta["job_id"] == qnu.QNU_JOB_ID)
        & train_meta["is_control"].astype(bool)
        & train_meta["user_compound_id"].astype(str).isin(CTRL_IDS)
    ].copy()
    ctrl_meta["user_compound_id"] = ctrl_meta["user_compound_id"].astype(str)

    plates = sorted(ctrl_meta["container_id"].unique())
    print(f"  Found {len(plates)} plates with positive controls", flush=True)

    # ── Per-plate, per-control expression ────────────────────────────────────
    # plate_ctrl_expr[plate_id] = genes × CTRL_IDS DataFrame (log2CPM, avg over 4 reps)
    plate_ctrl_expr: dict[int, pd.DataFrame] = {}
    for plate_id in plates:
        pmd = ctrl_meta[ctrl_meta["container_id"] == plate_id]
        present = [c for c in CTRL_IDS if c in pmd["user_compound_id"].values]
        if len(present) < 2:
            print(f"  Plate {plate_id}: only {len(present)} controls present, skipping", flush=True)
            continue
        try:
            expr = qnu.expression_wide(pmd, present, gene_filter)
            plate_ctrl_expr[plate_id] = expr
        except Exception as e:
            print(f"  Plate {plate_id}: expression load failed ({e}), skipping", flush=True)

    print(f"  Loaded control expression for {len(plate_ctrl_expr)} plates", flush=True)

    # ── Global per-control mean (average across plates) ───────────────────────
    global_ctrl_mean: dict[str, pd.Series] = {}
    for ctrl_id in CTRL_IDS:
        per_plate = [
            plate_ctrl_expr[p][ctrl_id]
            for p in plate_ctrl_expr
            if ctrl_id in plate_ctrl_expr[p].columns
        ]
        if per_plate:
            global_ctrl_mean[ctrl_id] = pd.concat(per_plate, axis=1).mean(axis=1)

    print(f"  Global means computed for controls: {list(global_ctrl_mean.keys())}", flush=True)

    # ── Per-plate bias (mean of per-control deviations) ───────────────────────
    plate_bias: dict[int, pd.Series] = {}
    stats_rows = []
    for plate_id, ctrl_df in plate_ctrl_expr.items():
        deviations = []
        for ctrl_id in CTRL_IDS:
            if ctrl_id in ctrl_df.columns and ctrl_id in global_ctrl_mean:
                dev = ctrl_df[ctrl_id] - global_ctrl_mean[ctrl_id]
                deviations.append(dev)
        if not deviations:
            continue
        bias = pd.concat(deviations, axis=1).mean(axis=1)  # Series[gene]
        plate_bias[plate_id] = bias

        abs_bias = bias.abs()
        stats_rows.append({
            "plate_id":          plate_id,
            "n_controls_used":   len(deviations),
            "mean_abs_bias":     round(float(abs_bias.mean()), 5),
            "p95_abs_bias":      round(float(abs_bias.quantile(0.95)), 5),
            "max_abs_bias":      round(float(abs_bias.max()), 5),
            "n_genes_shifted_gt05": int((abs_bias > 0.5).sum()),
        })

    plate_stats = pd.DataFrame(stats_rows).sort_values("mean_abs_bias", ascending=False)
    return plate_bias, plate_stats


# ── Expression matrix with optional plate correction ─────────────────────────

def build_expr(
    train_meta: pd.DataFrame,
    active_ids: set[str],
    gene_filter: list[str],
    plate_bias: dict[int, pd.Series] | None,
    compound_plate_map: dict[str, int],
) -> tuple[pd.DataFrame, list[str]]:
    """Build genes × compounds expression matrix, optionally plate-corrected."""
    qnu_active = train_meta[
        (train_meta["job_id"] == qnu.QNU_JOB_ID)
        & train_meta["user_compound_id"].astype(str).isin(active_ids)
    ]
    all_ids = sorted(qnu_active["user_compound_id"].astype(str).unique())
    expr = qnu.expression_wide(qnu_active, all_ids, gene_filter)

    if plate_bias is None:
        return expr[all_ids].astype(np.float32), all_ids

    # Apply per-plate correction: subtract plate_bias[plate(compound)]
    gene_index = {g: i for i, g in enumerate(gene_filter)}
    gene_arr   = list(gene_filter)
    corrected  = expr.copy()
    corrected_count = 0
    for uid in all_ids:
        plate_id = compound_plate_map.get(uid)
        if plate_id is None or plate_id not in plate_bias:
            continue
        bias = plate_bias[plate_id].reindex(gene_arr).fillna(0.0).to_numpy(dtype=np.float32)
        corrected[uid] = np.clip(corrected[uid].to_numpy(dtype=np.float32) - bias, 0.0, None)
        corrected_count += 1

    print(f"  Applied plate correction to {corrected_count}/{len(all_ids)} compounds", flush=True)
    return corrected[all_ids].astype(np.float32), all_ids


# ── Feature building (MACCS167 + Morgan512 + 8 desc = 687-dim) ───────────────

def build_fp_dict(
    train_chem: pd.DataFrame,
    all_ids: list[str],
) -> dict[str, np.ndarray]:
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
        bv    = MACCSkeys.GenMACCSKeys(mol)
        maccs = np.zeros(167, dtype=np.float32)
        for bit in bv.GetOnBits():
            if bit < 167:
                maccs[bit] = 1.0
        morgan = qnu.bitvect_to_array(morgan_gen.GetFingerprint(mol), 512)
        desc   = (
            desc_frame.loc[uid].to_numpy(dtype=np.float32)
            if uid in desc_frame.index
            else np.zeros(len(qnu.DESC_COLS), dtype=np.float32)
        )
        result[uid] = np.concatenate([maccs, morgan, desc])
    return result


def _robust_std(x_tr: np.ndarray, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu  = x_tr.mean(axis=0, keepdims=True)
    sig = x_tr.std(axis=0, keepdims=True)
    sig[sig < 1e-6] = 1.0
    return ((x_tr - mu) / sig).astype(np.float32), ((x_val - mu) / sig).astype(np.float32)


# ── Per-plate holdout evaluation ──────────────────────────────────────────────

def eval_condition(
    *,
    condition: str,
    plates: list[int],
    qnu_active: pd.DataFrame,
    expr: pd.DataFrame,
    compound_ids: list[str],
    fp_dict: dict[str, np.ndarray],
    train_meta: pd.DataFrame,
    gene_filter: list[str],
) -> list[float]:
    all_id_index = {uid: i for i, uid in enumerate(compound_ids)}
    y_all = expr[compound_ids].T.to_numpy(dtype=np.float32)

    scores = []
    for plate_id in plates:
        val_ids = sorted(
            qnu_active.loc[qnu_active["container_id"] == plate_id, "user_compound_id"]
            .astype(str).unique()
        )
        val_set = set(val_ids)
        ref_ids = [uid for uid in compound_ids if uid not in val_set]
        ref_cov = [uid for uid in ref_ids if uid in fp_dict]
        val_cov = [uid for uid in val_ids if uid in fp_dict]
        if len(ref_cov) < 20 or not val_cov:
            print(f"  [{condition}] plate {plate_id}: insufficient coverage, skipping", flush=True)
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

        # Truth always from all raw wells (matches contest scoring)
        plate_meta_val = train_meta[
            train_meta["user_compound_id"].astype(str).isin(val_ids)
            & (train_meta["container_id"] == plate_id)
        ]
        truth   = qnu.plate_expression_wide(plate_meta_val, gene_filter)
        truth   = truth.reindex(index=gene_filter, columns=val_ids)
        weights = qnu.load_weights(gene_filter, val_ids)

        fallback = y_all[
            [all_id_index[uid] for uid in ref_ids if uid in all_id_index]
        ].mean(axis=0)
        pred_df = pd.DataFrame(
            np.broadcast_to(fallback[None, :], (len(val_ids), len(fallback))).copy(),
            index=val_ids, columns=gene_filter,
        )
        for i, uid in enumerate(val_cov):
            pred_df.loc[uid] = pred_cov[i]
        pred_df = pred_df.T

        score = float(qnu.score_wmse(truth, pred_df, weights).mean())
        scores.append(score)
        print(
            f"  [{condition}] plate {plate_id}: wMSE={score:.5f}  "
            f"(ref={len(ref_cov)}, val={len(val_cov)})",
            flush=True,
        )
    return scores


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", default="eval_plate_correction")
    args = parser.parse_args()

    print("Loading inputs...", flush=True)
    train_chem, train_meta, query, _qpath, gene_filter, weight_cols = qnu.load_inputs()
    train_meta["user_compound_id"] = train_meta["user_compound_id"].astype(str)
    train_meta["is_control"]       = train_meta["is_control"].astype(bool)

    active_mask = qnu.target_active_mask(train_meta)
    active_ids  = set(train_meta.loc[active_mask, "user_compound_id"])
    all_ids_list, all_fps = qnu.build_chemistry_maps(train_chem, active_ids, weight_cols)
    all_ids_set = set(all_ids_list)

    # Map each compound to its plate (compounds live on exactly one plate)
    compound_plate_map: dict[str, int] = (
        train_meta[
            (train_meta["job_id"] == qnu.QNU_JOB_ID)
            & active_mask
        ]
        .groupby("user_compound_id")["container_id"]
        .first()
        .to_dict()
    )
    print(f"Compound→plate map: {len(compound_plate_map)} entries", flush=True)

    # Holdout plates (same 3-plate similarity-based selection as all other sweeps)
    qnu_active = train_meta[
        (train_meta["job_id"] == qnu.QNU_JOB_ID)
        & active_mask
        & train_meta["user_compound_id"].isin(all_ids_set)
    ].copy()

    mock_args = SimpleNamespace(
        plates=None, plate_selection="similarity", num_plates=3,
        max_plates=None, target_components=64, pls_components=16,
        min_plate_size=20, plate_similarity_top_k=10,
    )
    plates, _ = qnu.select_plates(args=mock_args, qnu_active=qnu_active, all_fps=all_fps, query=query)
    print(f"Holdout plates: {plates}", flush=True)

    # ── Compute per-plate shifts ─────────────────────────────────────────────
    print("\nComputing per-plate positive-control shifts...", flush=True)
    plate_bias, plate_stats = compute_plate_shifts(train_meta, gene_filter)
    print(f"\nPer-plate bias statistics ({len(plate_bias)} plates):", flush=True)
    print(plate_stats.head(20).to_string(index=False), flush=True)

    out_stats = ROOT / f"{args.output_prefix}_plate_stats.csv"
    plate_stats.to_csv(out_stats, index=False)
    print(f"\nPer-plate stats saved to {out_stats}", flush=True)

    # Summary of global shift magnitudes
    all_biases = pd.DataFrame({pid: bias for pid, bias in plate_bias.items()})
    mean_abs = all_biases.abs().mean(axis=1)
    print(f"\nGlobal shift summary across all plates:", flush=True)
    print(f"  Mean  |bias| per gene: {mean_abs.mean():.4f} log2CPM", flush=True)
    print(f"  P95   |bias| per gene: {mean_abs.quantile(0.95):.4f} log2CPM", flush=True)
    print(f"  Max   |bias| per gene: {mean_abs.max():.4f} log2CPM", flush=True)

    # ── Evaluate conditions ──────────────────────────────────────────────────
    CONDITIONS = {
        "no_correction": None,        # baseline
        "plate_shift":   plate_bias,  # per-plate correction
    }

    all_results: dict[str, list[float]] = {}
    for condition, bias_map in CONDITIONS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"CONDITION: {condition}", flush=True)
        print(f"{'='*60}", flush=True)

        print("Building expression matrix...", flush=True)
        expr, compound_ids = build_expr(
            train_meta, active_ids, gene_filter, bias_map, compound_plate_map
        )
        print(f"  Matrix: {len(compound_ids)} compounds × {len(gene_filter)} genes", flush=True)

        print("Building feature dict...", flush=True)
        fp_dict = build_fp_dict(train_chem, compound_ids)
        print(f"  Feature coverage: {len(fp_dict)}/{len(compound_ids)}", flush=True)

        scores = eval_condition(
            condition=condition,
            plates=plates,
            qnu_active=qnu_active,
            expr=expr,
            compound_ids=compound_ids,
            fp_dict=fp_dict,
            train_meta=train_meta,   # truth always from raw data
            gene_filter=gene_filter,
        )
        all_results[condition] = scores

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("FINAL RESULTS (mean wMSE across holdout plates, lower=better):", flush=True)
    print("=" * 60, flush=True)
    rows = []
    for condition, scores in all_results.items():
        mean_score = float(np.mean(scores)) if scores else float("nan")
        rows.append({
            "condition": condition,
            "n_plates":  len(scores),
            "wmse_mean": round(mean_score, 5),
        })
    df = pd.DataFrame(rows).sort_values("wmse_mean")
    print(df.to_string(index=False), flush=True)
    out_summary = ROOT / f"{args.output_prefix}_summary.csv"
    df.to_csv(out_summary, index=False)
    print(f"\nSaved to {out_summary}", flush=True)
    print("\nPrevious best (no correction): 0.47753", flush=True)


if __name__ == "__main__":
    main()
