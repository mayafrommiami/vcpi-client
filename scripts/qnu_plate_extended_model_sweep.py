"""Extended model sweep on the current tvc-qnu-012 3-plate validation.

This keeps the split logic from ``qnu_plate_holdout_eval.py`` fixed and
adds the simple RDKit/LPM feature models that were useful in earlier
experiments.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.utils.extmath import randomized_svd

import qnu_plate_holdout_eval as qnu


CHEM_LPM = Path("/lustre/groups/ml01/workspace/artur.szalata/code/chem-perturbridge_lpm")
MORGAN_META = (
    CHEM_LPM
    / ".plib_cache/morgan_perturbation_embeddings/pubchem_morgan_radius2_nbits128/compound_metadata.parquet"
)
MORGAN_EMB = (
    CHEM_LPM
    / ".plib_cache/morgan_perturbation_embeddings/pubchem_morgan_radius2_nbits128/compound_embeddings.npy"
)
LPM_ALL_META = (
    CHEM_LPM / "results/lpm_paper10_best_overall_source_embeddings/molecule/molecule_metadata.parquet"
)
LPM_ALL_EMB = (
    CHEM_LPM / "results/lpm_paper10_best_overall_source_embeddings/molecule/molecule_embeddings.npy"
)
FT_ROOT = CHEM_LPM / "results/lpm_paper10_ft_morgan_learned_fixmol_best_embeddings"

SEED = 13
TARGET_COMPONENTS = [64, 128]
RIDGE_ALPHAS = [100.0, 1000.0, 10000.0]
DIRECT_RIDGE_ALPHAS = [1000.0, 10000.0]
DIRECT_RIDGE_SHRINKS = [1.0, 1.25]
KNN_K = [5, 20, 50]


@dataclass
class FeatureSet:
    name: str
    by_user_id: dict[str, np.ndarray]


def robust_standardize_train_val(x_ref: np.ndarray, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x_ref, axis=0, keepdims=True)
    mean = np.where(np.isnan(mean), 0.0, mean)
    x_ref = np.where(np.isnan(x_ref), mean, x_ref)
    x_val = np.where(np.isnan(x_val), mean, x_val)
    scale = x_ref.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return ((x_ref - mean) / scale).astype(np.float32), ((x_val - mean) / scale).astype(np.float32)


def feature_matrix(feature: FeatureSet, ids: list[str]) -> tuple[list[str], np.ndarray]:
    covered = [uid for uid in ids if uid in feature.by_user_id]
    if not covered:
        return [], np.empty((0, 0), dtype=np.float32)
    return covered, np.vstack([feature.by_user_id[uid] for uid in covered]).astype(np.float32)


def load_external_canon_maps() -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    morgan_meta = pd.read_parquet(MORGAN_META)
    morgan_arr = np.load(MORGAN_EMB, mmap_mode="r")
    lpm_all_meta = pd.read_parquet(LPM_ALL_META)
    lpm_all_arr = np.load(LPM_ALL_EMB, mmap_mode="r")

    if not morgan_meta["symbol"].astype(str).equals(lpm_all_meta["symbol"].astype(str)):
        raise RuntimeError("Morgan and all-source LPM metadata row order does not match")

    symbol_to_canon: dict[str, str] = {}
    morgan_by_canon: dict[str, np.ndarray] = {}
    all_by_canon: dict[str, np.ndarray] = {}
    for i, row in morgan_meta.iterrows():
        canon = qnu.canonical_smiles(row.get("smiles"))
        if canon is None:
            continue
        symbol_to_canon[str(row["symbol"])] = canon
        morgan_by_canon.setdefault(canon, np.asarray(morgan_arr[i], dtype=np.float32))
        all_by_canon.setdefault(canon, np.asarray(lpm_all_arr[i], dtype=np.float32))

    ft_maps: dict[str, dict[str, np.ndarray]] = {}
    for name in ["vcpi_0001", "vcpi_0002"]:
        meta = pd.read_parquet(FT_ROOT / name / "molecule_metadata.parquet")
        arr = np.load(FT_ROOT / name / "molecule_embeddings.npy", mmap_mode="r")
        by_canon: dict[str, np.ndarray] = {}
        for i, row in meta.iterrows():
            canon = symbol_to_canon.get(str(row["symbol"]))
            if canon is not None:
                by_canon.setdefault(canon, np.asarray(arr[i], dtype=np.float32))
        ft_maps[name] = by_canon

    return morgan_by_canon, all_by_canon, ft_maps["vcpi_0001"], ft_maps["vcpi_0002"]


def build_feature_sets(train_chem: pd.DataFrame, all_ids: list[str]) -> list[FeatureSet]:
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    chem = train_chem.copy()
    chem["user_id"] = chem["user_compound_id"].astype(str)
    chem = chem[chem["user_id"].isin(all_ids)].drop_duplicates("user_id").set_index("user_id")
    chem["canon_smiles"] = chem["smiles"].map(qnu.canonical_smiles)

    desc_frame = chem[qnu.DESC_COLS].apply(pd.to_numeric, errors="coerce")
    desc_by_id = {uid: desc_frame.loc[uid].to_numpy(dtype=np.float32) for uid in desc_frame.index}

    rdkit_bits: dict[str, np.ndarray] = {}
    for uid, row in chem.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None:
            continue
        rdkit_bits[uid] = qnu.bitvect_to_array(fp_gen.GetFingerprint(mol), 512)

    morgan_canon, all_canon, ft1_canon, ft2_canon = load_external_canon_maps()

    def from_canon(source: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            uid: source[row.canon_smiles]
            for uid, row in chem.iterrows()
            if row.canon_smiles in source
        }

    external_morgan = from_canon(morgan_canon)
    lpm_all = from_canon(all_canon)
    ft1 = from_canon(ft1_canon)
    ft2 = from_canon(ft2_canon)

    ft12: dict[str, np.ndarray] = {}
    ft12_else_all: dict[str, np.ndarray] = {}
    for uid, row in chem.iterrows():
        canon = row.canon_smiles
        if canon in ft1_canon:
            ft12[uid] = ft1_canon[canon]
            ft12_else_all[uid] = ft1_canon[canon]
        elif canon in ft2_canon:
            ft12[uid] = ft2_canon[canon]
            ft12_else_all[uid] = ft2_canon[canon]
        elif canon in all_canon:
            ft12_else_all[uid] = all_canon[canon]

    def concat(name: str, *parts: dict[str, np.ndarray]) -> FeatureSet:
        ids = set(parts[0])
        for part in parts[1:]:
            ids &= set(part)
        return FeatureSet(name, {uid: np.concatenate([part[uid] for part in parts]) for uid in sorted(ids)})

    return [
        FeatureSet("descriptors", desc_by_id),
        FeatureSet("rdkit_morgan512", rdkit_bits),
        concat("rdkit_morgan512+descriptors", rdkit_bits, desc_by_id),
        FeatureSet("external_morgan128", external_morgan),
        concat("external_morgan128+descriptors", external_morgan, desc_by_id),
        FeatureSet("lpm_all_source", lpm_all),
        concat("lpm_all_source+descriptors", lpm_all, desc_by_id),
        FeatureSet("lpm_ft_vcpi_0001", ft1),
        FeatureSet("lpm_ft_vcpi_0002", ft2),
        FeatureSet("lpm_ft_vcpi_0001_0002", ft12),
        FeatureSet("lpm_ft12_else_all_source", ft12_else_all),
        concat("lpm_ft12_else_all_source+descriptors", ft12_else_all, desc_by_id),
    ]


def frame_from_prediction(arr: np.ndarray, gene_index: pd.Index, val_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(arr.T, index=gene_index, columns=val_ids)


def score_prediction(
    scores: dict[str, pd.Series],
    meta: dict[str, dict[str, object]],
    *,
    name: str,
    pred_arr: np.ndarray,
    truth: pd.DataFrame,
    val_ids: list[str],
    weights: pd.DataFrame,
    row: dict[str, object],
) -> None:
    pred = frame_from_prediction(pred_arr, truth.index, val_ids)
    scores[name] = qnu.score_wmse(truth, pred, weights)
    meta[name] = row


def cosine_knn_prediction(
    *,
    feature: FeatureSet,
    ref_ids: list[str],
    val_ids: list[str],
    y_ref_frame: pd.DataFrame,
    fallback_mean: np.ndarray,
    k: int,
) -> tuple[np.ndarray, int, int]:
    ref_cov, x_ref = feature_matrix(feature, ref_ids)
    val_cov, x_val = feature_matrix(feature, val_ids)
    pred_arr = np.broadcast_to(fallback_mean[None, :], (len(val_ids), len(fallback_mean))).copy()
    if len(ref_cov) < 20 or len(val_cov) == 0:
        return pred_arr, len(ref_cov), len(val_cov)

    x_ref, x_val = robust_standardize_train_val(x_ref, x_val)
    x_ref = x_ref / np.maximum(np.linalg.norm(x_ref, axis=1, keepdims=True), 1e-6)
    x_val = x_val / np.maximum(np.linalg.norm(x_val, axis=1, keepdims=True), 1e-6)
    ref_y = y_ref_frame[ref_cov].T.to_numpy(dtype=np.float32)
    val_pos = {uid: i for i, uid in enumerate(val_ids)}
    sims = x_val @ x_ref.T
    for i, uid in enumerate(val_cov):
        row = sims[i]
        k_eff = min(k, len(row))
        take = np.argpartition(-row, k_eff - 1)[:k_eff]
        w = np.maximum(row[take], 0.0)
        if float(w.sum()) <= 1e-8:
            w = np.full(k_eff, 1.0 / k_eff, dtype=np.float32)
        else:
            w = w / w.sum()
        pred_arr[val_pos[uid]] = w @ ref_y[take]
    return pred_arr.astype(np.float32), len(ref_cov), len(val_cov)


def fit_direct_ridge(
    *,
    x_ref: np.ndarray,
    x_val: np.ndarray,
    y_ref: np.ndarray,
    alpha: float,
    shrink: float,
) -> np.ndarray:
    x_ref, x_val = robust_standardize_train_val(x_ref, x_val)
    y_mean = y_ref.mean(axis=0, keepdims=True).astype(np.float32)
    y_centered = y_ref - y_mean
    xtx = x_ref.T @ x_ref
    xty = x_ref.T @ y_centered
    xtx.flat[:: xtx.shape[0] + 1] += alpha
    coef = np.linalg.solve(xtx.astype(np.float64), xty.astype(np.float64)).astype(np.float32)
    return np.clip(y_mean + shrink * (x_val @ coef), 0.0, None).astype(np.float32)


def target_basis(
    y_ref: np.ndarray,
    *,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_mean = y_ref.mean(axis=0, keepdims=True).astype(np.float32)
    y_centered = y_ref - y_mean
    u, s, vt = randomized_svd(
        y_centered,
        n_components=n_components,
        n_iter=5,
        random_state=SEED,
    )
    return y_mean, vt.astype(np.float32), (u * s[None, :]).astype(np.float32)


def reconstruct(pred_scores: np.ndarray, y_mean: np.ndarray, vt: np.ndarray, n_components: int) -> np.ndarray:
    return np.clip(pred_scores[:, :n_components] @ vt[:n_components] + y_mean, 0.0, None).astype(np.float32)


def add_feature_models_for_plate(
    *,
    feature: FeatureSet,
    ref_ids: list[str],
    val_ids: list[str],
    y_ref_frame: pd.DataFrame,
    y_all: np.ndarray,
    all_id_index: dict[str, int],
    fallback_mean: np.ndarray,
    truth: pd.DataFrame,
    weights: pd.DataFrame,
    target_cache: dict[tuple[str, ...], tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    scores: dict[str, pd.Series],
    meta: dict[str, dict[str, object]],
) -> None:
    for k in KNN_K:
        pred_arr, ref_cov_n, val_cov_n = cosine_knn_prediction(
            feature=feature,
            ref_ids=ref_ids,
            val_ids=val_ids,
            y_ref_frame=y_ref_frame,
            fallback_mean=fallback_mean,
            k=k,
        )
        if val_cov_n:
            score_prediction(
                scores,
                meta,
                name=f"{feature.name}__cosine_knn_k{k}",
                pred_arr=pred_arr,
                truth=truth,
                val_ids=val_ids,
                weights=weights,
                row={
                    "model_family": "cosine_knn",
                    "features": feature.name,
                    "k": k,
                    "ref_coverage": ref_cov_n,
                    "val_coverage": val_cov_n,
                },
            )

    ref_cov, x_ref = feature_matrix(feature, ref_ids)
    val_cov, x_val = feature_matrix(feature, val_ids)
    if len(ref_cov) < 20 or len(val_cov) == 0:
        return

    ref_pos = np.array([all_id_index[uid] for uid in ref_cov], dtype=np.int64)
    y_ref_cov = y_all[ref_pos]
    val_pos = {uid: i for i, uid in enumerate(val_ids)}

    def fill_prediction(pred_cov: np.ndarray) -> np.ndarray:
        pred_arr = np.broadcast_to(fallback_mean[None, :], (len(val_ids), len(fallback_mean))).copy()
        for i, uid in enumerate(val_cov):
            pred_arr[val_pos[uid]] = pred_cov[i]
        return pred_arr.astype(np.float32)

    for alpha in DIRECT_RIDGE_ALPHAS:
        for shrink in DIRECT_RIDGE_SHRINKS:
            pred_cov = fit_direct_ridge(
                x_ref=x_ref,
                x_val=x_val,
                y_ref=y_ref_cov,
                alpha=alpha,
                shrink=shrink,
            )
            score_prediction(
                scores,
                meta,
                name=f"{feature.name}__direct_ridge_alpha{alpha:g}_shrink{shrink:g}",
                pred_arr=fill_prediction(pred_cov),
                truth=truth,
                val_ids=val_ids,
                weights=weights,
                row={
                    "model_family": "direct_ridge",
                    "features": feature.name,
                    "alpha": alpha,
                    "shrink": shrink,
                    "ref_coverage": len(ref_cov),
                    "val_coverage": len(val_cov),
                },
            )

    max_components = min(max(TARGET_COMPONENTS), len(ref_cov) - 1, y_ref_cov.shape[1] - 1)
    if max_components < min(TARGET_COMPONENTS):
        return
    cache_key = tuple(ref_cov)
    cached = target_cache.get(cache_key)
    if cached is None or cached[0] < max_components:
        y_mean, vt, pc_ref = target_basis(y_ref_cov, n_components=max_components)
        target_cache[cache_key] = (max_components, y_mean, vt, pc_ref)
    else:
        _, y_mean, vt, pc_ref = cached
    x_ref_std, x_val_std = robust_standardize_train_val(x_ref, x_val)
    for n_components in TARGET_COMPONENTS:
        if n_components > max_components:
            continue
        y_pc = pc_ref[:, :n_components]
        for alpha in RIDGE_ALPHAS:
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(x_ref_std, y_pc)
            pred_pc = model.predict(x_val_std).astype(np.float32)
            pred_cov = reconstruct(pred_pc, y_mean, vt, n_components)
            score_prediction(
                scores,
                meta,
                name=f"{feature.name}__target_pca{n_components}_ridge_alpha{alpha:g}",
                pred_arr=fill_prediction(pred_cov),
                truth=truth,
                val_ids=val_ids,
                weights=weights,
                row={
                    "model_family": "target_pca_ridge",
                    "features": feature.name,
                    "n_target_components": n_components,
                    "alpha": alpha,
                    "ref_coverage": len(ref_cov),
                    "val_coverage": len(val_cov),
                },
            )

        if min(x_ref.shape[1], n_components) >= 16:
            pls = PLSRegression(n_components=16, scale=True, max_iter=300, tol=1e-5)
            pls.fit(x_ref, y_pc)
            pred_pc = pls.predict(x_val).astype(np.float32)
            pred_cov = reconstruct(pred_pc, y_mean, vt, n_components)
            score_prediction(
                scores,
                meta,
                name=f"{feature.name}__target_pca{n_components}_pls16",
                pred_arr=fill_prediction(pred_cov),
                truth=truth,
                val_ids=val_ids,
                weights=weights,
                row={
                    "model_family": "target_pca_pls",
                    "features": feature.name,
                    "n_target_components": n_components,
                    "pls_components": 16,
                    "ref_coverage": len(ref_cov),
                    "val_coverage": len(val_cov),
                },
            )


def summarize(per_compound: pd.DataFrame, meta_rows: dict[str, dict[str, object]]) -> pd.DataFrame:
    model_cols = [c for c in per_compound.columns if c not in {"plate_id", "compound"}]
    rows = []
    for model in model_cols:
        values = per_compound[model].to_numpy()
        row = {
            "model": model,
            "n_plates": int(per_compound["plate_id"].nunique()),
            "n_compounds": int(per_compound["compound"].nunique()),
            "wmse_mean": float(np.mean(values)),
        }
        row.update(meta_rows.get(model, {}))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["wmse_mean", "model"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", default="eval_qnu_plate_extended_models")
    parser.add_argument("--num-plates", type=int, default=qnu.DEFAULT_NUM_PLATES)
    parser.add_argument("--plates", nargs="*", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_chem, train_meta, query, _qpath, gene_filter, weight_cols = qnu.load_inputs()
    train_meta = train_meta.copy()
    train_meta["user_compound_id"] = train_meta["user_compound_id"].astype(str)

    active_mask = qnu.target_active_mask(train_meta)
    active_ids = set(train_meta.loc[active_mask, "user_compound_id"])
    all_ids, all_fps = qnu.build_chemistry_maps(train_chem, active_ids, weight_cols)
    all_id_index = {uid: i for i, uid in enumerate(all_ids)}

    qnu_active = train_meta[
        (train_meta["job_id"] == qnu.QNU_JOB_ID)
        & active_mask
        & train_meta["user_compound_id"].isin(all_ids)
    ].copy()
    qnu_ids = set(qnu_active["user_compound_id"].astype(str).unique())

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
    plate_stats.to_csv(qnu.ROOT / f"{args.output_prefix}_plate_similarity.csv", index=False)
    print(
        "Selected plates: "
        + ", ".join(
            f"{row.plate_id}(medianT={row.median_max_test_tanimoto:.3f}, meanT={row.mean_max_test_tanimoto:.3f})"
            for row in plate_stats[plate_stats["selected"]].itertuples(index=False)
        ),
        flush=True,
    )

    print("Aggregating all active-compound expression...", flush=True)
    expr_all = qnu.expression_wide(train_meta, all_ids, gene_filter)
    y_all = expr_all[all_ids].T.to_numpy(dtype=np.float32)

    print("Building RDKit regression features...", flush=True)
    rdkit_regression_all = qnu.regression_features(train_chem, all_ids)

    print("Building external feature sets...", flush=True)
    features = build_feature_sets(train_chem, all_ids)
    coverage_rows = []
    selected_val_ids = sorted(
        set(qnu_active.loc[qnu_active["container_id"].isin(plates), "user_compound_id"].astype(str))
    )
    for feature in features:
        dim = len(next(iter(feature.by_user_id.values()))) if feature.by_user_id else 0
        coverage_rows.append(
            {
                "features": feature.name,
                "dimension": dim,
                "reference_universe_covered": sum(uid in feature.by_user_id for uid in all_ids),
                "reference_universe_total": len(all_ids),
                "selected_validation_covered": sum(uid in feature.by_user_id for uid in selected_val_ids),
                "selected_validation_total": len(selected_val_ids),
            }
        )
    pd.DataFrame(coverage_rows).to_csv(qnu.ROOT / f"{args.output_prefix}_coverage.csv", index=False)
    print(pd.DataFrame(coverage_rows).to_string(index=False), flush=True)

    plate_rows: list[dict[str, object]] = []
    per_compound_pieces: list[pd.DataFrame] = []
    model_meta: dict[str, dict[str, object]] = {}
    split_rows = []

    for plate_idx, plate_id in enumerate(plates, start=1):
        print(f"[{plate_idx}/{len(plates)}] plate {plate_id}", flush=True)
        plate_meta = qnu_active[qnu_active["container_id"] == plate_id].copy()
        val_ids = sorted(set(plate_meta["user_compound_id"].astype(str)) & set(all_ids))
        for uid in val_ids:
            split_rows.append({"plate_id": plate_id, "compound": uid, "split_role": "validation"})
        if not val_ids:
            continue

        val_positions = {all_id_index[uid] for uid in val_ids}
        ref_mask = np.array([i not in val_positions for i in range(len(all_ids))], dtype=bool)
        qnu_ref_mask = np.array(
            [keep and uid in qnu_ids for uid, keep in zip(all_ids, ref_mask, strict=True)],
            dtype=bool,
        )
        ref_ids = [uid for uid, keep in zip(all_ids, ref_mask, strict=True) if keep]
        qnu_ref_ids = [uid for uid, keep in zip(all_ids, qnu_ref_mask, strict=True) if keep]
        y_ref_frame = expr_all[ref_ids]
        fallback_mean = y_ref_frame.mean(axis=1).to_numpy(dtype=np.float32)

        truth = qnu.plate_expression_wide(
            plate_meta[plate_meta["user_compound_id"].astype(str).isin(val_ids)],
            gene_filter,
        )
        truth = truth.reindex(index=gene_filter, columns=val_ids)
        weights = qnu.load_weights(gene_filter, val_ids)

        scores: dict[str, pd.Series] = {}
        plate_meta_rows: dict[str, dict[str, object]] = {}
        target_cache: dict[tuple[str, ...], tuple[int, np.ndarray, np.ndarray, np.ndarray]] = {}

        for k in [5, 20, 50, 100]:
            pred = qnu.tanimoto_knn_predictions(
                all_ids=all_ids,
                all_fps=all_fps,
                y_all=y_all,
                gene_index=truth.index,
                val_ids=val_ids,
                ref_mask=ref_mask,
                k=k,
            )
            name = f"rdkit_tanimoto_knn_k{k}"
            scores[name] = qnu.score_wmse(truth, pred, weights)
            plate_meta_rows[name] = {
                "model_family": "tanimoto_knn",
                "features": "rdkit_morgan2048_tanimoto",
                "k": k,
                "ref_coverage": len(ref_ids),
                "val_coverage": len(val_ids),
            }

            pred = qnu.tanimoto_knn_predictions(
                all_ids=all_ids,
                all_fps=all_fps,
                y_all=y_all,
                gene_index=truth.index,
                val_ids=val_ids,
                ref_mask=qnu_ref_mask,
                k=k,
            )
            name = f"qnu_ref_rdkit_tanimoto_knn_k{k}"
            scores[name] = qnu.score_wmse(truth, pred, weights)
            plate_meta_rows[name] = {
                "model_family": "tanimoto_knn",
                "features": "rdkit_morgan2048_tanimoto",
                "k": k,
                "ref_coverage": len(qnu_ref_ids),
                "val_coverage": len(val_ids),
                "reference_scope": "tvc-qnu-012",
            }

        val_idx = np.array([all_id_index[uid] for uid in val_ids], dtype=np.int64)
        x_ref = rdkit_regression_all[ref_mask]
        x_ref_qnu = rdkit_regression_all[qnu_ref_mask]
        x_val = rdkit_regression_all[val_idx]
        y_ref = y_all[ref_mask]
        y_ref_qnu = y_all[qnu_ref_mask]
        for scope_name, scope_x_ref, scope_y_ref, scope_ref_n in [
            ("", x_ref, y_ref, len(ref_ids)),
            ("qnu_ref_", x_ref_qnu, y_ref_qnu, len(qnu_ref_ids)),
        ]:
            for n_target_components in TARGET_COMPONENTS:
                for alpha in RIDGE_ALPHAS:
                    pred_arr = qnu.target_pca_ridge_prediction(
                        x_ref=scope_x_ref,
                        x_val=x_val,
                        y_ref=scope_y_ref,
                        n_target_components=n_target_components,
                        alpha=alpha,
                    )
                    name = (
                        f"{scope_name}rdkit_morgan512+descriptors"
                        f"_target_pca{n_target_components}_ridge_alpha{alpha:g}"
                    )
                    score_prediction(
                        scores,
                        plate_meta_rows,
                        name=name,
                        pred_arr=pred_arr,
                        truth=truth,
                        val_ids=val_ids,
                        weights=weights,
                        row={
                            "model_family": "target_pca_ridge",
                            "features": "rdkit_morgan512+descriptors",
                            "n_target_components": n_target_components,
                            "alpha": alpha,
                            "ref_coverage": scope_ref_n,
                            "val_coverage": len(val_ids),
                            "reference_scope": "tvc-qnu-012" if scope_name else "all_active",
                        },
                    )

        pred_arr = qnu.target_pca_pls_prediction(
            x_ref=x_ref,
            x_val=x_val,
            y_ref=y_ref,
            n_target_components=64,
            n_pls_components=16,
        )
        score_prediction(
            scores,
            plate_meta_rows,
            name="rdkit_morgan512+descriptors_target_pca64_pls16",
            pred_arr=pred_arr,
            truth=truth,
            val_ids=val_ids,
            weights=weights,
            row={
                "model_family": "target_pca_pls",
                "features": "rdkit_morgan512+descriptors",
                "n_target_components": 64,
                "pls_components": 16,
                "ref_coverage": len(ref_ids),
                "val_coverage": len(val_ids),
            },
        )

        for feature in features:
            print(f"  feature {feature.name}", flush=True)
            add_feature_models_for_plate(
                feature=feature,
                ref_ids=ref_ids,
                val_ids=val_ids,
                y_ref_frame=y_ref_frame,
                y_all=y_all,
                all_id_index=all_id_index,
                fallback_mean=fallback_mean,
                truth=truth,
                weights=weights,
                target_cache=target_cache,
                scores=scores,
                meta=plate_meta_rows,
            )

        per_compound = pd.DataFrame({"plate_id": plate_id, "compound": val_ids})
        for model, series in scores.items():
            values = series.reindex(val_ids).to_numpy()
            per_compound[model] = values
            row = {
                "plate_id": plate_id,
                "model": model,
                "n_compounds": len(val_ids),
                "wmse_mean": float(np.mean(values)),
            }
            row.update(plate_meta_rows.get(model, {}))
            plate_rows.append(row)
            model_meta.setdefault(model, plate_meta_rows.get(model, {}))
        per_compound_pieces.append(per_compound)

    pd.DataFrame(split_rows).to_csv(qnu.ROOT / f"{args.output_prefix}_split.csv", index=False)
    per_compound_all = pd.concat(per_compound_pieces, ignore_index=True)
    summary = summarize(per_compound_all, model_meta)
    plate_summary = pd.DataFrame(plate_rows).sort_values(["model", "plate_id"]).reset_index(drop=True)

    out_prefix = qnu.ROOT / args.output_prefix
    summary.to_csv(f"{out_prefix}_summary.csv", index=False)
    plate_summary.to_csv(f"{out_prefix}_per_plate.csv", index=False)
    per_compound_all.to_csv(f"{out_prefix}_per_compound.csv", index=False)

    print("\nTop models:")
    cols = [
        "model",
        "wmse_mean",
        "model_family",
        "features",
        "n_target_components",
        "alpha",
        "shrink",
        "k",
        "ref_coverage",
        "val_coverage",
    ]
    print(summary.reindex(columns=cols).head(40).to_string(index=False), flush=True)
    print(f"\nWrote {out_prefix}_summary.csv", flush=True)


if __name__ == "__main__":
    main()
