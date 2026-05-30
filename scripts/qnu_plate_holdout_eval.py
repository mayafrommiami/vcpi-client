"""Leave-one-plate-out validation for the tvc-qnu-012 release.

The final query set is a held-out tvc-qnu-012 plate with active compounds
only. By default this script picks three ``tvc-qnu-012`` plates whose active
compounds have the highest plate-level Morgan Tanimoto similarity to the
bundled final test compounds, then holds out one selected ``container_id``
at a time. The reference side contains all other active compounds with
usable chemistry, excluding the held-out compound IDs entirely.

Usage:

    uv run python scripts/qnu_plate_holdout_eval.py

To audit all plates:

    uv run python scripts/qnu_plate_holdout_eval.py --plate-selection all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(__file__).resolve().parents[1]
GENE_COL = "gene_id"
QNU_JOB_ID = "tvc-qnu-012"
BHR_JOB_ID = "tvc-bhr-009"
KDL_JOB_ID = "tvc-kdl-010"
DEFAULT_NUM_PLATES = 3

DESC_COLS = [
    "molecular_weight",
    "log_p",
    "tpsa",
    "num_rotatable_bonds",
    "num_h_acceptors",
    "num_h_donors",
    "num_atoms",
    "num_bonds",
]


def canonical_smiles(smiles: str | float | None) -> str | None:
    if smiles is None or pd.isna(smiles):
        return None
    text = str(smiles).strip()
    if not text or text.lower() == "control":
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def bitvect_to_array(fp: DataStructs.ExplicitBitVect, n_bits: int) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def weights_columns(path: Path) -> set[str]:
    names = pq.ParquetFile(path).schema_arrow.names
    return set(names) - {GENE_COL}


def query_path() -> Path:
    candidates = [
        ROOT / "test_queries.csv",
        ROOT / "src/vcpi_prediction_contest/data_files/test_queries.csv",
        ROOT / "src/vcpi_prediction_contest/data_files/test_compounds.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find test_queries.csv or test_compounds.csv")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, list[str], set[str]]:
    train_chem = pd.read_parquet(ROOT / "train_chemistry.parquet")
    train_meta = pd.read_parquet(ROOT / "train_metadata.parquet")
    qpath = query_path()
    query = pd.read_csv(qpath)
    gene_filter = pd.read_csv(ROOT / "src/vcpi_prediction_contest/data_files/gene_filter.csv")[
        GENE_COL
    ].astype(str).tolist()
    weight_cols = weights_columns(ROOT / "weights.parquet")
    return train_chem, train_meta, query, qpath, gene_filter, weight_cols


def expression_wide(
    metadata: pd.DataFrame,
    compound_ids: list[str],
    gene_filter: list[str],
) -> pd.DataFrame:
    selected = set(compound_ids) | {"DMSO"}
    meta = metadata.copy()
    meta["sequenced_id"] = meta["sequenced_id"].astype(str)
    meta["user_compound_id"] = meta["user_compound_id"].astype(str)
    meta = meta[meta["user_compound_id"].isin(selected)].copy()
    if meta.empty:
        raise RuntimeError("No metadata rows selected for expression aggregation")

    sample_cols = meta["sequenced_id"].tolist()
    counts = pd.read_parquet(ROOT / "train_counts.parquet", columns=[GENE_COL, *sample_cols])
    counts[GENE_COL] = counts[GENE_COL].astype(str)
    counts = counts[counts[GENE_COL].isin(gene_filter)].set_index(GENE_COL).reindex(gene_filter)

    arr = counts.to_numpy(dtype=np.float32, copy=True)
    library_size = arr.sum(axis=0, dtype=np.float64).astype(np.float32)
    library_size[library_size == 0] = np.nan
    log_cpm = np.log2(arr / library_size[None, :] * np.float32(1_000_000.0) + np.float32(1.0))

    sample_to_compound = meta.set_index("sequenced_id").loc[sample_cols, "user_compound_id"].tolist()
    expr = pd.DataFrame(log_cpm, index=counts.index, columns=sample_to_compound)
    return expr.T.groupby(level=0, sort=True).mean().T.astype(np.float32)


def load_weights(gene_filter: list[str], val_ids: list[str]) -> pd.DataFrame:
    weights = pd.read_parquet(ROOT / "weights.parquet", columns=val_ids)
    weights.index = weights.index.astype(str)
    weights = weights.reindex(index=gene_filter, columns=val_ids)
    if weights.isna().any().any():
        raise RuntimeError("Weights are missing entries for the validation grid")
    return weights.astype(np.float32)


def score_wmse(
    truth: pd.DataFrame,
    pred: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.Series:
    pred = pred.reindex(index=truth.index, columns=truth.columns)
    weights = weights.reindex(index=truth.index, columns=truth.columns)
    err2 = (truth - pred) ** 2
    return (err2 * weights).sum(axis=0)


def standardize_train_val(x_ref: np.ndarray, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_ref.mean(axis=0, keepdims=True)
    std = x_ref.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (x_ref - mean) / std, (x_val - mean) / std


def plate_expression_wide(metadata: pd.DataFrame, gene_filter: list[str]) -> pd.DataFrame:
    """Aggregate expression for exactly the sample rows in ``metadata``."""
    sample_cols = metadata["sequenced_id"].astype(str).tolist()
    if not sample_cols:
        raise RuntimeError("Cannot build plate expression from empty metadata")

    counts = pd.read_parquet(ROOT / "train_counts.parquet", columns=[GENE_COL, *sample_cols])
    counts[GENE_COL] = counts[GENE_COL].astype(str)
    counts = counts[counts[GENE_COL].isin(gene_filter)].set_index(GENE_COL).reindex(gene_filter)

    arr = counts.to_numpy(dtype=np.float32, copy=True)
    library_size = arr.sum(axis=0, dtype=np.float64).astype(np.float32)
    library_size[library_size == 0] = np.nan
    log_cpm = np.log2(arr / library_size[None, :] * np.float32(1_000_000.0) + np.float32(1.0))

    labels = metadata["user_compound_id"].astype(str).tolist()
    expr = pd.DataFrame(log_cpm, index=counts.index, columns=labels)
    return expr.T.groupby(level=0, sort=True).mean().T.astype(np.float32)


def build_chemistry_maps(
    train_chem: pd.DataFrame,
    active_ids: set[str],
    weight_cols: set[str],
) -> tuple[list[str], dict[str, DataStructs.ExplicitBitVect]]:
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    chem = train_chem.copy()
    chem["user_id"] = chem["user_compound_id"].astype(str)
    chem = chem[chem["user_id"].isin(active_ids) & chem["user_id"].isin(weight_cols)].copy()
    chem["canon_smiles"] = chem["smiles"].map(canonical_smiles)
    chem = chem.dropna(subset=["canon_smiles"]).drop_duplicates("user_id")

    fps: dict[str, DataStructs.ExplicitBitVect] = {}
    for row in chem.itertuples(index=False):
        mol = Chem.MolFromSmiles(row.canon_smiles)
        if mol is None:
            continue
        fps[str(row.user_id)] = fp_gen.GetFingerprint(mol)
    eligible_ids = sorted(fps)
    return eligible_ids, fps


def regression_features(train_chem: pd.DataFrame, all_ids: list[str]) -> np.ndarray:
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    chem = train_chem.copy()
    chem["user_id"] = chem["user_compound_id"].astype(str)
    chem = chem.drop_duplicates("user_id").set_index("user_id")
    rows = []
    for uid in all_ids:
        row = chem.loc[uid]
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        bits = bitvect_to_array(fp_gen.GetFingerprint(mol), 512)
        desc = pd.to_numeric(row[DESC_COLS], errors="coerce").to_numpy(dtype=np.float32)
        rows.append(np.concatenate([bits, desc]))
    return np.vstack(rows).astype(np.float32)


def query_fingerprints(query: pd.DataFrame) -> list[DataStructs.ExplicitBitVect]:
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    if "smiles" not in query.columns:
        raise RuntimeError("Query/test compound file must contain a smiles column")
    fps = []
    for smiles in query["smiles"].map(canonical_smiles).dropna():
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fps.append(fp_gen.GetFingerprint(mol))
    if not fps:
        raise RuntimeError("No query/test compound SMILES could be parsed")
    return fps


def plate_similarity_stats(
    qnu_active: pd.DataFrame,
    all_fps: dict[str, DataStructs.ExplicitBitVect],
    query_fps: list[DataStructs.ExplicitBitVect],
) -> pd.DataFrame:
    rows = []
    for plate_id, sub in qnu_active.groupby("container_id"):
        sims = []
        for uid in sorted(sub["user_compound_id"].astype(str).unique()):
            if uid not in all_fps:
                continue
            vals = DataStructs.BulkTanimotoSimilarity(all_fps[uid], query_fps)
            sims.append(max(vals))
        if not sims:
            continue
        series = pd.Series(sims, dtype=float)
        rows.append(
            {
                "plate_id": int(plate_id),
                "n_active": int(len(sims)),
                "mean_max_test_tanimoto": float(series.mean()),
                "median_max_test_tanimoto": float(series.median()),
                "p90_max_test_tanimoto": float(series.quantile(0.9)),
                "max_test_tanimoto": float(series.max()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["median_max_test_tanimoto", "mean_max_test_tanimoto", "plate_id"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def select_plates(
    *,
    args: argparse.Namespace,
    qnu_active: pd.DataFrame,
    all_fps: dict[str, DataStructs.ExplicitBitVect],
    query: pd.DataFrame,
) -> tuple[list[int], pd.DataFrame]:
    stats = plate_similarity_stats(qnu_active, all_fps, query_fingerprints(query))
    available = sorted(int(x) for x in qnu_active["container_id"].unique())

    if args.plates:
        requested = [int(p) for p in args.plates]
        selected = [p for p in requested if p in set(available)]
    elif args.plate_selection == "all":
        selected = available
    elif args.plate_selection == "first":
        selected = available[: args.num_plates]
    else:
        selected = stats["plate_id"].head(args.num_plates).astype(int).tolist()

    if args.max_plates is not None:
        selected = selected[: args.max_plates]
    if not selected:
        raise RuntimeError("No tvc-qnu-012 plates selected")

    stats["selected"] = stats["plate_id"].isin(selected)
    return selected, stats


def target_pca_pls_prediction(
    *,
    x_ref: np.ndarray,
    x_val: np.ndarray,
    y_ref: np.ndarray,
    n_target_components: int,
    n_pls_components: int,
) -> np.ndarray:
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.utils.extmath import randomized_svd

    x_ref, x_val = standardize_train_val(x_ref, x_val)
    y_mean = y_ref.mean(axis=0, keepdims=True).astype(np.float32)
    y_centered = y_ref - y_mean
    u, s, vt = randomized_svd(
        y_centered,
        n_components=n_target_components,
        n_iter=5,
        random_state=13,
    )
    target_scores = (u * s[None, :]).astype(np.float32)
    model = PLSRegression(
        n_components=n_pls_components,
        scale=True,
        max_iter=300,
        tol=1e-5,
    )
    model.fit(x_ref, target_scores)
    pred_scores = model.predict(x_val).astype(np.float32)
    return np.clip(pred_scores @ vt.astype(np.float32) + y_mean, 0.0, None)


def target_pca_ridge_prediction(
    *,
    x_ref: np.ndarray,
    x_val: np.ndarray,
    y_ref: np.ndarray,
    n_target_components: int,
    alpha: float,
) -> np.ndarray:
    from sklearn.linear_model import Ridge
    from sklearn.utils.extmath import randomized_svd

    x_ref, x_val = standardize_train_val(x_ref, x_val)
    y_mean = y_ref.mean(axis=0, keepdims=True).astype(np.float32)
    y_centered = y_ref - y_mean
    u, s, vt = randomized_svd(
        y_centered,
        n_components=n_target_components,
        n_iter=5,
        random_state=13,
    )
    target_scores = (u * s[None, :]).astype(np.float32)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x_ref, target_scores)
    pred_scores = model.predict(x_val).astype(np.float32)
    return np.clip(pred_scores @ vt.astype(np.float32) + y_mean, 0.0, None)


def tanimoto_knn_predictions(
    *,
    all_ids: list[str],
    all_fps: dict[str, DataStructs.ExplicitBitVect],
    y_all: np.ndarray,
    gene_index: pd.Index,
    val_ids: list[str],
    ref_mask: np.ndarray,
    k: int,
) -> pd.DataFrame:
    ref_positions = np.flatnonzero(ref_mask)
    ref_fps = [all_fps[all_ids[i]] for i in ref_positions]
    preds = []
    for uid in val_ids:
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(all_fps[uid], ref_fps), dtype=np.float32)
        k_eff = min(k, len(sims))
        take_local = np.argpartition(-sims, k_eff - 1)[:k_eff]
        sims_k = sims[take_local]
        if float(sims_k.sum()) <= 1e-8:
            weights = np.full(k_eff, 1.0 / k_eff, dtype=np.float32)
        else:
            weights = sims_k / sims_k.sum()
        take = ref_positions[take_local]
        preds.append(weights @ y_all[take])
    return pd.DataFrame(np.vstack(preds).T, index=gene_index, columns=val_ids)


def score_plate(
    *,
    plate_id: int,
    plate_meta: pd.DataFrame,
    all_ids: list[str],
    active_ids_by_job: dict[str, set[str]],
    all_fps: dict[str, DataStructs.ExplicitBitVect],
    expr_all: pd.DataFrame,
    expr_by_job: dict[str, pd.DataFrame],
    y_all: np.ndarray,
    feature_all: np.ndarray | None,
    gene_filter: list[str],
    include_regression: bool,
    target_components: int,
    pls_components: int,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    val_ids = sorted(set(plate_meta["user_compound_id"].astype(str)) & set(all_ids))
    if not val_ids:
        return [], pd.DataFrame()

    all_id_index = {uid: i for i, uid in enumerate(all_ids)}
    val_positions = {all_id_index[uid] for uid in val_ids}
    ref_mask = np.array([i not in val_positions for i in range(len(all_ids))], dtype=bool)
    ref_ids = [uid for uid, keep in zip(all_ids, ref_mask, strict=True) if keep]

    truth = plate_expression_wide(plate_meta[plate_meta["user_compound_id"].astype(str).isin(val_ids)], gene_filter)
    truth = truth.reindex(index=gene_filter, columns=val_ids)
    weights = load_weights(gene_filter, val_ids)

    scores: dict[str, pd.Series] = {}
    reference_counts: dict[str, int] = {}

    def add_mean_baseline(name: str, expr: pd.DataFrame, candidate_ids: set[str]) -> None:
        ref_subset = [uid for uid in ref_ids if uid in candidate_ids and uid in expr.columns]
        if not ref_subset:
            return
        mean_expr = expr[ref_subset].mean(axis=1).to_numpy(dtype=np.float32)
        pred = pd.DataFrame(np.broadcast_to(mean_expr[:, None], truth.shape), index=truth.index, columns=val_ids)
        scores[name] = score_wmse(truth, pred, weights)
        reference_counts[name] = len(ref_subset)

    add_mean_baseline("all_active_mean_without_plate", expr_all, set(ref_ids))
    add_mean_baseline(
        "qnu_active_mean_without_plate",
        expr_by_job[QNU_JOB_ID],
        active_ids_by_job[QNU_JOB_ID],
    )
    add_mean_baseline(
        "bhr_active_mean_without_plate",
        expr_by_job[BHR_JOB_ID],
        active_ids_by_job[BHR_JOB_ID],
    )
    add_mean_baseline(
        "kdl_active_mean_without_plate",
        expr_by_job[KDL_JOB_ID],
        active_ids_by_job[KDL_JOB_ID],
    )

    if "DMSO" in expr_all.columns:
        dmso = expr_all["DMSO"].to_numpy(dtype=np.float32)
        pred = pd.DataFrame(np.broadcast_to(dmso[:, None], truth.shape), index=truth.index, columns=val_ids)
        scores["dmso_control_mean"] = score_wmse(truth, pred, weights)
        reference_counts["dmso_control_mean"] = 1

    for k in [5, 20, 50, 100]:
        pred = tanimoto_knn_predictions(
            all_ids=all_ids,
            all_fps=all_fps,
            y_all=y_all,
            gene_index=truth.index,
            val_ids=val_ids,
            ref_mask=ref_mask,
            k=k,
        )
        scores[f"rdkit_tanimoto_knn_k{k}"] = score_wmse(truth, pred, weights)
        reference_counts[f"rdkit_tanimoto_knn_k{k}"] = len(ref_ids)

    if include_regression:
        if feature_all is None:
            raise RuntimeError("feature_all is required when include_regression=True")
        val_idx = np.array([all_id_index[uid] for uid in val_ids], dtype=np.int64)
        x_ref = feature_all[ref_mask]
        x_val = feature_all[val_idx]
        y_ref = y_all[ref_mask]

        pred_arr = target_pca_pls_prediction(
            x_ref=x_ref,
            x_val=x_val,
            y_ref=y_ref,
            n_target_components=target_components,
            n_pls_components=pls_components,
        )
        pred = pd.DataFrame(pred_arr.T, index=truth.index, columns=val_ids)
        scores[
            f"rdkit_morgan512+descriptors_target_pca{target_components}_pls{pls_components}"
        ] = score_wmse(truth, pred, weights)
        reference_counts[
            f"rdkit_morgan512+descriptors_target_pca{target_components}_pls{pls_components}"
        ] = len(ref_ids)

        pred_arr = target_pca_ridge_prediction(
            x_ref=x_ref,
            x_val=x_val,
            y_ref=y_ref,
            n_target_components=target_components,
            alpha=1000.0,
        )
        pred = pd.DataFrame(pred_arr.T, index=truth.index, columns=val_ids)
        scores[
            f"rdkit_morgan512+descriptors_target_pca{target_components}_ridge_alpha1000"
        ] = score_wmse(truth, pred, weights)
        reference_counts[
            f"rdkit_morgan512+descriptors_target_pca{target_components}_ridge_alpha1000"
        ] = len(ref_ids)

    rows = []
    per_compound = pd.DataFrame({"plate_id": plate_id, "compound": val_ids})
    for model, series in scores.items():
        values = series.reindex(val_ids).to_numpy()
        per_compound[model] = values
        rows.append(
            {
                "plate_id": plate_id,
                "model": model,
                "n_compounds": len(val_ids),
                "wmse_mean": float(np.mean(values)),
                "reference_compounds": reference_counts[model],
            }
        )
    return rows, per_compound


def summarize(per_compound: pd.DataFrame) -> pd.DataFrame:
    model_cols = [c for c in per_compound.columns if c not in {"plate_id", "compound"}]
    rows = []
    for model in model_cols:
        values = per_compound[model].to_numpy()
        rows.append(
            {
                "model": model,
                "n_plates": int(per_compound["plate_id"].nunique()),
                "n_compounds": int(per_compound["compound"].nunique()),
                "wmse_mean": float(np.mean(values)),
            }
        )
    return pd.DataFrame(rows).sort_values(["wmse_mean", "model"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plates", nargs="*", type=int, default=None)
    parser.add_argument("--num-plates", type=int, default=DEFAULT_NUM_PLATES)
    parser.add_argument("--max-plates", type=int, default=None, help="Deprecated alias for truncating selected plates.")
    parser.add_argument(
        "--plate-selection",
        choices=["test-similar", "first", "all"],
        default="test-similar",
    )
    parser.add_argument("--output-prefix", default="eval_qnu_plate_holdout_quick")
    parser.add_argument("--include-regression", action="store_true")
    parser.add_argument("--target-components", type=int, default=64)
    parser.add_argument("--pls-components", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_chem, train_meta, query, _qpath, gene_filter, weight_cols = load_inputs()
    train_meta = train_meta.copy()
    train_meta["user_compound_id"] = train_meta["user_compound_id"].astype(str)

    active_mask = ~train_meta["is_control"].astype(bool)
    active_ids = set(train_meta.loc[active_mask, "user_compound_id"])
    all_ids, all_fps = build_chemistry_maps(train_chem, active_ids, weight_cols)

    qnu_active = train_meta[
        (train_meta["job_id"] == QNU_JOB_ID) & active_mask & train_meta["user_compound_id"].isin(all_ids)
    ].copy()
    active_ids_by_job = {
        job_id: set(
            train_meta.loc[
                (train_meta["job_id"] == job_id)
                & active_mask
                & train_meta["user_compound_id"].isin(all_ids),
                "user_compound_id",
            ]
            .astype(str)
            .unique()
        )
        for job_id in [BHR_JOB_ID, KDL_JOB_ID, QNU_JOB_ID]
    }
    plates, plate_stats = select_plates(args=args, qnu_active=qnu_active, all_fps=all_fps, query=query)
    plate_stats.to_csv(ROOT / f"{args.output_prefix}_plate_similarity.csv", index=False)

    split_rows = []
    for plate_id in plates:
        ids = sorted(
            qnu_active.loc[qnu_active["container_id"] == plate_id, "user_compound_id"].astype(str).unique()
        )
        for uid in ids:
            split_rows.append({"plate_id": plate_id, "compound": uid, "split_role": "validation"})
    pd.DataFrame(split_rows).to_csv(ROOT / f"{args.output_prefix}_split.csv", index=False)

    print(
        f"Selected {len(plates)} qnu plates with "
        f"{len({r['compound'] for r in split_rows})} unique active validation compounds",
        flush=True,
    )
    print(
        "Selected plates: "
        + ", ".join(
            f"{row.plate_id}(medianT={row.median_max_test_tanimoto:.3f}, meanT={row.mean_max_test_tanimoto:.3f})"
            for row in plate_stats[plate_stats["selected"]].itertuples(index=False)
        ),
        flush=True,
    )
    print(f"Eligible active reference universe: {len(all_ids)} compounds", flush=True)

    print("Aggregating all active-compound expression once...", flush=True)
    expr_all = expression_wide(train_meta, all_ids, gene_filter)
    expr_by_job = {}
    for job_id in [BHR_JOB_ID, KDL_JOB_ID, QNU_JOB_ID]:
        print(f"Aggregating {job_id} active-compound expression...", flush=True)
        expr_by_job[job_id] = expression_wide(
            train_meta[train_meta["job_id"] == job_id],
            sorted(active_ids_by_job[job_id]),
            gene_filter,
        )
    y_all = expr_all[all_ids].T.to_numpy(dtype=np.float32)
    feature_all = None
    if args.include_regression:
        print("Building RDKit Morgan512+descriptor regression features...", flush=True)
        feature_all = regression_features(train_chem, all_ids)

    plate_rows: list[dict[str, object]] = []
    per_compound_pieces: list[pd.DataFrame] = []
    for idx, plate_id in enumerate(plates, start=1):
        print(f"[{idx}/{len(plates)}] plate {plate_id}", flush=True)
        plate_meta = qnu_active[qnu_active["container_id"] == plate_id].copy()
        rows, per_compound = score_plate(
            plate_id=plate_id,
            plate_meta=plate_meta,
            all_ids=all_ids,
            active_ids_by_job=active_ids_by_job,
            all_fps=all_fps,
            expr_all=expr_all,
            expr_by_job=expr_by_job,
            y_all=y_all,
            feature_all=feature_all,
            gene_filter=gene_filter,
            include_regression=args.include_regression,
            target_components=args.target_components,
            pls_components=args.pls_components,
        )
        plate_rows.extend(rows)
        if not per_compound.empty:
            per_compound_pieces.append(per_compound)

    plate_summary = pd.DataFrame(plate_rows).sort_values(["model", "plate_id"]).reset_index(drop=True)
    per_compound = pd.concat(per_compound_pieces, ignore_index=True)
    overall = summarize(per_compound)

    out_prefix = ROOT / args.output_prefix
    overall.to_csv(f"{out_prefix}_summary.csv", index=False)
    plate_summary.to_csv(f"{out_prefix}_per_plate.csv", index=False)
    per_compound.to_csv(f"{out_prefix}_per_compound.csv", index=False)

    print("\nOverall plate-holdout results:")
    print(overall.to_string(index=False))
    print(f"\nWrote {out_prefix}_summary.csv")
    print(f"Wrote {out_prefix}_per_plate.csv")
    print(f"Wrote {out_prefix}_per_compound.csv")
    print(f"Wrote {out_prefix}_split.csv")
    print(f"Wrote {out_prefix}_plate_similarity.csv")


if __name__ == "__main__":
    main()
