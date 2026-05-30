"""
Lasso/ElasticNet + weight-filtered PCA sweep.

Two key ideas motivated by the contest info that every test compound
has ≥5 differentially expressed genes (LFC≥0.5, padj≤0.05):

  1. Lasso / ElasticNet instead of Ridge — L1 pushes predictions toward
     zero for inactive genes, concentrating signal on the genes that are
     actually differentially expressed (what wMSE rewards).

  2. Weight-filtered PCA — decompose only the top-N contest-weighted
     genes before fitting. The PCA components then represent directions
     that matter to the leaderboard, not just directions of raw variance.

Run:
    python scripts/qnu_lasso_weighted_sweep.py \\
        --output-prefix eval_qnu_lasso_weighted
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import qnu_plate_holdout_eval as qnu
import qnu_plate_extended_model_sweep as sweep
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.utils.extmath import randomized_svd


# Hyperparameter grids
LASSO_ALPHAS      = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
ENET_ALPHAS       = [0.005, 0.01, 0.05, 0.1, 0.5]
ENET_L1_RATIOS    = [0.5, 0.75, 0.9]
RIDGE_ALPHAS_FINE = [500.0, 1000.0, 2000.0]   # focused on known-good region

# Gene subset sizes for weight-filtered PCA
TOP_GENE_COUNTS   = [500, 1000, 2000, 3000]
RIDGE_PCA_COMPONENTS = [64, 128, 192]
RIDGE_PCA_ALPHAS  = [100.0, 1000.0, 10000.0]

SEED = 13

# Disable the stock feature-set loop — we provide everything inline
sweep.DIRECT_RIDGE_ALPHAS  = []
sweep.DIRECT_RIDGE_SHRINKS = [1.0]
sweep.KNN_K                = []
sweep.TARGET_COMPONENTS    = []   # we handle target-PCA ourselves
sweep.RIDGE_ALPHAS         = []


# ── helpers ──────────────────────────────────────────────────────────────────

def load_contest_gene_weights(gene_filter: list[str]) -> np.ndarray:
    """Per-gene contest weights — proxy variance from weights.parquet."""
    w_path = ROOT / "weights.parquet"
    if not w_path.exists():
        return np.ones(len(gene_filter), dtype=np.float32)
    w = pd.read_parquet(w_path)
    # weights.parquet has genes as index, compounds as columns — mean across compounds
    if "gene_id" in w.columns:
        w = w.set_index("gene_id")
    gene_means = w.mean(axis=1).rename("weight")
    weights = gene_means.reindex(gene_filter).fillna(gene_means.median()).to_numpy(dtype=np.float32)
    return weights / weights.mean()   # normalise to mean=1


def weighted_target_basis(
    y_ref: np.ndarray,
    gene_weights: np.ndarray,
    *,
    top_n: int,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA on the top-N contest-weighted genes only.
    Returns (y_mean, vt_full, pc_ref, gene_mask) where vt_full has shape
    (n_components, n_all_genes) so reconstruction works on the full space.
    """
    # Select top-N genes by contest weight
    gene_mask = np.argsort(gene_weights)[-top_n:]   # indices of top-N genes
    y_sub    = y_ref[:, gene_mask]
    y_mean   = y_ref.mean(axis=0, keepdims=True).astype(np.float32)
    y_sub_c  = y_sub - y_mean[:, gene_mask]

    n_comp = min(n_components, y_sub_c.shape[0] - 1, y_sub_c.shape[1] - 1)
    u, s, vt_sub = randomized_svd(y_sub_c, n_components=n_comp, n_iter=5, random_state=SEED)

    # Embed vt back into full gene space (zeros for non-selected genes)
    vt_full = np.zeros((n_comp, y_ref.shape[1]), dtype=np.float32)
    vt_full[:, gene_mask] = vt_sub.astype(np.float32)
    pc_ref = (u * s[None, :]).astype(np.float32)

    return y_mean, vt_full, pc_ref, gene_mask


def reconstruct_full(
    pred_scores: np.ndarray,
    y_mean: np.ndarray,
    vt_full: np.ndarray,
) -> np.ndarray:
    return np.clip(pred_scores @ vt_full + y_mean, 0.0, None).astype(np.float32)


# ── patched build_feature_sets: only morgan512+desc ──────────────────────────

def _focused_features(train_chem: pd.DataFrame, all_ids: list[str]) -> list[sweep.FeatureSet]:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    chem = train_chem.copy()
    chem["user_id"] = chem["user_compound_id"].astype(str)
    chem = chem[chem["user_id"].isin(all_ids)].drop_duplicates("user_id").set_index("user_id")
    fp_gen     = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    desc_frame = chem[qnu.DESC_COLS].apply(pd.to_numeric, errors="coerce")
    bits: dict[str, np.ndarray] = {}
    for uid, row in chem.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None:
            continue
        bits[uid] = qnu.bitvect_to_array(fp_gen.GetFingerprint(mol), 512)
    desc_by_id = {uid: desc_frame.loc[uid].to_numpy(dtype=np.float32) for uid in desc_frame.index}
    common = sorted(set(bits) & set(desc_by_id))
    return [sweep.FeatureSet(
        "rdkit_morgan512+descriptors",
        {uid: np.concatenate([bits[uid], desc_by_id[uid]]) for uid in common},
    )]


sweep.build_feature_sets = _focused_features


# ── patched add_feature_models_for_plate ─────────────────────────────────────

def _add_lasso_enet_and_weighted_pca(
    *,
    feature: sweep.FeatureSet,
    ref_ids: list[str],
    val_ids: list[str],
    y_ref_frame: pd.DataFrame,
    y_all: np.ndarray,
    all_id_index: dict[str, int],
    fallback_mean: np.ndarray,
    truth: pd.DataFrame,
    weights: pd.DataFrame,
    target_cache: dict,
    scores: dict,
    meta: dict,
    gene_weights: np.ndarray,
) -> None:
    ref_cov, x_ref = sweep.feature_matrix(feature, ref_ids)
    val_cov, x_val = sweep.feature_matrix(feature, val_ids)
    if len(ref_cov) < 20 or len(val_cov) == 0:
        return

    ref_pos  = np.array([all_id_index[uid] for uid in ref_cov], dtype=np.int64)
    y_ref    = y_all[ref_pos]
    val_pos  = {uid: i for i, uid in enumerate(val_ids)}
    x_ref_s, x_val_s = sweep.robust_standardize_train_val(x_ref, x_val)
    y_mean   = y_ref.mean(axis=0, keepdims=True).astype(np.float32)

    def fill(pred_cov: np.ndarray) -> np.ndarray:
        out = np.broadcast_to(fallback_mean[None, :], (len(val_ids), len(fallback_mean))).copy()
        for i, uid in enumerate(val_cov):
            out[val_pos[uid]] = pred_cov[i]
        return out.astype(np.float32)

    n_genes = y_ref.shape[1]

    # ── 1. Lasso on raw targets ───────────────────────────────────────────────
    y_centered = y_ref - y_mean
    for alpha in LASSO_ALPHAS:
        try:
            model = Lasso(alpha=alpha, fit_intercept=False, max_iter=2000)
            model.fit(x_ref_s, y_centered)
            pred = np.clip(y_mean + model.predict(x_val_s), 0.0, None).astype(np.float32)
            sweep.score_prediction(scores, meta,
                name=f"{feature.name}__lasso_alpha{alpha:g}",
                pred_arr=fill(pred), truth=truth, val_ids=val_ids, weights=weights,
                row={"model_family": "lasso", "features": feature.name,
                     "alpha": alpha, "ref_coverage": len(ref_cov), "val_coverage": len(val_cov)})
        except Exception as e:
            print(f"  Lasso alpha={alpha} failed: {e}", flush=True)

    # ── 2. ElasticNet ─────────────────────────────────────────────────────────
    for alpha in ENET_ALPHAS:
        for l1 in ENET_L1_RATIOS:
            try:
                model = ElasticNet(alpha=alpha, l1_ratio=l1, fit_intercept=False, max_iter=2000)
                model.fit(x_ref_s, y_centered)
                pred = np.clip(y_mean + model.predict(x_val_s), 0.0, None).astype(np.float32)
                sweep.score_prediction(scores, meta,
                    name=f"{feature.name}__enet_alpha{alpha:g}_l1r{l1:g}",
                    pred_arr=fill(pred), truth=truth, val_ids=val_ids, weights=weights,
                    row={"model_family": "elasticnet", "features": feature.name,
                         "alpha": alpha, "l1_ratio": l1,
                         "ref_coverage": len(ref_cov), "val_coverage": len(val_cov)})
            except Exception as e:
                print(f"  ElasticNet alpha={alpha} l1={l1} failed: {e}", flush=True)

    # ── 3. Weight-filtered PCA + Ridge ───────────────────────────────────────
    for top_n in TOP_GENE_COUNTS:
        if top_n >= n_genes:
            continue
        for n_comp in RIDGE_PCA_COMPONENTS:
            try:
                y_mean_full, vt_full, pc_ref, _ = weighted_target_basis(
                    y_ref, gene_weights, top_n=top_n, n_components=n_comp
                )
                y_pc = pc_ref[:, :n_comp]
                for alpha in RIDGE_PCA_ALPHAS:
                    ridge = Ridge(alpha=alpha, fit_intercept=True)
                    ridge.fit(x_ref_s, y_pc)
                    pred_pc   = ridge.predict(x_val_s).astype(np.float32)
                    pred_full = reconstruct_full(pred_pc, y_mean_full, vt_full[:n_comp])
                    sweep.score_prediction(scores, meta,
                        name=f"{feature.name}__wtpca{n_comp}_top{top_n}_ridge_alpha{alpha:g}",
                        pred_arr=fill(pred_full), truth=truth, val_ids=val_ids, weights=weights,
                        row={"model_family": "weighted_target_pca_ridge",
                             "features": feature.name,
                             "n_target_components": n_comp, "top_n_genes": top_n,
                             "alpha": alpha,
                             "ref_coverage": len(ref_cov), "val_coverage": len(val_cov)})
            except Exception as e:
                print(f"  WeightedPCA top{top_n} n{n_comp} alpha={alpha} failed: {e}", flush=True)


sweep.add_feature_models_for_plate = lambda **kw: _add_lasso_enet_and_weighted_pca(
    gene_weights=_GENE_WEIGHTS, **kw
)

_GENE_WEIGHTS: np.ndarray = None   # filled in main()


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global _GENE_WEIGHTS

    # Load gene filter first to get gene order
    _, _, _, _, gene_filter, _ = qnu.load_inputs()
    _GENE_WEIGHTS = load_contest_gene_weights(list(gene_filter))
    print(f"Gene weights loaded: {len(_GENE_WEIGHTS)} genes, "
          f"top-500 threshold = {np.sort(_GENE_WEIGHTS)[-500]:.3f}", flush=True)

    sweep.main()


if __name__ == "__main__":
    main()
