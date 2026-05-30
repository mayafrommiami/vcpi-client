"""
Fine-grained hyperparameter search around the current best model:
  qnu_ref_rdkit_morgan512+descriptors_target_pca128_ridge_alpha1000

Runs ONLY rdkit_morgan512+descriptors with qnu-ref scope and a dense
alpha × PCA grid around the known optimum.

Run:
    python scripts/qnu_finetuned_hyperparam_sweep.py \\
        --output-prefix eval_qnu_finetuned_hyperparam
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


# Dense grid around the known optimum (alpha=1000, pca=128)
sweep.TARGET_COMPONENTS = [96, 112, 128, 144, 160, 192, 224, 256]
sweep.RIDGE_ALPHAS      = [50.0, 100.0, 200.0, 500.0, 750.0,
                            1000.0, 1500.0, 2000.0, 3000.0, 5000.0,
                            10000.0, 50000.0]
# Skip PLS and direct-ridge for speed — focus on target-PCA-ridge only
sweep.DIRECT_RIDGE_ALPHAS  = []
sweep.DIRECT_RIDGE_SHRINKS = [1.0]
sweep.KNN_K                = []   # skip KNN — already evaluated


# Only return the one feature set that matters
def _focused_build_feature_sets(
    train_chem: pd.DataFrame,
    all_ids: list[str],
) -> list[sweep.FeatureSet]:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    chem = train_chem.copy()
    chem["user_id"] = chem["user_compound_id"].astype(str)
    chem = (
        chem[chem["user_id"].isin(all_ids)]
        .drop_duplicates("user_id")
        .set_index("user_id")
    )
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


sweep.build_feature_sets = _focused_build_feature_sets


if __name__ == "__main__":
    sweep.main()
