"""
ECFP6 (radius=3) fingerprint sweep.

Tests whether larger structural neighborhoods improve predictions.
Current best uses ECFP4 (radius=2, 512-bit). This sweep tries:
  - radius=3, 512 / 1024 / 2048 bits + descriptors → PCA + Ridge

Run:
    python scripts/qnu_radius3_sweep.py \\
        --output-prefix eval_qnu_radius3
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


sweep.TARGET_COMPONENTS    = [64, 128, 192, 256]
sweep.RIDGE_ALPHAS         = [100.0, 1000.0, 10000.0]
sweep.DIRECT_RIDGE_ALPHAS  = []
sweep.DIRECT_RIDGE_SHRINKS = [1.0]
sweep.KNN_K                = []  # Tanimoto KNN still uses r=2 from qnu.build_chemistry_maps


def _radius3_build_feature_sets(
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
    desc_frame = chem[qnu.DESC_COLS].apply(pd.to_numeric, errors="coerce")
    desc_by_id = {uid: desc_frame.loc[uid].to_numpy(dtype=np.float32) for uid in desc_frame.index}

    features = []
    for nbits in [512, 1024, 2048]:
        fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=nbits)
        bits: dict[str, np.ndarray] = {}
        for uid, row in chem.iterrows():
            mol = Chem.MolFromSmiles(str(row["smiles"]))
            if mol is None:
                continue
            bits[uid] = qnu.bitvect_to_array(fp_gen.GetFingerprint(mol), nbits)
        features.append(sweep.FeatureSet(f"rdkit_morgan_r3_{nbits}", bits))
        common = sorted(set(bits) & set(desc_by_id))
        features.append(sweep.FeatureSet(
            f"rdkit_morgan_r3_{nbits}+descriptors",
            {uid: np.concatenate([bits[uid], desc_by_id[uid]]) for uid in common},
        ))
        print(f"  r=3 Morgan {nbits}-bit: {len(bits)} compounds", flush=True)

    return features


sweep.build_feature_sets = _radius3_build_feature_sets


if __name__ == "__main__":
    sweep.main()
