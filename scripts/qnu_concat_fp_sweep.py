"""
Concatenated fingerprint feature sweep.

Tests richer molecular representations by concatenating multiple FPs:
  - r2_512 + r3_512 (1024-dim)           — multi-radius ECFP4+6
  - r2_512 + r2_2048 (2560-dim)          — multi-resolution ECFP4
  - r2_512 + r3_512 + r2_2048 (3072-dim) — full combo
All variants + 8 descriptors.

Run:
    python scripts/qnu_concat_fp_sweep.py \\
        --output-prefix eval_qnu_concat_fp
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
sweep.KNN_K                = []


def _concat_build_feature_sets(
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

    # Compute individual FP banks
    banks: dict[str, dict[str, np.ndarray]] = {}
    for radius, nbits in [(2, 512), (2, 2048), (3, 512)]:
        key = f"r{radius}_{nbits}"
        fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
        bits: dict[str, np.ndarray] = {}
        for uid, row in chem.iterrows():
            mol = Chem.MolFromSmiles(str(row["smiles"]))
            if mol is None:
                continue
            bits[uid] = qnu.bitvect_to_array(fp_gen.GetFingerprint(mol), nbits)
        banks[key] = bits
        print(f"  Bank {key}: {len(bits)} compounds", flush=True)

    def concat_banks(*keys: str, suffix: str = "") -> sweep.FeatureSet:
        name = "+".join(keys) + (f"+{suffix}" if suffix else "")
        common = sorted(set.intersection(*[set(banks[k]) for k in keys]))
        return sweep.FeatureSet(
            name,
            {uid: np.concatenate([banks[k][uid] for k in keys]) for uid in common},
        )

    def concat_banks_desc(*keys: str) -> sweep.FeatureSet:
        common = sorted(set.intersection(*[set(banks[k]) for k in keys], set(desc_by_id)))
        name = "+".join(keys) + "+descriptors"
        return sweep.FeatureSet(
            name,
            {uid: np.concatenate([banks[k][uid] for k in keys] + [desc_by_id[uid]]) for uid in common},
        )

    features = [
        # r2+r3 multi-radius
        concat_banks("r2_512", "r3_512"),
        concat_banks_desc("r2_512", "r3_512"),

        # r2 multi-resolution
        concat_banks("r2_512", "r2_2048"),
        concat_banks_desc("r2_512", "r2_2048"),

        # r2+r3 multi-resolution
        concat_banks("r2_512", "r3_512", "r2_2048"),
        concat_banks_desc("r2_512", "r3_512", "r2_2048"),
    ]

    for fs in features:
        print(f"  Feature set '{fs.name}': {len(fs.by_user_id)} compounds, "
              f"dim={len(next(iter(fs.by_user_id.values()))) if fs.by_user_id else 0}", flush=True)

    return features


sweep.build_feature_sets = _concat_build_feature_sets


if __name__ == "__main__":
    sweep.main()
