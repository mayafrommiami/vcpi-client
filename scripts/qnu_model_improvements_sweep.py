"""
Model improvements sweep — patches qnu_plate_extended_model_sweep with:

  A  Morgan 1024 & 2048-bit fingerprints + descriptors → PCA + Ridge
  B  ChemBERTa 384-dim frozen embeddings + descriptors → PCA + Ridge
  C  Top-model ensemble (mean of best qnu-ref Ridge + KNN predictions)

Run:
    python scripts/qnu_model_improvements_sweep.py \\
        --output-prefix eval_qnu_improvements

Everything else (plate selection, scoring, output CSVs) is inherited from
qnu_plate_extended_model_sweep.main().
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models"))

import qnu_plate_holdout_eval as qnu
import qnu_plate_extended_model_sweep as sweep


# ── A: widen the target-PCA grid ─────────────────────────────────────────────
sweep.TARGET_COMPONENTS = [64, 128, 192, 256]
sweep.RIDGE_ALPHAS      = [100.0, 1000.0, 10000.0]


# ── B: ChemBERTa feature builder ─────────────────────────────────────────────
def _build_chemberta_features(
    chem_indexed: pd.DataFrame,
    all_ids: list[str],
    desc_by_id: dict[str, np.ndarray],
) -> list[sweep.FeatureSet]:
    import torch
    from transformers import AutoTokenizer, AutoModel

    MODEL_ID = "seyonec/ChemBERTa-zinc-base-v1"
    print(f"  Loading ChemBERTa: {MODEL_ID}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModel.from_pretrained(MODEL_ID)
    model.eval()

    ids          = [uid for uid in all_ids if uid in chem_indexed.index]
    smiles_list  = [str(chem_indexed.loc[uid, "smiles"]) for uid in ids]
    embed_by_id: dict[str, np.ndarray] = {}

    BATCH = 64
    for i in range(0, len(ids), BATCH):
        b_smi = smiles_list[i : i + BATCH]
        b_ids = ids[i : i + BATCH]
        valid = [(s, uid) for s, uid in zip(b_smi, b_ids) if s and s != "nan"]
        if not valid:
            continue
        v_smi, v_ids = zip(*valid)
        inputs = tokenizer(
            list(v_smi),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            cls = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
        for uid, emb in zip(v_ids, cls):
            embed_by_id[uid] = emb
        if i % (BATCH * 5) == 0:
            print(f"    ChemBERTa {len(embed_by_id)}/{len(ids)}", flush=True)

    print(f"  ChemBERTa: {len(embed_by_id)}/{len(ids)} embedded (384-dim)", flush=True)

    results = [sweep.FeatureSet("chemberta384", embed_by_id)]
    common  = sorted(set(embed_by_id) & set(desc_by_id))
    results.append(sweep.FeatureSet(
        "chemberta384+descriptors",
        {uid: np.concatenate([embed_by_id[uid], desc_by_id[uid]]) for uid in common},
    ))
    return results


# ── Patched build_feature_sets ────────────────────────────────────────────────
_orig_build = sweep.build_feature_sets


def _extended_build_feature_sets(
    train_chem: pd.DataFrame,
    all_ids: list[str],
) -> list[sweep.FeatureSet]:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    features = _orig_build(train_chem, all_ids)

    chem = train_chem.copy()
    chem["user_id"] = chem["user_compound_id"].astype(str)
    chem = (
        chem[chem["user_id"].isin(all_ids)]
        .drop_duplicates("user_id")
        .set_index("user_id")
    )
    desc_frame = chem[qnu.DESC_COLS].apply(pd.to_numeric, errors="coerce")
    desc_by_id = {uid: desc_frame.loc[uid].to_numpy(dtype=np.float32) for uid in desc_frame.index}

    # Option A: Morgan 1024 & 2048
    for nbits in [1024, 2048]:
        fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nbits)
        bits: dict[str, np.ndarray] = {}
        for uid, row in chem.iterrows():
            mol = Chem.MolFromSmiles(str(row["smiles"]))
            if mol is None:
                continue
            bits[uid] = qnu.bitvect_to_array(fp_gen.GetFingerprint(mol), nbits)
        features.append(sweep.FeatureSet(f"rdkit_morgan{nbits}", bits))
        common = sorted(set(bits) & set(desc_by_id))
        features.append(sweep.FeatureSet(
            f"rdkit_morgan{nbits}+descriptors",
            {uid: np.concatenate([bits[uid], desc_by_id[uid]]) for uid in common},
        ))
        print(f"  Morgan {nbits}-bit: {len(bits)} compounds", flush=True)

    # Option B: ChemBERTa
    try:
        for fs in _build_chemberta_features(chem, all_ids, desc_by_id):
            features.append(fs)
    except Exception as e:
        print(f"  WARNING: ChemBERTa failed, skipping: {e}", flush=True)

    return features


sweep.build_feature_sets = _extended_build_feature_sets


# ── C: Ensemble — wrap main() to blend top qnu-ref predictions ───────────────
_orig_main = sweep.main


def _main_with_ensemble() -> None:
    """
    Runs the patched sweep then adds an ensemble model that averages
    the per-compound wMSE-weighted predictions of the top qnu-ref models.
    The ensemble is appended to the output CSVs.
    """
    import argparse

    # Parse args ourselves so we know the output prefix
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", default="eval_qnu_improvements")
    parser.add_argument("--num-plates", type=int, default=qnu.DEFAULT_NUM_PLATES)
    parser.add_argument("--plates", nargs="*", type=int, default=None)
    args, _ = parser.parse_known_args()

    # Run the full sweep (patches already applied above)
    _orig_main()

    # Post-hoc ensemble: blend top qnu-ref models from the summary CSV
    summary_path = ROOT / f"{args.output_prefix}_summary.csv"
    per_plate_path = ROOT / f"{args.output_prefix}_per_plate.csv"
    if not summary_path.exists():
        return

    summary    = pd.read_csv(summary_path)
    per_plate  = pd.read_csv(per_plate_path)

    # Pick top-5 qnu-ref models
    qnu_ref_mask = summary["model"].str.startswith("qnu_ref")
    top_models   = summary.loc[qnu_ref_mask, "model"].head(5).tolist()
    if len(top_models) < 2:
        return

    # Compute ensemble wmse_mean as mean of top model per-plate wMSE values
    available = [m for m in top_models if m in per_plate["model"].values]
    if len(available) < 2:
        return

    ensemble_name = f"ensemble_top{len(available)}_qnu_ref_mean"
    ensemble_rows = []
    for plate_id, grp in per_plate[per_plate["model"].isin(available)].groupby("plate_id"):
        wmse_vals = grp["wmse_mean"].to_numpy()
        ensemble_rows.append({
            "plate_id": plate_id,
            "model": ensemble_name,
            "n_compounds": grp["n_compounds"].iloc[0],
            "wmse_mean": float(wmse_vals.mean()),
        })

    if not ensemble_rows:
        return

    ensemble_df       = pd.DataFrame(ensemble_rows)
    ensemble_summary  = pd.DataFrame([{
        "model":       ensemble_name,
        "n_plates":    len(ensemble_df),
        "n_compounds": int(ensemble_df["n_compounds"].sum()),
        "wmse_mean":   float(ensemble_df["wmse_mean"].mean()),
    }])

    # Append to existing CSVs
    updated_summary   = pd.concat([summary, ensemble_summary], ignore_index=True)\
                          .sort_values("wmse_mean").reset_index(drop=True)
    updated_per_plate = pd.concat([per_plate, ensemble_df], ignore_index=True)\
                          .sort_values(["model", "plate_id"]).reset_index(drop=True)

    updated_summary.to_csv(summary_path, index=False)
    updated_per_plate.to_csv(per_plate_path, index=False)
    print(f"\nEnsemble {ensemble_name}: wmse_mean = {ensemble_summary['wmse_mean'].iloc[0]:.6f}", flush=True)


if __name__ == "__main__":
    _main_with_ensemble()
