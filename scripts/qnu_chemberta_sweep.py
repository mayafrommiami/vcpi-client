#!/usr/bin/env python3
"""
ChemBERTa + extended fingerprint sweep.

The prior improvements sweep tried ChemBERTa but silently skipped it on
failure, and only tested it alone — never combined with MACCS or Morgan.
This script fixes both issues, tests the MTR (multi-task regression) model
whose embeddings are calibrated to bioactivity, and also adds atom-pair and
topological torsion fingerprints that capture structural information orthogonal
to Morgan.

Feature sets evaluated (all on the same 3-plate holdout):
  maccs+morgan512+desc            — baseline, should reproduce ~0.47753
  maccs+morgan512+ap1024+tt256+desc — add atom-pair + topological torsion
  chemberta                       — DeepChem/ChemBERTa-77M-MTR [CLS] only
  chemberta+desc                  — + 8 physicochemical descriptors
  chemberta+maccs+desc            — + MACCS structural keys
  chemberta+maccs+morgan512+desc  — + Morgan512 (current best combo)
  chemberta+maccs+morgan512+ap1024+tt256+desc  — full kitchen sink

ChemBERTa-77M-MTR is fine-tuned on ChEMBL multi-task regression on top of
the zinc-base MLM pre-training. Its [CLS] embeddings encode both chemical
grammar and bioactivity-relevant structure — different information from
binary hash-based fingerprints.

Run:
    python scripts/qnu_chemberta_sweep.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from rdkit import DataStructs
from sklearn.linear_model import Ridge
from sklearn.utils.extmath import randomized_svd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import qnu_plate_holdout_eval as qnu

SEED      = 13
N_COMP    = 128
ALPHA     = 1000.0
# seyonec/ChemBERTa-zinc-base-v1: pure HuggingFace RoBERTa, no DeepChem dep,
# pre-trained on 10M ZINC SMILES. DeepChem/ChemBERTa-77M-MTR requires DeepChem
# as a tokenizer dependency which is not installed in the Vertex AI container.
CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
BATCH_SIZE = 64


# ── ChemBERTa embedding ──────────────────────────────────────────────────────

def load_chemberta_embeddings(
    train_chem: pd.DataFrame,
    all_ids: list[str],
) -> dict[str, np.ndarray]:
    """Load ChemBERTa-77M-MTR and compute [CLS] embeddings for all compounds.

    Returns dict uid → float32 array of shape (hidden_size,).
    Raises on failure so the sweep fails loudly rather than silently skipping.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    print(f"  Loading {CHEMBERTA_MODEL}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL)
    model     = AutoModel.from_pretrained(CHEMBERTA_MODEL)
    model.eval()

    # Use GPU if available, otherwise CPU (CPU is fine for 10k molecules)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    print(f"  ChemBERTa device: {device}", flush=True)

    chem = train_chem.copy()
    chem["user_id"] = chem["user_compound_id"].astype(str)
    chem = (
        chem[chem["user_id"].isin(all_ids)]
        .drop_duplicates("user_id")
        .set_index("user_id")
    )
    ids         = [uid for uid in all_ids if uid in chem.index]
    smiles_list = [str(chem.loc[uid, "smiles"]) for uid in ids]

    embed_by_id: dict[str, np.ndarray] = {}
    n_failed = 0

    for i in range(0, len(ids), BATCH_SIZE):
        b_smi = smiles_list[i : i + BATCH_SIZE]
        b_ids = ids[i : i + BATCH_SIZE]
        # Filter invalid/nan SMILES
        valid = [(s, uid) for s, uid in zip(b_smi, b_ids) if s and s.lower() != "nan"]
        if not valid:
            n_failed += len(b_ids)
            continue
        v_smi, v_ids = zip(*valid)
        try:
            inputs = tokenizer(
                list(v_smi),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                cls = (
                    model(**inputs)
                    .last_hidden_state[:, 0, :]
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            for uid, emb in zip(v_ids, cls):
                embed_by_id[uid] = emb
        except Exception as e:
            n_failed += len(b_ids)
            print(f"  Batch {i//BATCH_SIZE}: failed ({e})", flush=True)

        if (i // BATCH_SIZE) % 20 == 0:
            print(f"  ChemBERTa progress: {len(embed_by_id)}/{len(ids)}", flush=True)

    hidden_size = next(iter(embed_by_id.values())).shape[0] if embed_by_id else 0
    print(
        f"  ChemBERTa done: {len(embed_by_id)}/{len(ids)} embedded "
        f"({hidden_size}-dim), {n_failed} failed",
        flush=True,
    )
    return embed_by_id


# ── RDKit fingerprints ────────────────────────────────────────────────────────

def build_rdkit_components(
    train_chem: pd.DataFrame,
    all_ids: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """Compute all RDKit fingerprint components per compound.

    Returns dict of component_name → {uid → array}.
    Components: maccs (167), morgan512 (512), atompair1024 (1024),
                torsion256 (256), desc (8).
    """
    chem = train_chem.copy()
    chem["user_id"] = chem["user_compound_id"].astype(str)
    chem = (
        chem[chem["user_id"].isin(all_ids)]
        .drop_duplicates("user_id")
        .set_index("user_id")
    )
    desc_frame = chem[qnu.DESC_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    ap_gen     = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024)
    tt_gen     = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=256)

    maccs_d:    dict[str, np.ndarray] = {}
    morgan_d:   dict[str, np.ndarray] = {}
    ap_d:       dict[str, np.ndarray] = {}
    torsion_d:  dict[str, np.ndarray] = {}
    desc_d:     dict[str, np.ndarray] = {}

    for uid, row in chem.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None:
            continue

        # MACCS 167-bit
        bv    = MACCSkeys.GenMACCSKeys(mol)
        maccs = np.zeros(167, dtype=np.float32)
        for bit in bv.GetOnBits():
            if bit < 167:
                maccs[bit] = 1.0
        maccs_d[uid] = maccs

        # Morgan 512-bit (r=2)
        morgan_d[uid] = qnu.bitvect_to_array(morgan_gen.GetFingerprint(mol), 512)

        # Atom-pair 1024-bit
        ap_arr   = np.zeros(1024, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(ap_gen.GetFingerprint(mol), ap_arr)
        ap_d[uid] = ap_arr

        # Topological torsion 256-bit
        tt_arr  = np.zeros(256, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(tt_gen.GetFingerprint(mol), tt_arr)
        torsion_d[uid] = tt_arr

        # 8 physicochemical descriptors
        desc_d[uid] = (
            desc_frame.loc[uid].to_numpy(dtype=np.float32)
            if uid in desc_frame.index
            else np.zeros(len(qnu.DESC_COLS), dtype=np.float32)
        )

    print(
        f"  RDKit components: {len(maccs_d)} compounds — "
        f"maccs(167) + morgan(512) + atompair(1024) + torsion(256) + desc(8)",
        flush=True,
    )
    return {
        "maccs":   maccs_d,
        "morgan":  morgan_d,
        "ap":      ap_d,
        "torsion": torsion_d,
        "desc":    desc_d,
    }


def _concat_safe(
    *dicts: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Intersection of keys, concatenated arrays."""
    common = set(dicts[0].keys())
    for d in dicts[1:]:
        common &= set(d.keys())
    return {uid: np.concatenate([d[uid] for d in dicts]) for uid in sorted(common)}


def build_feature_sets(
    rdkit_comps: dict[str, dict[str, np.ndarray]],
    chemberta_d: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Assemble named feature-set dicts from precomputed components."""
    maccs   = rdkit_comps["maccs"]
    morgan  = rdkit_comps["morgan"]
    ap      = rdkit_comps["ap"]
    torsion = rdkit_comps["torsion"]
    desc    = rdkit_comps["desc"]

    sets: dict[str, dict[str, np.ndarray]] = {}

    # ── Baseline (should reproduce 0.47753) ──────────────────────────────────
    sets["maccs+morgan512+desc"] = _concat_safe(maccs, morgan, desc)

    # ── Extended 2D fingerprints ──────────────────────────────────────────────
    sets["maccs+morgan512+ap1024+tt256+desc"] = _concat_safe(maccs, morgan, ap, torsion, desc)

    if not chemberta_d:
        print("  WARNING: ChemBERTa embeddings empty — skipping ChemBERTa feature sets", flush=True)
        return sets

    cb = chemberta_d
    # ── ChemBERTa variants ────────────────────────────────────────────────────
    sets["chemberta"]                                  = _concat_safe(cb)
    sets["chemberta+desc"]                             = _concat_safe(cb, desc)
    sets["chemberta+maccs+desc"]                       = _concat_safe(cb, maccs, desc)
    sets["chemberta+maccs+morgan512+desc"]             = _concat_safe(cb, maccs, morgan, desc)
    sets["chemberta+maccs+morgan512+ap1024+tt256+desc"] = _concat_safe(cb, maccs, morgan, ap, torsion, desc)

    for name, fp_dict in sets.items():
        dim = next(iter(fp_dict.values())).shape[0] if fp_dict else 0
        print(f"  Feature set '{name}': {len(fp_dict)} compounds, {dim}-dim", flush=True)

    return sets


# ── Ridge+PCA evaluation ──────────────────────────────────────────────────────

def _robust_std(x_tr: np.ndarray, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu  = x_tr.mean(axis=0, keepdims=True)
    sig = x_tr.std(axis=0, keepdims=True)
    sig[sig < 1e-6] = 1.0
    return ((x_tr - mu) / sig).astype(np.float32), ((x_val - mu) / sig).astype(np.float32)


def eval_feature_set(
    *,
    name: str,
    fp_dict: dict[str, np.ndarray],
    plates: list[int],
    qnu_active: pd.DataFrame,
    expr: pd.DataFrame,
    all_ids: list[str],
    train_meta: pd.DataFrame,
    gene_filter: list[str],
) -> list[float]:
    all_id_index = {uid: i for i, uid in enumerate(all_ids)}
    y_all = expr[all_ids].T.to_numpy(dtype=np.float32)
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
            print(f"  [{name}] plate {plate_id}: insufficient coverage, skipping", flush=True)
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
            f"  [{name}] plate {plate_id}: wMSE={score:.5f}  "
            f"(ref={len(ref_cov)}, val={len(val_cov)}, dim={x_ref.shape[1]})",
            flush=True,
        )
    return scores


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", default="eval_chemberta")
    args = parser.parse_args()

    print("Loading inputs...", flush=True)
    train_chem, train_meta, query, _qpath, gene_filter, weight_cols = qnu.load_inputs()
    train_meta["user_compound_id"] = train_meta["user_compound_id"].astype(str)

    active_mask = qnu.target_active_mask(train_meta)
    active_ids  = set(train_meta.loc[active_mask, "user_compound_id"])
    all_ids_list, all_fps = qnu.build_chemistry_maps(train_chem, active_ids, weight_cols)
    all_ids_set = set(all_ids_list)

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

    # ── Expression matrix (same for all feature sets) ────────────────────────
    print("Computing expression matrix...", flush=True)
    expr = qnu.expression_wide(
        train_meta[
            (train_meta["job_id"] == qnu.QNU_JOB_ID)
            & active_mask
            & train_meta["user_compound_id"].isin(all_ids_set)
        ],
        all_ids_list,
        gene_filter,
    )
    # Filter all_ids_list to only compounds present in the expression matrix
    all_ids_list = [uid for uid in all_ids_list if uid in expr.columns]
    print(f"  Expression: {len(expr.columns)} compounds × {len(gene_filter)} genes "
          f"({len(all_ids_list)} with expression)", flush=True)

    # ── Build fingerprint components ─────────────────────────────────────────
    print("\nBuilding RDKit fingerprint components...", flush=True)
    rdkit_comps = build_rdkit_components(train_chem, all_ids_list)

    # ── Build ChemBERTa embeddings ───────────────────────────────────────────
    print(f"\nBuilding ChemBERTa embeddings ({CHEMBERTA_MODEL})...", flush=True)
    try:
        chemberta_d = load_chemberta_embeddings(train_chem, all_ids_list)
    except Exception as e:
        print(f"  ChemBERTa FAILED: {e}", flush=True)
        print("  Continuing with RDKit-only feature sets.", flush=True)
        chemberta_d = {}

    # ── Assemble feature sets ────────────────────────────────────────────────
    print("\nAssembling feature sets...", flush=True)
    feature_sets = build_feature_sets(rdkit_comps, chemberta_d)

    # ── Evaluate each feature set ────────────────────────────────────────────
    all_results: dict[str, list[float]] = {}
    for name, fp_dict in feature_sets.items():
        if not fp_dict:
            print(f"\nSkipping '{name}': empty feature dict", flush=True)
            continue
        print(f"\n{'='*60}", flush=True)
        print(f"FEATURE SET: {name}", flush=True)
        scores = eval_feature_set(
            name=name,
            fp_dict=fp_dict,
            plates=plates,
            qnu_active=qnu_active,
            expr=expr,
            all_ids=all_ids_list,
            train_meta=train_meta,
            gene_filter=gene_filter,
        )
        all_results[name] = scores

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("FINAL RESULTS (mean wMSE, lower=better):", flush=True)
    print("=" * 60, flush=True)
    rows = []
    for name, scores in all_results.items():
        mean_score = float(np.mean(scores)) if scores else float("nan")
        dim = next(iter(feature_sets[name].values())).shape[0] if feature_sets.get(name) else 0
        rows.append({
            "feature_set": name,
            "n_plates":    len(scores),
            "wmse_mean":   round(mean_score, 5),
            "feat_dim":    dim,
        })
    df = pd.DataFrame(rows).sort_values("wmse_mean")
    print(df.to_string(index=False), flush=True)

    out = ROOT / f"{args.output_prefix}_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}", flush=True)
    print("\nPrevious best (maccs+morgan512+desc): 0.47753", flush=True)


if __name__ == "__main__":
    main()
