"""
Per-gene additive mean-shift batch correction using shared positive controls.

The 4 controls present in all 3 batches (Staurosporine, Brefeldin-A,
Trichostatin-A, Rigosertib) provide a "same biology, different batch" signal.
Any expression difference between batches for these compounds is pure batch
effect; we remove it with a per-gene additive shift.

Reference batch: tvc-qnu-012 (largest, ~10 k compounds).
All other batches are shifted to match the reference control means.
"""

import pandas as pd
import numpy as np

CONTROL_INCHIKEYS: dict[str, str] = {
    "HKSZLNNOFSGOKW-FYTWVXJKSA-N": "Staurosporine",
    "KQNZDYYTLMIZCT-KQPMLPITSA-N": "Brefeldin-A",
    "RTKIYFITIVXBLE-QEQCGCAPSA-N": "Trichostatin-A",
    "OWBFCJROIKNMGD-BQYQJAHWSA-N": "Rigosertib",
}

REFERENCE_BATCH = "tvc-qnu-012"


def _control_means(
    expr_long: pd.DataFrame,  # columns: compound, gene_id, expression
    chem: pd.DataFrame,        # columns include: compound, inchi_key
) -> pd.Series | None:
    """
    Per-gene mean expression of the shared positive controls in one batch.
    Returns a Series indexed by gene_id, or None if no controls found.
    """
    ctrl_ids = set(
        chem.loc[chem["inchi_key"].isin(CONTROL_INCHIKEYS), "compound"]
    )
    if not ctrl_ids:
        return None
    ctrl_expr = expr_long[expr_long["compound"].isin(ctrl_ids)]
    return ctrl_expr.groupby("gene_id")["expression"].mean()


def apply_batch_correction(
    batch_data: list[tuple[str, pd.DataFrame, pd.DataFrame]],
) -> list[tuple[str, pd.DataFrame]]:
    """
    Correct expression DataFrames so shared controls align across batches.

    Parameters
    ----------
    batch_data : list of (job_id, expr_long, chem)
        expr_long — long-format DataFrame with columns [compound, gene_id, expression]
        chem      — chemistry DataFrame with columns [compound, inchi_key, ...]

    Returns
    -------
    list of (job_id, corrected_expr_long)
    """
    # Compute per-gene control means for every batch
    ctrl_means: dict[str, pd.Series | None] = {}
    for job_id, expr_long, chem in batch_data:
        m = _control_means(expr_long, chem)
        if m is None:
            print(f"  WARNING [{job_id}]: no shared controls found — skipping correction")
        else:
            print(f"  [{job_id}] controls found: {len(m)} genes")
        ctrl_means[job_id] = m

    ref_means = ctrl_means.get(REFERENCE_BATCH)
    if ref_means is None:
        print(f"  WARNING: reference batch {REFERENCE_BATCH!r} has no controls — "
              "returning uncorrected data")
        return [(jid, expr) for jid, expr, _ in batch_data]

    corrected: list[tuple[str, pd.DataFrame]] = []
    for job_id, expr_long, _chem in batch_data:
        if job_id == REFERENCE_BATCH or ctrl_means.get(job_id) is None:
            corrected.append((job_id, expr_long))
            continue

        batch_means = ctrl_means[job_id]
        shared_genes = ref_means.index.intersection(batch_means.index)
        shift = (ref_means.loc[shared_genes] - batch_means.loc[shared_genes])

        # Map gene_id → shift for genes that need correction
        shift_map = shift.to_dict()
        expr_c = expr_long.copy()
        mask = expr_c["gene_id"].isin(shift_map)
        expr_c.loc[mask, "expression"] = (
            expr_c.loc[mask, "expression"]
            + expr_c.loc[mask, "gene_id"].map(shift_map)
        )

        abs_shift = shift.abs()
        print(f"  [{job_id}] shift applied — "
              f"mean={abs_shift.mean():.4f}  "
              f"p95={abs_shift.quantile(0.95):.4f}  "
              f"max={abs_shift.max():.4f}  "
              f"({len(shared_genes)} genes)")
        corrected.append((job_id, expr_c))

    return corrected
