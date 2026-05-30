# tvc-qnu-012 3-Plate Validation

This is the minimal offline validation we are using for the current
test-set assumption: the final compounds are active compounds from a
held-out plate in `tvc-qnu-012`.

## What The Eval Does

- Uses active, non-control compounds only.
- Restricts active training/reference compounds to the target dose:
  `10000 nM`, equivalent to the `10 uM` test compounds.
- Restricts validation to `tvc-qnu-012`.
- Selects 3 validation plates whose active compounds are most similar to
  `src/vcpi_prediction_contest/data_files/test_compounds.csv` by
  plate-level Morgan Tanimoto similarity.
- Holds out one selected `container_id` at a time.
- Excludes held-out compound IDs from the reference/training side.
- Scores raw `log2(CPM + 1)` expression, not DMSO deltas.
- Ranks models only by `wmse_mean`, matching the package leaderboard
  aggregate.

Current selected validation plates:

| plate_id | active compounds | median max Tanimoto to test | mean max Tanimoto to test |
| --- | ---: | ---: | ---: |
| `1839083` | 180 | 0.699 | 0.663 |
| `1839058` | 180 | 0.687 | 0.628 |
| `1839082` | 180 | 0.650 | 0.618 |

## Required Local Files

Run the main `README.md` "Getting the training data" recipe first if
these files are missing:

- `train_counts.parquet`
- `train_metadata.parquet`
- `train_chemistry.parquet`
- `weights.parquet`

The training data recipe downloads the contest condition: THP-1, 24h,
active compounds at the target dosage, plus DMSO controls. This eval then
excludes DMSO from active-compound baselines and model reference sets.

## Reproduce The Current Eval

From the repo root:

```bash
uv run --with rdkit --with scikit-learn \
  python scripts/qnu_plate_holdout_eval.py \
  --include-regression \
  --target-components 64 \
  --pls-components 16 \
  --output-prefix eval_qnu_plate_holdout_quick_regression
```

Main output:

```bash
cat eval_qnu_plate_holdout_quick_regression_summary.csv
```

The summary schema is intentionally small:

```text
model,n_plates,n_compounds,wmse_mean
```

## Best Model So Far

The best model on this validation currently is:

```text
qnu_ref_rdkit_morgan512+descriptors_target_pca128_ridge_alpha1000
```

It scores `wmse_mean = 0.478663` over the same 3 held-out
`tvc-qnu-012` plates / 540 active validation compounds.

Model definition:

- Reference/training compounds: active `tvc-qnu-012` compounds at the
  target 10 uM dose only, excluding the held-out validation plate
  compounds and excluding DMSO.
- Input features: RDKit Morgan fingerprint, radius 2, 512 bits, plus
  the 8 chemistry descriptors in `train_chemistry.parquet`.
- Target compression: fit PCA/SVD on the reference gene-expression
  matrix and keep 128 target components.
- Regressor: `sklearn.linear_model.Ridge(alpha=1000)`, fit from the
  molecular features to the 128 target-PC scores.
- Prediction: reconstruct full scored-gene expression from predicted
  target-PC scores and clip to non-negative raw `log2(CPM + 1)`
  expression.

Reproduce the extended sweep that includes this model:

```bash
uv run --with rdkit --with scikit-learn \
  python scripts/qnu_plate_extended_model_sweep.py \
  --output-prefix eval_qnu_plate_extended_models
```

Main output:

```bash
cat eval_qnu_plate_extended_models_summary.csv
```

Top current extended-sweep results:

| model | wmse_mean |
| --- | ---: |
| `qnu_ref_rdkit_morgan512+descriptors_target_pca128_ridge_alpha1000` | 0.478663 |
| `qnu_ref_rdkit_morgan512+descriptors_target_pca128_ridge_alpha100` | 0.478733 |
| `qnu_ref_rdkit_morgan512+descriptors_target_pca64_ridge_alpha1000` | 0.478864 |
| `qnu_ref_rdkit_tanimoto_knn_k100` | 0.478876 |
| `qnu_ref_rdkit_tanimoto_knn_k50` | 0.478988 |
| `rdkit_morgan512+descriptors_target_pca128_ridge_alpha100` | 0.481332 |

The extended sweep also audits molecule embeddings from
`/lustre/groups/ml01/workspace/artur.szalata/code/chem-perturbridge_lpm`.
Those LPM/external Morgan embeddings cover some reference compounds, but
they currently cover `0 / 540` selected validation compounds on this
qnu plate-holdout split, so they cannot make non-fallback predictions
for this eval setup.

## Current Scores

Current `wmse_mean` results over 3 held-out plates / 540 active compounds:

| model | wmse_mean |
| --- | ---: |
| `rdkit_morgan512+descriptors_target_pca64_ridge_alpha1000` | 0.481800 |
| `rdkit_tanimoto_knn_k50` | 0.482248 |
| `rdkit_tanimoto_knn_k100` | 0.482811 |
| `rdkit_morgan512+descriptors_target_pca64_pls16` | 0.483726 |
| `rdkit_tanimoto_knn_k20` | 0.485758 |
| `qnu_active_mean_without_plate` | 0.488814 |
| `rdkit_tanimoto_knn_k5` | 0.506388 |
| `all_active_mean_without_plate` | 0.506464 |
| `dmso_control_mean` | 0.549848 |
| `bhr_active_mean_without_plate` | 0.567208 |
| `kdl_active_mean_without_plate` | 0.654191 |

The mean perturbed baselines are:

| baseline | definition | wmse_mean |
| --- | --- | ---: |
| `qnu_active_mean_without_plate` | per-gene mean over active `tvc-qnu-012` training compounds at 10 uM only, excluding the held-out plate compounds and excluding DMSO | 0.488814 |
| `all_active_mean_without_plate` | per-gene mean over active 10 uM training compounds from `tvc-bhr-009`, `tvc-kdl-010`, and `tvc-qnu-012`, excluding the held-out compounds and excluding DMSO | 0.506464 |
| `bhr_active_mean_without_plate` | per-gene mean over active `tvc-bhr-009` training compounds at 10 uM only, excluding any held-out compounds and excluding DMSO | 0.567208 |
| `kdl_active_mean_without_plate` | per-gene mean over active `tvc-kdl-010` training compounds at 10 uM only, excluding any held-out compounds and excluding DMSO | 0.654191 |

## Outputs

The eval writes:

- `eval_qnu_plate_holdout_quick_regression_summary.csv`
- `eval_qnu_plate_holdout_quick_regression_per_plate.csv`
- `eval_qnu_plate_holdout_quick_regression_per_compound.csv`
- `eval_qnu_plate_holdout_quick_regression_split.csv`
- `eval_qnu_plate_holdout_quick_regression_plate_similarity.csv`

The extended sweep writes:

- `eval_qnu_plate_extended_models_summary.csv`
- `eval_qnu_plate_extended_models_per_plate.csv`
- `eval_qnu_plate_extended_models_per_compound.csv`
- `eval_qnu_plate_extended_models_split.csv`
- `eval_qnu_plate_extended_models_plate_similarity.csv`
- `eval_qnu_plate_extended_models_coverage.csv`

Use `*_summary.csv` for model ranking. The `*_per_plate.csv` and
`*_per_compound.csv` files are diagnostics.
