# tvc-qnu-012 3-Plate Validation

This is the minimal offline validation we are using for the current
test-set assumption: the final compounds are active compounds from a
held-out plate in `tvc-qnu-012`.

## What The Eval Does

- Uses active, non-control compounds only.
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
| `qnu_active_mean_without_plate` | per-gene mean over active `tvc-qnu-012` training compounds only, excluding the held-out plate compounds and excluding DMSO | 0.488814 |
| `all_active_mean_without_plate` | per-gene mean over active training compounds from `tvc-bhr-009`, `tvc-kdl-010`, and `tvc-qnu-012`, excluding the held-out compounds and excluding DMSO | 0.506464 |
| `bhr_active_mean_without_plate` | per-gene mean over active `tvc-bhr-009` training compounds only, excluding any held-out compounds and excluding DMSO | 0.567208 |
| `kdl_active_mean_without_plate` | per-gene mean over active `tvc-kdl-010` training compounds only, excluding any held-out compounds and excluding DMSO | 0.654191 |

## Outputs

The eval writes:

- `eval_qnu_plate_holdout_quick_regression_summary.csv`
- `eval_qnu_plate_holdout_quick_regression_per_plate.csv`
- `eval_qnu_plate_holdout_quick_regression_per_compound.csv`
- `eval_qnu_plate_holdout_quick_regression_split.csv`
- `eval_qnu_plate_holdout_quick_regression_plate_similarity.csv`

Use `*_summary.csv` for model ranking. The `*_per_plate.csv` and
`*_per_compound.csv` files are diagnostics.
