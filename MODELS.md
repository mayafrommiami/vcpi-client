# VCPI Hackathon — Model Code Overview

## models/data_loader.py
Handles all data loading and preprocessing. Pulls counts, metadata, and chemistry parquet files from the public GCS bucket (`gs://vcpi-drugseq-2026`), runs `counts_to_expression()` to normalize raw UMI counts to log2(CPM+1), and pivots everything into a genes × compounds matrix restricted to the 12,995 scored genes. Computes Morgan fingerprints (ECFP4, radius=2, 2048 bits) for each compound and packages everything into a `DrugExpressionDataset` — a PyTorch Dataset where each sample is one compound, the input is its fingerprint, and the target is the residual expression vector (deviation from per-gene training mean). Which datasets to load is controlled by the `TRAIN_DATASETS` environment variable (default: `tvc-qnu-012`).

## models/batch_correct.py
Applies a per-gene additive mean-shift correction to align expression data across plate batches before merging them for training. The 4 positive controls shared across all plates — Staurosporine, Brefeldin-A, Trichostatin-A, and Rigosertib — provide a "same biology, different batch" anchor. For each non-reference batch, the per-gene mean expression of these controls is computed, and a per-gene shift is added to bring it in line with the reference batch (`tvc-qnu-012`). This is a simple but principled correction that only requires a few shared compounds.

## models/mlp.py
Defines the neural network architecture and loss function used by both models. `FingerprintMLP` is a 4-layer MLP with BatchNorm, GELU activations, Dropout(0.3), and a skip connection from the input to the penultimate layer — it works for any input dimension so it handles both 2048-dim Morgan fingerprints and 384-dim ChemBERTa embeddings. The loss is a weighted MSE (`WeightedMSELoss`) where each gene's weight is proportional to its variance across training compounds, which roughly approximates the contest's Mejia wMSE metric. Also includes `mc_predict()` for MC Dropout uncertainty estimates.

## models/chemberta_loader.py
Pre-computes frozen ChemBERTa embeddings (seyonec/ChemBERTa-zinc-base-v1, 384-dim CLS token) for all training compounds and caches them to GCS so they only need to be computed once. `build_chemberta_dataset()` runs the full pipeline — loads expression data, computes or loads cached embeddings, aligns compounds — and returns a `DrugExpressionDataset` with embeddings in place of fingerprints. The idea is that ChemBERTa's pre-trained chemical language model may capture structural features that binary fingerprints miss, without needing end-to-end fine-tuning.

## models/train.py
Self-contained training script for the Morgan fingerprint MLP, designed to run on Vertex AI or locally. Loads data from GCS, trains the MLP with AdamW + cosine LR schedule for a configurable number of epochs (default 80), tracks the best validation loss, then generates predictions for all test compounds and writes `submission.parquet`, `model.pt`, and `history.parquet` back to GCS. All key hyperparameters (epochs, LR, batch size, dataset IDs, output path) are overridable via environment variables.

## models/train_chemberta.py
Same training loop as `train.py` but uses ChemBERTa embeddings instead of Morgan fingerprints. At training time it calls `build_chemberta_dataset()` to load the cached 384-dim embeddings; at inference time it loads the ChemBERTa tokenizer and model directly to embed the test compounds. Falls back to the per-gene training mean for any compound with an invalid or missing SMILES string.

## scripts/submit_vertex_job.py
Submits Vertex AI Custom Training Jobs via the Python SDK. Takes `--model morgan|chemberta`, `--gpu T4|A100|CPU`, `--epochs`, and `--datasets` flags. The bootstrap command clones the `hackathon` branch, installs dependencies (including vcpi-prediction-contest with `--ignore-requires-python` since no Python 3.11 PyTorch GPU container exists on Vertex AI), and runs the appropriate training script. Outputs land at `gs://vcpi-drugseq-2026/runs/{model}-{gpu}/`.
