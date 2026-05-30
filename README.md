# vcpi-hack — VCPI Drug-seq Prediction Contest 2026

Predict transcriptomic responses (log₂CPM+1, 12,995 genes) to 1,064 unseen drug compounds in THP-1 cells.

## Quickstart

```bash
# 1. Set credentials
export TVC_TOKEN="your_token_here"
export GCS_BUCKET="vcpi-drugseq-2026"

# 2. Set up GCS (first time only)
bash scripts/setup_gcs.sh

# 3. Download training data and upload to GCS (~30 min)
/opt/homebrew/bin/python3.13 scripts/upload_data.py

# 4. Find shared controls/compounds between -009 and -012
/opt/homebrew/bin/python3.13 scripts/find_shared_anchors.py

# 5. Characterize batch effects
/opt/homebrew/bin/python3.13 analysis/batch_effects.py
```

## Data

Three training releases accessible via `vcpi-client` (requires `TVC_TOKEN`):

| Dataset | Compounds | Samples |
|---------|-----------|---------|
| tvc-bhr-009 | 2,277 | 4,554 |
| tvc-kdl-010 | 1,493 | 2,986 |
| tvc-qnu-012 | 10,256 | 20,520 |

All filtered to: THP-1 cells, 24h timepoint, 10 µM (10,000 nM).

GCS bucket: `gs://vcpi-drugseq-2026/`

## Evaluation

**Weighted MSE (wMSE)** — lower is better. Genes differentially expressed by a compound are weighted more.

Baseline to beat: **0.507** (per-gene training mean)

## GPU Runtime

Open `notebooks/colab_setup.ipynb` in [Google Colab](https://colab.research.google.com):
- Set secret `TVC_TOKEN` in Colab sidebar (🔑)
- Select T4 GPU runtime
- Run cells top to bottom

### Alternative: Compute Engine VM

```bash
gcloud compute instances create vcpi-gpu-vm \
    --project=virtual-cell-hack \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --maintenance-policy=TERMINATE \
    --image-family=common-cu124-debian-12-py310 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --metadata="install-nvidia-driver=True" \
    --scopes=cloud-platform

# SSH in
gcloud compute ssh vcpi-gpu-vm --zone=us-central1-a

# Stop when not in use (~$0.35/hr while running)
gcloud compute instances stop vcpi-gpu-vm --zone=us-central1-a
```
