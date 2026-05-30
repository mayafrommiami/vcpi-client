#!/usr/bin/env python3
"""
Submit a Vertex AI Custom Training Job using the Python SDK.

Usage:
    python scripts/submit_vertex_job.py          # T4 GPU, 80 epochs
    python scripts/submit_vertex_job.py --gpu A100 --epochs 150
"""

import argparse
import os

PROJECT  = os.environ.get("GCLOUD_PROJECT", "cell-imaging-ml")
REGION   = os.environ.get("GCLOUD_REGION",  "us-central1")
BUCKET   = os.environ.get("GCS_BUCKET",     "vcpi-drugseq-2026")
REPO_URL = "https://github.com/mayafrommiami/vcpi-client.git"
BRANCH   = "hackathon"

MACHINE_CONFIGS = {
    "T4":   dict(machine_type="n1-standard-8",  accelerator_type="NVIDIA_TESLA_T4",   accelerator_count=1),
    "A100": dict(machine_type="a2-highgpu-1g",  accelerator_type="NVIDIA_TESLA_A100", accelerator_count=1),
    "CPU":  dict(machine_type="n1-standard-8"),
}

# Pre-built Google PyTorch container (PyTorch 2.3, Python 3.10, CUDA 12.4)
CONTAINER = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3.py310:latest"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",    default="T4",  choices=["T4", "A100", "CPU"])
    parser.add_argument("--epochs", default=80,    type=int)
    parser.add_argument("--lr",     default=1e-3,  type=float)
    args = parser.parse_args()

    from google.cloud import aiplatform
    aiplatform.init(project=PROJECT, location=REGION, staging_bucket=f"gs://{BUCKET}")

    output_dir = f"gs://{BUCKET}/runs/vertex-{args.gpu.lower()}"

    # Bootstrap command: clone repo → install deps → train
    bootstrap = " && ".join([
        f"git clone --branch {BRANCH} --single-branch {REPO_URL} /vcpi-hack",
        "cd /vcpi-hack",
        "pip install -q "
        "  git+https://github.com/virtualcell-vcpi/vcpi-client.git "
        "  git+https://github.com/virtualcell-vcpi/vcpi-prediction-contest-2026.git "
        "  google-cloud-storage polars pyarrow rdkit scipy scikit-learn",
        "python models/train.py",
    ])

    machine_cfg = MACHINE_CONFIGS[args.gpu]
    worker_spec = {
        "machine_spec": {
            "machine_type": machine_cfg["machine_type"],
            **({
                "accelerator_type": machine_cfg["accelerator_type"],
                "accelerator_count": machine_cfg["accelerator_count"],
            } if "accelerator_type" in machine_cfg else {}),
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": CONTAINER,
            "command":   ["bash", "-c"],
            "args":      [bootstrap],
            "env": [
                {"name": "GCS_BUCKET",     "value": BUCKET},
                {"name": "AIP_MODEL_DIR",  "value": output_dir},
                {"name": "EPOCHS",         "value": str(args.epochs)},
                {"name": "LR",             "value": str(args.lr)},
            ],
        },
    }

    job = aiplatform.CustomJob(
        display_name=f"vcpi-mlp-{args.gpu.lower()}-e{args.epochs}",
        worker_pool_specs=[worker_spec],
    )

    print(f"Submitting Vertex AI job...")
    print(f"  GPU:      {args.gpu}")
    print(f"  Epochs:   {args.epochs}  LR: {args.lr}")
    print(f"  Output:   {output_dir}")

    job.submit()

    print(f"\nJob submitted: {job.display_name}")
    print(f"Resource:      {job.resource_name}")
    print(f"\nMonitor:")
    print(f"  Console: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT}")
    print(f"  CLI:     gcloud ai custom-jobs describe {job.resource_name.split('/')[-1]} --project={PROJECT} --region={REGION}")
    print(f"\nResults will appear at: {output_dir}/")


if __name__ == "__main__":
    main()
