#!/usr/bin/env python3
"""
Submit a Vertex AI Custom Training Job using the Python SDK.

Usage:
    python scripts/submit_vertex_job.py                          # Morgan FP MLP, T4 GPU, 80 epochs
    python scripts/submit_vertex_job.py --model chemberta        # ChemBERTa MLP
    python scripts/submit_vertex_job.py --gpu A100 --epochs 150
    python scripts/submit_vertex_job.py --datasets tvc-qnu-012,tvc-bhr-009
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

# Python 3.11 container — required by vcpi-prediction-contest (needs >=3.11)
CONTAINERS = {
    "gpu": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py311:latest",
    "cpu": "us-docker.pkg.dev/vertex-ai/training/pytorch-cpu.2-4.py311:latest",
}

# Extra packages for ChemBERTa model
CHEMBERTA_EXTRA = "transformers sentencepiece"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="morgan",     choices=["morgan", "chemberta"],
                        help="Which compound representation to use")
    parser.add_argument("--gpu",      default="T4",         choices=["T4", "A100", "CPU"])
    parser.add_argument("--epochs",   default=80,           type=int)
    parser.add_argument("--lr",       default=1e-3,         type=float)
    parser.add_argument("--datasets", default="tvc-qnu-012",
                        help="Comma-separated dataset IDs to train on")
    args = parser.parse_args()

    from google.cloud import aiplatform
    aiplatform.init(project=PROJECT, location=REGION, staging_bucket=f"gs://{BUCKET}")

    run_tag    = f"{args.model}-{args.gpu.lower()}"
    output_dir = f"gs://{BUCKET}/runs/{run_tag}"
    train_script = "models/train_chemberta.py" if args.model == "chemberta" else "models/train.py"

    extra_pkgs = CHEMBERTA_EXTRA if args.model == "chemberta" else ""
    container  = CONTAINERS["cpu"] if args.gpu == "CPU" else CONTAINERS["gpu"]

    # Bootstrap: clone repo → install deps → run training script
    pip_install = (
        "pip install -q "
        "git+https://github.com/virtualcell-vcpi/vcpi-client.git "
        "git+https://github.com/virtualcell-vcpi/vcpi-prediction-contest-2026.git "
        "google-cloud-storage polars pyarrow rdkit scipy scikit-learn"
    )
    if extra_pkgs:
        pip_install += f" {extra_pkgs}"

    bootstrap = " && ".join([
        f"git clone --branch {BRANCH} --single-branch {REPO_URL} /vcpi-hack",
        "cd /vcpi-hack",
        pip_install,
        f"python {train_script}",
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
            "image_uri": container,
            "command":   ["bash", "-c"],
            "args":      [bootstrap],
            "env": [
                {"name": "GCS_BUCKET",      "value": BUCKET},
                {"name": "AIP_MODEL_DIR",   "value": output_dir},
                {"name": "EPOCHS",          "value": str(args.epochs)},
                {"name": "LR",              "value": str(args.lr)},
                {"name": "TRAIN_DATASETS",  "value": args.datasets},
            ],
        },
    }

    job = aiplatform.CustomJob(
        display_name=f"vcpi-{run_tag}-e{args.epochs}",
        worker_pool_specs=[worker_spec],
    )

    print(f"Submitting Vertex AI job...")
    print(f"  Model:    {args.model}")
    print(f"  GPU:      {args.gpu}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Epochs:   {args.epochs}  LR: {args.lr}")
    print(f"  Output:   {output_dir}")
    print(f"  Container:{container}")

    job.submit()

    print(f"\nJob submitted: {job.display_name}")
    print(f"Resource:      {job.resource_name}")
    print(f"\nMonitor:")
    print(f"  Console: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT}")
    print(f"  CLI:     gcloud ai custom-jobs describe {job.resource_name.split('/')[-1]} --project={PROJECT} --region={REGION}")
    print(f"\nResults will appear at: {output_dir}/")


if __name__ == "__main__":
    main()
