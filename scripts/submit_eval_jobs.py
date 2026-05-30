#!/usr/bin/env python3
"""
Submit Vertex AI jobs for the plate-holdout evaluation scripts.

Each job:
  1. Clones the repo
  2. Installs deps (rdkit, sklearn, pyarrow, vcpi-prediction-contest)
  3. Downloads training parquets from GCS via prep_eval_data.py
  4. Runs the eval script
  5. Uploads results CSVs to GCS

Usage:
    python scripts/submit_eval_jobs.py                       # submit all eval jobs
    python scripts/submit_eval_jobs.py --eval baseline       # just baseline eval
    python scripts/submit_eval_jobs.py --eval mlp            # just MLP eval
    python scripts/submit_eval_jobs.py --eval extended       # just extended sweep
    python scripts/submit_eval_jobs.py --eval improvements   # wider FP + ChemBERTa + ensemble
"""

import argparse
import os

PROJECT  = os.environ.get("GCLOUD_PROJECT", "cell-imaging-ml")
REGION   = os.environ.get("GCLOUD_REGION",  "us-central1")
BUCKET   = os.environ.get("GCS_BUCKET",     "vcpi-drugseq-2026")
REPO_URL = "https://github.com/mayafrommiami/vcpi-client.git"
BRANCH   = "hackathon"

# CPU container — eval doesn't need GPU (except MLP training, which is small enough for CPU)
CONTAINER = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"
MACHINE   = "n1-highmem-8"  # 52GB RAM — needed for 1.4GB counts + expression aggregation


EVAL_CONFIGS = {
    "baseline": {
        "display_name": "vcpi-eval-baseline",
        "script": (
            "python scripts/qnu_plate_holdout_eval.py "
            "--include-regression "
            "--target-components 64 "
            "--pls-components 16 "
            "--output-prefix eval_qnu_plate_holdout_quick_regression"
        ),
        "output_prefix": "eval_qnu_plate_holdout_quick_regression",
    },
    "extended": {
        "display_name": "vcpi-eval-extended-sweep",
        "script": (
            "python scripts/qnu_plate_extended_model_sweep.py "
            "--output-prefix eval_qnu_plate_extended_models"
        ),
        "output_prefix": "eval_qnu_plate_extended_models",
    },
    "mlp": {
        "display_name": "vcpi-eval-mlp",
        "script": (
            "python scripts/qnu_mlp_eval.py "
            "--epochs 30 --lr 1e-3 "
            "--include-ensemble "
            "--output-prefix eval_mlp"
        ),
        "output_prefix": "eval_mlp",
    },
    "improvements": {
        "display_name": "vcpi-eval-improvements",
        "script": (
            "python scripts/qnu_model_improvements_sweep.py "
            "--output-prefix eval_qnu_improvements"
        ),
        "output_prefix": "eval_qnu_improvements",
        "extra_packages": "transformers sentencepiece",
    },
    "finetuned": {
        "display_name": "vcpi-eval-finetuned-hyperparam",
        "script": (
            "python scripts/qnu_finetuned_hyperparam_sweep.py "
            "--output-prefix eval_qnu_finetuned_hyperparam"
        ),
        "output_prefix": "eval_qnu_finetuned_hyperparam",
    },
    "radius3": {
        "display_name": "vcpi-eval-radius3",
        "script": (
            "python scripts/qnu_radius3_sweep.py "
            "--output-prefix eval_qnu_radius3"
        ),
        "output_prefix": "eval_qnu_radius3",
    },
    "concat": {
        "display_name": "vcpi-eval-concat-fp",
        "script": (
            "python scripts/qnu_concat_fp_sweep.py "
            "--output-prefix eval_qnu_concat_fp"
        ),
        "output_prefix": "eval_qnu_concat_fp",
    },
}


def submit_job(name: str, config: dict) -> None:
    from google.cloud import aiplatform
    aiplatform.init(project=PROJECT, location=REGION, staging_bucket=f"gs://{BUCKET}")

    output_dir = f"gs://{BUCKET}/eval_runs/{name}"

    # Bootstrap: clone repo → install deps → download data → run eval → upload results
    pip_pin_numpy = "pip install -q numpy==1.26.4"
    extra = config.get("extra_packages", "")
    pip_install = (
        "pip install -q "
        "google-cloud-storage pyarrow pandas scikit-learn rdkit scipy"
        + (f" {extra}" if extra else "")
    )
    pip_install_contest = (
        "pip install -q --ignore-requires-python "
        "git+https://github.com/virtualcell-vcpi/vcpi-prediction-contest-2026.git"
    )

    # Upload results CSVs to GCS after eval
    upload_results = (
        f"gsutil -m cp {config['output_prefix']}*.csv {output_dir}/"
    )

    bootstrap = " && ".join([
        f"git clone --branch {BRANCH} --single-branch {REPO_URL} /vcpi-eval",
        "cd /vcpi-eval",
        pip_pin_numpy,
        pip_install,
        pip_install_contest,
        # Set up data_files directory
        "mkdir -p src/vcpi_prediction_contest/data_files",
        "cp -r $(python -c \"import vcpi_prediction_contest; import os; print(os.path.dirname(vcpi_prediction_contest.__file__))\")//data_files/. src/vcpi_prediction_contest/data_files/",
        # Download training data from GCS
        "python scripts/prep_eval_data.py",
        # Run the eval
        config["script"],
        # Upload results
        upload_results,
    ])

    worker_spec = {
        "machine_spec": {
            "machine_type": MACHINE,
            "accelerator_type": "NVIDIA_TESLA_T4",
            "accelerator_count": 1,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": CONTAINER,
            "command": ["bash", "-c"],
            "args": [bootstrap],
            "env": [
                {"name": "GCS_BUCKET", "value": BUCKET},
            ],
        },
    }

    job = aiplatform.CustomJob(
        display_name=config["display_name"],
        worker_pool_specs=[worker_spec],
    )

    print(f"\nSubmitting: {config['display_name']}")
    print(f"  Script: {config['script']}")
    print(f"  Output: {output_dir}/")
    job.submit()
    print(f"  Job: {job.resource_name}")
    return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval",
        nargs="*",
        choices=["baseline", "extended", "mlp", "improvements", "finetuned", "radius3", "concat"],
        default=["baseline", "extended", "mlp", "improvements", "finetuned", "radius3", "concat"],
        help="Which eval jobs to submit (default: all)",
    )
    args = parser.parse_args()

    jobs = []
    for name in args.eval:
        job = submit_job(name, EVAL_CONFIGS[name])
        jobs.append((name, job))

    print(f"\n{'='*60}")
    print(f"Submitted {len(jobs)} eval jobs:")
    for name, job in jobs:
        print(f"  {name}: {job.display_name}")
    print(f"\nMonitor:")
    print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT}")
    print(f"\nResults will appear at: gs://{BUCKET}/eval_runs/")


if __name__ == "__main__":
    main()
