#!/usr/bin/env bash
set -euo pipefail

PROJECT="${GCLOUD_PROJECT:-virtual-cell-hack}"
BUCKET="${GCS_BUCKET:-vcpi-drugseq-2026}"
REGION="${GCS_REGION:-us-central1}"

echo "==> Creating GCP project: ${PROJECT}"
gcloud projects create "${PROJECT}" --name="VirtualCell" || echo "Project may already exist, continuing..."

echo "==> Setting default project"
gcloud config set project "${PROJECT}"

echo "==> Enabling billing (manual step required if new project)"
echo "    If this fails, link billing at: https://console.cloud.google.com/billing"
echo "    Billing account:"
gcloud billing accounts list 2>/dev/null || echo "    (cannot list billing accounts)"

echo "==> Enabling Cloud Storage API"
gcloud services enable storage.googleapis.com --project="${PROJECT}"

echo "==> Creating GCS bucket: gs://${BUCKET}"
gsutil mb -p "${PROJECT}" -l "${REGION}" -b on "gs://${BUCKET}/" || echo "Bucket may already exist, continuing..."

echo "==> Creating directory structure"
gsutil cp /dev/null "gs://${BUCKET}/data/tvc-bhr-009/.keep"
gsutil cp /dev/null "gs://${BUCKET}/data/tvc-kdl-010/.keep"
gsutil cp /dev/null "gs://${BUCKET}/data/tvc-qnu-012/.keep"

echo "==> Done! Bucket contents:"
gsutil ls "gs://${BUCKET}/data/"
