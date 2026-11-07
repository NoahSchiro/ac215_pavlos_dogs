#!/usr/bin/env bash
# One-time setup of the GCP project for training and serving. Safe to re-run.
#
#   PROJECT=my-project deploy/setup-gcp.sh
#
# Creates:
#   - the Artifact Registry repo both images live in (gaze-model, gaze-ui)
#   - a PUBLIC-READ bucket for published models, with CORS so browsers can
#     fetch them. Only models go here. Never make the data bucket public:
#     GazeCapture's license forbids redistributing it.
#   - write access to that bucket for the service account Vertex AI runs as,
#     so the pipeline's publish step can write to it
set -euo pipefail
: "${PROJECT:?set PROJECT to your GCP project id}"
REGION="${REGION:-us-east1}"
REPO="${REPO:-gaze}"
MODELS_BUCKET="${MODELS_BUCKET:-$PROJECT-gaze-models}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "enabling APIs"
gcloud services enable --project "$PROJECT" \
  run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  aiplatform.googleapis.com logging.googleapis.com monitoring.googleapis.com

if ! gcloud artifacts repositories describe "$REPO" --project "$PROJECT" --location "$REGION" >/dev/null 2>&1; then
  echo "creating Artifact Registry repo $REPO in $REGION"
  gcloud artifacts repositories create "$REPO" --project "$PROJECT" \
    --location "$REGION" --repository-format docker
fi

if ! gcloud storage buckets describe "gs://$MODELS_BUCKET" --project "$PROJECT" >/dev/null 2>&1; then
  echo "creating models bucket gs://$MODELS_BUCKET in $REGION"
  gcloud storage buckets create "gs://$MODELS_BUCKET" --project "$PROJECT" \
    --location "$REGION" --uniform-bucket-level-access
fi
echo "making gs://$MODELS_BUCKET publicly readable, with CORS for browsers"
gcloud storage buckets add-iam-policy-binding "gs://$MODELS_BUCKET" \
  --member allUsers --role roles/storage.objectViewer >/dev/null
gcloud storage buckets update "gs://$MODELS_BUCKET" --cors-file "$HERE/gcs-cors.json"

# Vertex AI pipelines run as the Compute Engine default service account
# unless submit.py is given another.
SA="${PIPELINE_SA:-$(gcloud projects describe "$PROJECT" --format 'value(projectNumber)')-compute@developer.gserviceaccount.com}"
echo "letting $SA write models"
gcloud storage buckets add-iam-policy-binding "gs://$MODELS_BUCKET" \
  --member "serviceAccount:$SA" --role roles/storage.objectAdmin >/dev/null

cat <<EOF

done. For the rest of the deploy:
  export PROJECT=$PROJECT REGION=$REGION MODELS_BUCKET=$MODELS_BUCKET
EOF
