#!/usr/bin/env bash
# One-time setup of the GCP project for serving the web UI. Safe to re-run.
#
# Creates:
#   - the Artifact Registry repo the UI image lives in
#   - a PUBLIC-READ models bucket with CORS so the website can fetch models
#     from it, writable by the training pipeline's service account.
set -euo pipefail
cd "$(dirname "$0")"

PROJECT_ID="ac215-508601"
REGION="us-east1"
REPO="gaze"
MODELS_BUCKET="$PROJECT_ID-gaze-models"
SA_UI="gaze-ui@$PROJECT_ID.iam.gserviceaccount.com"
SA_TRAIN="gaze-train@$PROJECT_ID.iam.gserviceaccount.com"

for SA in "$SA_UI" "$SA_TRAIN"; do
  if ! gcloud iam service-accounts describe "$SA" --project "$PROJECT_ID" >/dev/null 2>&1; then
    echo "creating service account $SA"
    gcloud iam service-accounts create "${SA%%@*}" --project "$PROJECT_ID" \
      --display-name "AC215 ${SA%%@*}" >/dev/null
  fi
done

echo "enabling APIs"
gcloud services enable --project "$PROJECT_ID" \
  run.googleapis.com artifactregistry.googleapis.com

if ! gcloud artifacts repositories describe "$REPO" --project "$PROJECT_ID" --location "$REGION" >/dev/null 2>&1; then
  echo "creating Artifact Registry repo $REPO in $REGION"
  gcloud artifacts repositories create "$REPO" --project "$PROJECT_ID" \
    --location "$REGION" --repository-format docker
fi

if ! gcloud storage buckets describe "gs://$MODELS_BUCKET" --project "$PROJECT_ID" >/dev/null 2>&1; then
  echo "creating models bucket gs://$MODELS_BUCKET in $REGION"
  gcloud storage buckets create "gs://$MODELS_BUCKET" --project "$PROJECT_ID" \
    --location "$REGION" --uniform-bucket-level-access
fi

echo "making gs://$MODELS_BUCKET publicly readable, with CORS for browsers"
gcloud storage buckets add-iam-policy-binding "gs://$MODELS_BUCKET" \
  --member allUsers --role roles/storage.objectViewer >/dev/null

CORS="$(mktemp)"
trap 'rm -f "$CORS"' EXIT
cat > "$CORS" <<'EOF'
[
  {
    "origin": ["*"],
    "method": ["GET", "HEAD"],
    "responseHeader": ["Content-Type", "Content-Length", "ETag"],
    "maxAgeSeconds": 3600
  }
]
EOF
gcloud storage buckets update "gs://$MODELS_BUCKET" --cors-file "$CORS"

echo "letting $SA_TRAIN write models"
gcloud storage buckets add-iam-policy-binding "gs://$MODELS_BUCKET" \
  --member "serviceAccount:$SA_TRAIN" --role roles/storage.objectAdmin >/dev/null

echo
echo "done. deploy the site:"
echo "  ui/deploy.sh"
