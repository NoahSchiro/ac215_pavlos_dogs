#!/usr/bin/env bash
# One time commands, if you haven't configured GCP in the terminal. You can
# just uncomment these if you want.
#   
#   # logs you in
#   gcloud auth login 
#
#   # lets you use the stuff you need to use to run this command
#   # (before you do this, you may need to set up which project you are in, which
#   # is something like: `gcloud config set project "$PROJECT_ID"`)
#   gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
#       cloudbuild.googleapis.com storage.googleapis.com
#
#   # Create the output bucket where we store model weights
#   gcloud storage buckets create "gs://$RESULTS_BUCKET" --location "$REGION"
#
#   # Create the artifact registry
#   gcloud artifacts repositories create gaze --repository-format=docker --location=us-east1
#
#   # Create a service account with the appropriate permissions
#   gcloud iam service-accounts create gaze-train
#   gcloud storage buckets add-iam-policy-binding "gs://$RESULTS_BUCKET" \
#       --member "serviceAccount:$SA_EMAIL" --role roles/storage.objectAdmin
#
# Note that if we want to use the NVIDIA L4, we don't need to request a quota for that.
# You get it by default. Note also that the Dockerfile is not configured to use
# GPU at the moment. That is issue #52.
#
# If you want wandb logging, then you need to give GCP your API key:
#   gcloud secrets create wandb-api-key --data-file=<your-api-key
#   gcloud secrets add-iam-policy-binding wandb-api-key \
#    --member "serviceAccount:$SA_EMAIL" --role roles/secretmanager.secretAccessor

set -euo pipefail

PROJECT_ID="ac215-508601"
REGION="us-east1"

# Bucket holding the preprocessed cache.
# Note this is my own bucket, not the group bucket
DATA_BUCKET="ac215-508601-gaze-data"
DATA_ROOT="/MPIIFaceGaze/"

RESULTS_BUCKET="$PROJECT_ID-gaze-results"
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/gaze/gaze-model"
JOB_NAME="gaze-train"
SA_EMAIL="gaze-train@$PROJECT_ID.iam.gserviceaccount.com"

DEVICE="cpu"
# Add ",--wandb,gaze" to the end of this if you want wandb logging
TRAIN_ARGS="--workers,8,--device,$DEVICE,--data-root,/data/$DATA_ROOT"

cd "$(dirname "$0")"

echo "Changing project to $PROJECT_ID"
gcloud config set project "$PROJECT_ID"

echo "Authenticating Docker with Artifact Registry..."
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet

echo "Building Docker image locally..."
docker build -t "$IMAGE" .

echo "Pushing container to Artifact Registry..."
docker push "$IMAGE"

gcloud run jobs deploy "$JOB_NAME" \
  --image "$IMAGE" \
  --region "$REGION" \
  --service-account "$SA_EMAIL" \
  --cpu 8 --memory 16Gi \
  --task-timeout=4h \
  --add-volume="mount-path=/data,type=cloud-storage,bucket=$DATA_BUCKET" \
  --add-volume="mount-path=/app/checkpoints,type=cloud-storage,bucket=$RESULTS_BUCKET,readonly=false" \
  --args="$TRAIN_ARGS"

echo "Running job..."
gcloud run jobs execute "$JOB_NAME" --region "$REGION" --wait

echo "checkpoints in gs://$RESULTS_BUCKET:"
gcloud storage ls "gs://$RESULTS_BUCKET/checkpoints/"
