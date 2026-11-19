#!/usr/bin/env bash
# Build the UI image and deploy it to Cloud Run.
#
#   PROJECT=my-project MODEL_URL=https://storage.googleapis.com/... deploy/deploy-ui.sh
#
# MODEL_URL is what deploy/publish-model.sh (or the Vertex pipeline) printed.
# Every deploy is a new Cloud Run revision, so rolling back a bad model or
# build is `gcloud run services update-traffic --to-revisions <rev>=100`.
set -euo pipefail
: "${PROJECT:?set PROJECT}"
: "${MODEL_URL:?set MODEL_URL to a published model (deploy/publish-model.sh prints it)}"
REGION="${REGION:-us-east1}"
REPO="${REPO:-gaze}"
SERVICE="${SERVICE:-gaze-maze}"
ROOT="$(git rev-parse --show-toplevel)"
VERSION="${VERSION:-$(git -C "$ROOT" rev-parse --short HEAD)}"
IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO/gaze-ui:$VERSION"

# Cloud Build builds linux/amd64, which Cloud Run requires. A local build on
# an Apple Silicon Mac would produce arm64 unless told --platform linux/amd64.
gcloud builds submit "$ROOT/ui" --project "$PROJECT" --tag "$IMAGE"

# Cloud Run sets PORT=8080, which the nginx template listens on. The model is
# fetched by the browser straight from Cloud Storage, so this service needs no
# access to it: it only serves static files, telemetry and /healthz.
gcloud run deploy "$SERVICE" \
  --project "$PROJECT" --region "$REGION" --image "$IMAGE" \
  --port 8080 \
  --set-env-vars "MODEL_URL=$MODEL_URL,APP_VERSION=$VERSION,TELEMETRY=true" \
  --allow-unauthenticated \
  --cpu 1 --memory 256Mi --concurrency 80 \
  --min-instances 0 --max-instances 3

URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format 'value(status.url)')"
echo "deployed $VERSION at $URL"
echo "check it end to end: deploy/smoke-test.sh $URL"
