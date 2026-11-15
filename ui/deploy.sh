#!/usr/bin/env bash
# Build the gaze maze UI and deploy it to Cloud Run.
#
# Every deploy is a new Cloud Run revision, so rolling back is
#   gcloud run services update-traffic --to-revisions <rev>=100
#
# Prerequisites: setup-gcp.sh once per project, docker running,
# gcloud login.
#
# Usage: ./deploy.sh

set -euo pipefail
cd "$(dirname "$0")"

PROJECT_ID="ac215-508601"
REGION="us-east1"
REPO="gaze"
SERVICE="gaze-maze"
MODEL_URL="/models/gazenet.onnx"
SA_UI="gaze-ui@$PROJECT_ID.iam.gserviceaccount.com"
VERSION="$(git rev-parse --short HEAD)"
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/gaze-ui:$VERSION"

# Package and push the serving image.
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet
docker build --platform linux/amd64 --target serve -t "$IMAGE" .
docker push "$IMAGE"

# Deploy
gcloud run deploy "$SERVICE" \
  --project "$PROJECT_ID" --region "$REGION" --image "$IMAGE" \
  --service-account "$SA_UI" \
  --port 8080 \
  --set-env-vars "MODEL_URL=$MODEL_URL" \
  --allow-unauthenticated \
  --cpu 1 --memory 256Mi --concurrency 80 \
  --min-instances 0 --max-instances 3 \
  --quiet

URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --format 'value(status.url)')"
echo "deployed $VERSION at $URL"
curl -fsS "$URL/health" >/dev/null && echo "health: ok"
