#!/usr/bin/env bash
# Publish a trained model so a UI deploy can point at it.
#
#   MODELS_BUCKET=my-models deploy/publish-model.sh model/checkpoints/20261107-120000
#
# Writes the run's best.onnx to gs://$MODELS_BUCKET/models/<run>/gazenet.onnx
# with its metrics beside it, and prints the MODEL_URL to deploy with. The
# Vertex pipeline's publish step writes the same layout, so a model trained
# either way deploys the same way.
#
# Versions are immutable: an existing one is never overwritten, which is what
# makes it safe to cache them forever and to roll back to one.
set -euo pipefail
RUN_DIR="${1:?usage: publish-model.sh <training run directory>}"
: "${MODELS_BUCKET:?set MODELS_BUCKET}"
VERSION="$(basename "$RUN_DIR")"
DEST="gs://$MODELS_BUCKET/models/$VERSION"

[[ -f "$RUN_DIR/best.onnx" ]] || { echo "no best.onnx in $RUN_DIR" >&2; exit 1; }
if gcloud storage ls "$DEST/gazenet.onnx" >/dev/null 2>&1; then
  echo "$DEST already exists; versions are immutable" >&2
  exit 1
fi

gcloud storage cp --cache-control "public, max-age=31536000, immutable" \
  "$RUN_DIR/best.onnx" "$DEST/gazenet.onnx"
if [[ -f "$RUN_DIR/best_metrics.json" ]]; then
  gcloud storage cp "$RUN_DIR/best_metrics.json" "$DEST/metrics.json"
fi

echo "MODEL_URL=https://storage.googleapis.com/$MODELS_BUCKET/models/$VERSION/gazenet.onnx"
