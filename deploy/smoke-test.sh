#!/usr/bin/env bash
# Prove the UI tracks a face end to end, in a real browser.
#
#   deploy/smoke-test.sh                    build ui/ and test that image
#   deploy/smoke-test.sh https://<service>  test a live deployment
#
# The browser's camera is the repo's MPII test face. Needs only Docker.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
PLAYWRIGHT_IMAGE=mcr.microsoft.com/playwright:v1.52.0-noble
RUN=(docker run --rm
  -v "$ROOT/ui/e2e/smoke.mjs:/smoke/smoke.mjs:ro"
  -v "$ROOT/ui/src/vision/fixtures/face.jpg:/smoke/face.mjpeg:ro")
SMOKE='cd /tmp && cp /smoke/smoke.mjs . && npm i -s --no-audit --no-fund playwright@1.52.0 >/dev/null && node smoke.mjs "$0" /smoke/face.mjpeg'

if [[ $# -gt 0 ]]; then
  "${RUN[@]}" "$PLAYWRIGHT_IMAGE" sh -c "$SMOKE" "$1"
  exit
fi

# The bundled model must be the real file, not a git-lfs pointer.
if [[ $(wc -c <"$ROOT/ui/public/models/gazenet.onnx") -lt 1000000 ]]; then
  echo "ui/public/models/gazenet.onnx is a git-lfs pointer; run: git lfs pull" >&2
  exit 1
fi
docker build -q -t gaze-ui:smoke "$ROOT/ui" >/dev/null
docker run -d --rm --name gaze-smoke-ui -e PORT=8080 gaze-ui:smoke >/dev/null
trap 'docker stop gaze-smoke-ui >/dev/null 2>&1 || true' EXIT
# Share the UI container's network so the page is http://localhost, which the
# browser treats as a secure origin (the camera is only offered to those).
"${RUN[@]}" --network container:gaze-smoke-ui "$PLAYWRIGHT_IMAGE" sh -c "$SMOKE" http://localhost:8080/
