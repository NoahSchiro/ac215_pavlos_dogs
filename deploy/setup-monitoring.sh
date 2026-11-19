#!/usr/bin/env bash
# Create (or update) the log-based metrics behind production monitoring, and
# the model-load alert. Safe to re-run.
#
#   PROJECT=my-project deploy/setup-monitoring.sh
#   PROJECT=my-project ALERT_CHANNEL=projects/.../notificationChannels/123 deploy/setup-monitoring.sh
#
# The metrics read the telemetry lines nginx writes (ui/nginx/default.conf.template)
# and appear in Cloud Monitoring as logging.googleapis.com/user/<name>.
set -euo pipefail
: "${PROJECT:?set PROJECT}"
HERE="$(cd "$(dirname "$0")" && pwd)"

for config in "$HERE"/monitoring/*.yaml; do
  name="$(basename "$config" .yaml)"
  if gcloud logging metrics describe "$name" --project "$PROJECT" >/dev/null 2>&1; then
    gcloud logging metrics update "$name" --project "$PROJECT" --config-from-file "$config"
  else
    gcloud logging metrics create "$name" --project "$PROJECT" --config-from-file "$config"
  fi
done

POLICY="$HERE/monitoring/alert_model_load_errors.json"
TITLE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["displayName"])' "$POLICY")"
if [[ -n "$(gcloud alpha monitoring policies list --project "$PROJECT" --filter "displayName=\"$TITLE\"" --format 'value(name)')" ]]; then
  echo "alert policy already exists: $TITLE"
else
  gcloud alpha monitoring policies create --project "$PROJECT" --policy-from-file "$POLICY" \
    ${ALERT_CHANNEL:+--notification-channels "$ALERT_CHANNEL"}
fi
[[ -n "${ALERT_CHANNEL:-}" ]] || echo "note: no ALERT_CHANNEL given, so the alert will fire without notifying anyone"
