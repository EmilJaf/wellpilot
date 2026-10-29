#!/bin/sh
# Check the live site after a deploy: it runs the given commit with its data and the model
# key, lists exactly fly.toml's featured runs, and replays each of them. No tokens are spent.
#   scripts/smoke_live.sh https://wellpilot.fly.dev [commit]
set -eu
cd "$(dirname "$0")/.."
URL=${1:-https://wellpilot.fly.dev}
COMMIT=${2:-}
FEATURED=$(sed -n 's/^ *FEATURED_RUNS *= *"\([^"]*\)".*/\1/p' fly.toml)

health() {  # one value of the health endpoint, as Python prints it
  curl -sf -m 10 "$URL/api/health" | python3 -c "import json, sys; print(json.load(sys.stdin).get('$1'))"
}
# fly deploy returns once the health checks pass; allow the proxy a moment more.
for _ in $(seq 1 12); do
  [ -z "$COMMIT" ] || [ "$(health commit 2>/dev/null || true)" = "$COMMIT" ] && break
  sleep 5
done

runs_commit() { [ "$(health commit)" = "$COMMIT" ]; }
ready() { [ "$(health data_ready)" = True ] && [ "$(health llm_configured)" = True ]; }
page() { curl -sf -m 10 "$URL/" | grep -q '<title>'; }
listed() {
  curl -sf -m 10 "$URL/api/runs" | python3 -c "
import json, sys
listed = sorted(run['id'] for run in json.load(sys.stdin))
sys.exit(listed != sorted('$FEATURED'.split(',')))"
}
replays() { curl -sf -m 20 "$URL/api/runs/$1/events" | grep -q '"type": *"final"'; }

fail=0
check() {  # name, then a command that must succeed
  name=$1; shift
  if "$@" >/dev/null 2>&1; then echo "ok    $name"; else echo "FAIL  $name"; fail=1; fi
}
[ -z "$COMMIT" ] || check "runs commit $(echo "$COMMIT" | cut -c1-7)" runs_commit
check "data ready, model key set" ready
check "the page is served" page
check "lists fly.toml's featured runs" listed
for run in $(echo "$FEATURED" | tr ',' ' '); do
  check "replays $run" replays "$run"
done
exit $fail
