#!/usr/bin/env bash
# Score base and tuned Muse Glimmer on one Fireworks deployment, then delete the deployment
# whatever happens: it bills $8 an hour while it runs. A watchdog deletes it at the time cap
# even if a step hangs. See docs/glimmer-sft.md.
#
# Usage: scripts/glimmer_eval.sh <account id> <deployment id> [cap in minutes, default 90]
set -euo pipefail
ACCOUNT=${1:?account id (firectl whoami)}
DEPLOYMENT=${2:?deployment id (firectl deployment list)}
CAP_MIN=${3:-90}
cd "$(dirname "$0")/../backend"

DELETED=$(mktemp -u)
delete() {
  [ -e "$DELETED" ] && return 0
  if firectl deployment delete "$DEPLOYMENT"; then
    touch "$DELETED"
    echo "deleted deployment $DEPLOYMENT"
  else
    echo "COULD NOT DELETE $DEPLOYMENT: run 'firectl deployment delete $DEPLOYMENT' and check 'firectl deployment list'"
  fi
}
(sleep $((CAP_MIN * 60)) && echo "time cap of $CAP_MIN min reached" && delete) &
WATCHDOG=$!
finish() {
  kill "$WATCHDOG" 2>/dev/null || true
  delete
  firectl deployment list || true
}
trap finish EXIT
trap 'exit 130' INT TERM

firectl load-lora wellpilot-glimmer-writer --deployment "$DEPLOYMENT"
BASE="accounts/fireworks/models/muse-glimmer-30b#accounts/$ACCOUNT/deployments/$DEPLOYMENT"
TUNED="accounts/$ACCOUNT/models/wellpilot-glimmer-writer#accounts/$ACCOUNT/deployments/$DEPLOYMENT"
score() { LLM_PROVIDER=fireworks FIREWORKS_MODEL="$1" uv run python -m wellpilot.training eval --label "$2" "${@:3}"; }

score "$BASE" glimmer-base --limit 3
score "$BASE" glimmer-base
score "$TUNED" glimmer-tuned
