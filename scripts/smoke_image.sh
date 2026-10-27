#!/bin/sh
# Smoke-test the production image locally before a deploy. No API key goes in, so no
# tokens are spent: starting a run must answer 503, everything else must work.
#   docker build -t wellpilot:smoke . && scripts/smoke_image.sh
set -eu
IMAGE=${1:-wellpilot:smoke}
RUN=${FEATURED_RUN:-8aef8d548380}
NAME=wellpilot-smoke
PORT=18080
URL="http://localhost:$PORT"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" -p "$PORT:8080" --memory=1g -e FEATURED_RUNS="$RUN" "$IMAGE" >/dev/null
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 40); do curl -sf "$URL/api/health" >/dev/null 2>&1 && break; sleep 0.5; done

fail=0
check() {  # name, then a command that must succeed
  name=$1; shift
  if "$@" >/dev/null 2>&1; then echo "ok    $name"; else echo "FAIL  $name"; fail=1; fi
}
check "health: data ready, no key"   sh -c "curl -sf $URL/api/health | grep -q '\"data_ready\":true' && curl -sf $URL/api/health | grep -q '\"llm_configured\":false'"
check "the page is served"           sh -c "curl -sf $URL/ | grep -q '<title>'"
check "only the featured run listed" sh -c "curl -sf $URL/api/runs | grep -q '$RUN' && [ \$(curl -sf $URL/api/runs | grep -o '\"id\"' | wc -l) -eq 1 ]"
check "featured run replays (SSE)"   sh -c "curl -sf -m 10 $URL/api/runs/$RUN/events | grep -q '\"type\": *\"final\"'"
check "well data"                    sh -c "curl -sf $URL/api/wells | grep -q 'F-12'"
check "PUD page image"               sh -c "curl -sf -o /dev/null -w '%{content_type}' $URL/api/reports/volve_pud/pages/23/image | grep -q 'image/png'"
check "quotes located on the page"   sh -c "curl -sf $URL/api/reports/volve_pud/pages/23/quotes | grep -q 'Maksimum vannkutt'"
check "field vs the 2005 plan"       sh -c "curl -sf $URL/api/field/plan | grep -q 'plan_p50_msm3'"
check "a backtest case up close"     sh -c "curl -sf $URL/api/backtest/cases/F-12/2010-02-28 | grep -q '\"agent\": *{'"
check "the forecast film"            sh -c "curl -sf $URL/api/backtest/film/F-12 | grep -q '\"frames\"'"
check "the field's history"          sh -c "curl -sf $URL/api/field/timeline | grep -q '\"replayed\"'"
check "work on F-12 from its reports" sh -c "curl -sf $URL/api/wells/F-12/operations | grep -q '3150 m'"
check "field against design limits"  sh -c "curl -sf $URL/api/field/design | grep -q 'Maksimum vannkutt'"
check "a tool's source code"         sh -c "curl -sf $URL/api/code/wellpilot.agent.tools.detect_events@0 | grep -q 'def detect_events'"
check "no key: a run is refused 503" sh -c "[ \$(curl -s -o /dev/null -w '%{http_code}' -X POST -H 'content-type: application/json' -d '{\"question\":\"Which well produced most?\"}' $URL/api/runs) = 503 ]"
check "no key: voice is refused 503" sh -c "[ \$(curl -s -o /dev/null -w '%{http_code}' -X POST -H 'content-type: application/json' -d '{\"audio_wav_b64\":\"$(printf 'UklGRg%.0s' $(seq 1 20))\"}' $URL/api/transcribe) = 503 ]"
check "a reader's edit is checked"   sh -c "curl -sf -X POST -H 'content-type: application/json' -d '{\"findings\":[{\"index\":1,\"claim\":\"Nothing at all.\",\"confidence\":\"fact\"}]}' $URL/api/runs/$RUN/check | grep -q '\"accepted\"'"
echo "memory: $(docker stats --no-stream --format '{{.MemUsage}}' "$NAME")"
check "stops cleanly on SIGTERM"     sh -c "docker stop -t 10 $NAME && docker logs $NAME 2>&1 | grep -q 'Application shutdown complete'"

exit $fail
