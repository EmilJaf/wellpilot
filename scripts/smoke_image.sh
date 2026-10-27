#!/bin/sh
# Smoke-test the production image locally before a deploy. No API key goes in, so no
# tokens are spent: starting a run must answer 503, everything else must work.
#   docker build -t wellpilot:smoke . && scripts/smoke_image.sh
set -eu
IMAGE=${1:-wellpilot:smoke}
RUN=${FEATURED_RUN:-fcfb4188de3b}
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
check "no key: a run is refused 503" sh -c "[ \$(curl -s -o /dev/null -w '%{http_code}' -X POST -H 'content-type: application/json' -d '{\"question\":\"Which well produced most?\"}' $URL/api/runs) = 503 ]"
echo "memory: $(docker stats --no-stream --format '{{.MemUsage}}' "$NAME")"
check "stops cleanly on SIGTERM"     sh -c "docker stop -t 10 $NAME && docker logs $NAME 2>&1 | grep -q 'Application shutdown complete'"

exit $fail
