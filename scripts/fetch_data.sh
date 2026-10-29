#!/bin/sh
# Download the processed Volve data pinned in deploy/data.env into data/processed: the
# deploy job builds the image with it, and a clone can run the app without the raw files.
#   scripts/fetch_data.sh            # refuses to replace an existing data/processed
#   scripts/fetch_data.sh --force
set -eu
cd "$(dirname "$0")/.."
. deploy/data.env
REPO=${GITHUB_REPOSITORY:-EmilJaf/wellpilot}
ASSET=processed.tar.gz

if [ -e data/processed ] && [ "${1:-}" != --force ]; then
  echo "data/processed exists; scripts/fetch_data.sh --force replaces it" >&2
  exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
# gh reaches the release while the repository is private; curl once it is public.
if command -v gh >/dev/null 2>&1 && { [ -n "${GH_TOKEN:-}" ] || gh auth status >/dev/null 2>&1; }; then
  gh release download "$DATA_RELEASE" --repo "$REPO" --pattern "$ASSET" --dir "$tmp"
else
  curl -fsSL -o "$tmp/$ASSET" "https://github.com/$REPO/releases/download/$DATA_RELEASE/$ASSET"
fi

if command -v sha256sum >/dev/null 2>&1; then
  sum=$(sha256sum "$tmp/$ASSET" | cut -d' ' -f1)
else
  sum=$(shasum -a 256 "$tmp/$ASSET" | cut -d' ' -f1)
fi
if [ "$sum" != "$DATA_SHA256" ]; then
  echo "$ASSET of $DATA_RELEASE has sha256 $sum, deploy/data.env pins $DATA_SHA256" >&2
  exit 1
fi

rm -rf data/processed
mkdir -p data
tar -xzf "$tmp/$ASSET" -C data
echo "data/processed: $DATA_RELEASE ($(du -sh data/processed | cut -f1))"
