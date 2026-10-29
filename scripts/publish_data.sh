#!/bin/sh
# Publish data/processed as a release asset of this repository and pin it in deploy/data.env.
# Run it after an ingest change, then commit deploy/data.env and push: the deploy job builds
# the image with the pinned data. The asset goes public with the repository, so it carries
# the licence notice (deploy/DATA_NOTICE.md) and is scanned for secrets first.
#   scripts/publish_data.sh
set -eu
cd "$(dirname "$0")/.."
REPO=${GITHUB_REPOSITORY:-EmilJaf/wellpilot}
TAG=data-$(date -u +%Y%m%d-%H%M)

[ -f data/processed/production_daily.parquet ] || { echo "no data/processed: run the ingest first" >&2; exit 1; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir "$tmp/stage"
cp -R data/processed "$tmp/stage/processed"
cp deploy/DATA_NOTICE.md "$tmp/stage/processed/NOTICE.md"
find "$tmp/stage" -name .DS_Store -delete
gitleaks dir "$tmp/stage" -c .gitleaks.toml --redact --no-banner
# No macOS metadata: Linux tar warns on it, and ._ files would land in the image.
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$tmp/processed.tar.gz" -C "$tmp/stage" processed
sum=$(shasum -a 256 "$tmp/processed.tar.gz" | cut -d' ' -f1)

gh release create "$TAG" "$tmp/processed.tar.gz" --repo "$REPO" --latest=false \
  --title "Processed Volve data, $TAG" --notes-file deploy/DATA_NOTICE.md
cat > deploy/data.env <<EOF
# The processed Volve data the image is built with: a release asset of this repository,
# made by scripts/publish_data.sh and downloaded by scripts/fetch_data.sh.
DATA_RELEASE=$TAG
DATA_SHA256=$sum
EOF
echo "Published $TAG ($(du -h "$tmp/processed.tar.gz" | cut -f1)); commit deploy/data.env and push to deploy it."
