#!/usr/bin/env bash
# Upload business_entity_resolution/src as the private Kaggle dataset "ber-code".
# First call creates it, later calls add a version. Usage: kaggle/sync_code.sh "message"
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
KAGGLE="$ROOT/.venv/bin/kaggle"
USER=peeyushprashant
STAGE="$ROOT/work/kaggle_code"
rm -rf "$STAGE" && mkdir -p "$STAGE/ber"
cp "$ROOT"/business_entity_resolution/src/ber/*.py "$STAGE/ber/"
cat > "$STAGE/dataset-metadata.json" <<EOF
{"title": "ber-code", "id": "$USER/ber-code", "licenses": [{"name": "other"}]}
EOF
if "$KAGGLE" datasets status "$USER/ber-code" >/dev/null 2>&1; then
  "$KAGGLE" datasets version -p "$STAGE" -m "${1:-sync}" --dir-mode zip
else
  "$KAGGLE" datasets create -p "$STAGE" --dir-mode zip
fi
