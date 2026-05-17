#!/usr/bin/env bash
# Dry-run then full sam3.py pipeline on data/control.mp4
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
fi

PYTHON="${PYTHON:-python}"
SAM3="$ROOT/sam3.py"
VIDEO="${VIDEO:-data/control.mp4}"
QUESTION="${QUESTION:-What color is the pen?}"
WRONG="${WRONG:-blue}"
OBJECT="${OBJECT:-pen}"

# Always two separate output trees (never share --output-dir).
DRY_OUT="${DRY_OUT:-test_sam3_dryrun}"
FULL_OUT="${FULL_OUT:-test_sam3_full}"

if [[ "$DRY_OUT" == "$FULL_OUT" ]]; then
  echo "Error: dry-run and full-run output folders must differ (got: $DRY_OUT)"
  exit 1
fi

echo "Dry-run output:  $ROOT/$DRY_OUT/"
echo "Full-run output: $ROOT/$FULL_OUT/"
echo ""

run_sam3() {
  "$PYTHON" "$SAM3" \
    --video "$VIDEO" \
    --question "$QUESTION" \
    --wrong-option "$WRONG" \
    --object-prompt "$OBJECT" \
    --interval-pick middle \
    --max-segments 1 \
    "$@"
}

echo "=== sam3.py: dry run → $DRY_OUT ==="
run_sam3 --dry-run --output-dir "$DRY_OUT"
echo ""
echo "Dry run OK. Manifest: $ROOT/$DRY_OUT/pipeline_manifest.json"
echo ""

if [[ ! -f "$ROOT/.env" ]]; then
  echo "Error: .env not found. Copy .env.example and set OPENROUTER_API_KEY and FAL_KEY."
  exit 1
fi

echo "=== sam3.py: full run → $FULL_OUT (SAM3 + Kling; may take several minutes) ==="
echo "    (sam3.py loads keys from .env automatically)"
run_sam3 --output-dir "$FULL_OUT"
echo ""
echo "Full run OK."
echo "  Spliced video: $ROOT/$FULL_OUT/final_spliced.mp4"
echo "  Manifest:      $ROOT/$FULL_OUT/pipeline_manifest.json"
