#!/usr/bin/env bash
# Run the full Hugging Face temporal-conflict dataset through sam3.py (one subprocess per row).
#
# Prerequisites:
#   - .venv with: pip install requests python-dotenv
#   - .env with OPENROUTER_API_KEY (and FAL_KEY for non-dry-run)
#
# Usage:
#   ./run_dataset.sh smoke          # 2 rows, dry-run, verbose (no Kling cost)
#   ./run_dataset.sh dry-run        # all 75 rows, dry-run
#   ./run_dataset.sh full           # all 75 rows, real SAM3 + Kling
#   ./run_dataset.sh resume         # full run, skip rows already finished
#   ./run_dataset.sh split moving_attribute dry-run
#
# Environment overrides (optional):
#   OUT_DIR=hf_conflict_outputs
#   DATASET=shivank21/mvbench-temporal-conflict-subset
#   LIMIT=5
#   SPLIT_FILTER=object_interaction

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
fi

PYTHON="${PYTHON:-python}"
RUNNER="$ROOT/run_hf_dataset.py"

OUT_DIR="${OUT_DIR:-hf_conflict_outputs}"
DATASET="${DATASET:-shivank21/mvbench-temporal-conflict-subset}"

if [[ ! -f "$RUNNER" ]]; then
  echo "Error: run_hf_dataset.py not found at $RUNNER"
  exit 1
fi

if [[ ! -f "$ROOT/.env" ]]; then
  echo "Warning: .env not found. Copy .env.example and set OPENROUTER_API_KEY (and FAL_KEY for full runs)."
fi

run_hf() {
  "$PYTHON" "$RUNNER" \
    --dataset "$DATASET" \
    --out-dir "$OUT_DIR" \
    "$@"
}

MODE="${1:-help}"
shift || true

case "$MODE" in
  smoke)
    echo "=== Smoke test: 2 rows, dry-run, verbose ==="
    run_hf --limit 2 --dry-run --verbose "$@"
    ;;

  dry-run)
    echo "=== Full dataset dry-run (no SAM3/Kling API; Gemini prompts still run) ==="
    run_hf --dry-run "$@"
    ;;

  full)
    echo "=== Full dataset run (SAM3 + Kling + splice) ==="
    run_hf "$@"
    ;;

  resume)
    echo "=== Resume full run (skip rows with pipeline_manifest.json) ==="
    run_hf --skip-existing "$@"
    ;;

  split)
    SPLIT_NAME="${1:-}"
    if [[ -z "$SPLIT_NAME" ]]; then
      echo "Usage: $0 split <moving_attribute|moving_count|object_existence|moving_direction|object_interaction> [dry-run]"
      exit 1
    fi
    shift || true
    EXTRA=()
    if [[ "${1:-}" == "dry-run" ]]; then
      EXTRA=(--dry-run)
      shift || true
    fi
    echo "=== Split only: $SPLIT_NAME ==="
    run_hf --split-filter "$SPLIT_NAME" "${EXTRA[@]}" "$@"
    ;;

  help|-h|--help)
    cat <<EOF
Run Hugging Face dataset rows through sam3.py (automated batch).

  $0 smoke              2 rows, --dry-run --verbose
  $0 dry-run            all rows, --dry-run
  $0 full               all rows, production (needs FAL_KEY)
  $0 resume             full + --skip-existing
  $0 split <name>       one split only (append 'dry-run' for mock run)

Examples:
  $0 smoke
  $0 dry-run
  $0 full
  LIMIT=5 $0 dry-run
  OUT_DIR=my_outputs $0 resume

Outputs:
  \$OUT_DIR/<row_id>/final_spliced.mp4
  \$OUT_DIR/<row_id>/splice_parts/
  \$OUT_DIR/run_progress.json

Python entry (same as this script):
  python run_hf_dataset.py --out-dir hf_conflict_outputs
EOF
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo "Run: $0 help"
    exit 1
    ;;
esac

echo ""
echo "Done. Progress: $ROOT/$OUT_DIR/run_progress.json"
