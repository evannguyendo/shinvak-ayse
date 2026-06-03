#!/usr/bin/env bash
#
# Run the COMPLETE dataset for BOTH baseline and edited inference, back to back.
#
# Smoke tests have already been run and passed for all models (2 videos each,
# both modes), and those results are already saved — so this script skips the
# smoke gate and goes straight to the full run, continuing from where smoke left off.
#
# FULLY RESUMABLE: every response is written to responses.json immediately, and
# completed (status=ok) rows are skipped on restart. If the key runs out, the box
# sleeps, or you Ctrl-C, just run this script again to continue where it stopped.
# Failed rows (api_error/timeout) are retried; option order is seed-shuffled and
# deterministic per row, so it's identical across re-runs and across models.
#
#   ./run_full_dataset.sh                      # run everything (default seed 42)
#   SEED=123 ./run_full_dataset.sh             # different shuffle seed
#   PYTHON=/path/to/python ./run_full_dataset.sh
#
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
SEED="${SEED:-42}"
CONCURRENCY="${CONCURRENCY:-5}"

echo "============================================================"
echo " COMPLETE-DATASET RUN  (seed=$SEED)"
echo "   baseline -> results/baseline_results/responses.json      (75 rows, originals from HF)"
echo "   edited   -> results/complete_edit_results/responses.json (72 rows, final_spliced.mp4)"
echo "   Resumable: re-run this script to continue after any interruption."
echo "============================================================"

echo ""
echo "==> [1/3] BASELINE — all rows (original videos downloaded from HF)"
"$PYTHON" run_conflict_inference.py --mode baseline --out-dir results/baseline_results --seed "$SEED" --concurrency "$CONCURRENCY"

echo ""
echo "==> [2/3] EDITED — all rows that have final_spliced.mp4"
"$PYTHON" run_conflict_inference.py --mode edited --out-dir results/complete_edit_results --seed "$SEED" --concurrency "$CONCURRENCY"

echo ""
echo "==> [3/3] EVALUATE — baseline_correct% and edited(none-of-these)% per model"
"$PYTHON" evaluate_results.py

echo ""
echo "============================================================"
echo " DONE."
echo "   Baseline:   results/baseline_results/responses.json"
echo "   Edited:     results/complete_edit_results/responses.json"
echo "   Evaluation: results/evaluation_summary.json"
echo "============================================================"
