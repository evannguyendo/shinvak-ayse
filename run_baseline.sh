#!/usr/bin/env bash
#
# Baseline inference — every enabled model (models.json) on the ORIGINAL videos
# downloaded from the HF dataset, with 3-way options [correct, wrong, none]
# shuffled per-row with a fixed seed (to prevent answer-position bias).
#
#   1. Smoke test: each model on the first 2 videos.
#   2. Gate: abort if any model produced zero OK responses in the smoke.
#   3. Full run over all rows.
#
# Resumable: responses are written to results/baseline_results/responses.json
# after every call. If the key runs out (or you Ctrl-C), just re-run this script —
# rows already marked status=ok are skipped, failures are retried. The shuffle is
# seeded by (SEED, row_id), so option order is identical across re-runs and models.
#
#   PYTHON=/path/to/python SEED=123 ./run_baseline.sh   # override interpreter / seed
#
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
SEED="${SEED:-42}"
MODE="baseline"
OUT_DIR="results/baseline_results"

echo "==> [1/3] Smoke test: each model on 2 videos  (mode=$MODE, seed=$SEED)"
"$PYTHON" run_conflict_inference.py --mode "$MODE" --out-dir "$OUT_DIR" --seed "$SEED" --limit 2

echo ""
echo "==> [2/3] Smoke gate: verify each model produced at least one OK response"
"$PYTHON" - "$OUT_DIR/responses.json" <<'PY'
import json, sys
from pathlib import Path
from models_config import load_enabled_models
p = Path(sys.argv[1])
res = json.loads(p.read_text()) if p.is_file() else {}
bad = []
for m in load_enabled_models():
    mid = m["id"]
    oks = sum(1 for v in res.values() if v.get("model_id") == mid and v.get("status") == "ok")
    print(f"   {mid}: {oks} OK in smoke")
    if oks == 0:
        bad.append(mid)
if bad:
    print(f"\nSMOKE FAILED for {bad}. Fix (key/credits/model id/video support) before the full run.")
    sys.exit(1)
print("Smoke passed for all models.")
PY

echo ""
echo "==> [3/3] Full baseline run (resumes; skips smoke rows already OK)"
"$PYTHON" run_conflict_inference.py --mode "$MODE" --out-dir "$OUT_DIR" --seed "$SEED"

echo ""
echo "==> Done. Results: $OUT_DIR/responses.json"
