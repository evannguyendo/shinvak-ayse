#!/usr/bin/env python3
"""
run_conflict_inference.py — baseline / edited video inference, resumable.

Modes
-----
  baseline : run each enabled model on the ORIGINAL video, downloaded from the
             HF dataset (metadata `video_path`). NOT the local hf_videos copies
             (hashed names) and NOT source_clips.
  edited   : run each enabled model on the local edited video
             hf_conflict_outputs/<row_id>/final_spliced.mp4.

Options for every question are taken from each row's pipeline_manifest.json:
  [ correct (manifest "answer"), wrong (manifest "wrong_option"), "none of these" ]

Resumability
------------
All responses are written to <out-dir>/responses.json (atomically) after EVERY
call, keyed by "<model_id>|<row_id>". On restart, rows already marked status=ok
are skipped; anything else (api_error from a dead key, timeout, etc.) is retried.
So if the key runs out you can just re-run the same command to continue.

Usage
-----
  python run_conflict_inference.py --mode baseline --out-dir results/baseline_results
  python run_conflict_inference.py --mode edited   --out-dir results/complete_edit_results
  python run_conflict_inference.py --mode baseline --out-dir results/baseline_results --limit 2   # smoke
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

from models_config import load_enabled_models
from run_single_video_question import (
    ask_video_question_structured,
    build_instructor_client,
    encode_video_to_data_url,
)

HF_REPO = "shivank21/mvbench-temporal-conflict-subset"
HF_BRANCH = "main"
OUT_ROOT = _PROJECT_ROOT / "hf_conflict_outputs"
META_PATH = OUT_ROOT / ".meta_cache" / "metadata.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_rows() -> dict[str, dict[str, Any]]:
    if not META_PATH.is_file():
        raise SystemExit(f"metadata not found: {META_PATH} (run the dataset pipeline first)")
    rows = json.loads(META_PATH.read_text(encoding="utf-8"))
    return {str(r["id"]): r for r in rows}


def load_manifest(row_id: str) -> Optional[dict[str, Any]]:
    p = OUT_ROOT / row_id / "pipeline_manifest.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_options(manifest: dict[str, Any], row: dict[str, Any],
                  row_id: str, seed: int) -> tuple[list[str], str, str]:
    """Return (options, correct, wrong) with options = [correct, wrong, 'none of these'],
    deterministically SHUFFLED to prevent answer-position bias.

    correct: manifest 'answer', falling back to metadata 'answer'.
    wrong:   manifest 'wrong_option', falling back to a metadata candidate != correct.

    The shuffle is seeded by (seed, row_id), so a given row's option order is fixed
    across models, across baseline/edited, and across re-runs — it only varies per row.
    Correctness is scored on the `correct`/`wrong` text, so order never affects scoring.
    """
    correct = str(manifest.get("answer", "")).strip() or str(row.get("answer", "")).strip()
    wrong = str(manifest.get("wrong_option", "")).strip()
    if not wrong:
        for c in row.get("candidates", []) or []:
            if str(c).strip().lower() != correct.lower():
                wrong = str(c).strip()
                break
    opts: list[str] = []
    for o in (correct, wrong):
        if o and o not in opts:
            opts.append(o)
    opts.append("none of these")
    random.Random(f"{seed}:{row_id}").shuffle(opts)
    return opts, correct, wrong


def _hf_url(path_in_repo: str) -> str:
    return f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_BRANCH}/{path_in_repo}"


# One lock per destination file so concurrent workers don't fetch the same video twice.
_DL_GUARD = threading.Lock()
_DL_LOCKS: dict[str, threading.Lock] = {}


def _download_lock(dest: Path) -> threading.Lock:
    with _DL_GUARD:
        return _DL_LOCKS.setdefault(str(dest), threading.Lock())


def download_baseline_video(row: dict[str, Any], cache_dir: Path) -> Path:
    """Download the original full video from HF (cached). Never touches hf_videos/.

    Concurrency-safe: a per-destination lock means the same video is fetched once
    (other workers wait, then hit the cache); a unique tmp name + atomic rename
    avoids partial/clobbered files."""
    vp = str(row.get("video_path") or "").strip()
    if not vp:
        vid = str(row.get("video") or "").strip()
        vp = f"videos/{vid}" if vid else ""
    if not vp:
        raise RuntimeError("row has no video_path/video")
    dest = cache_dir / Path(vp).name
    with _download_lock(dest):
        if dest.is_file() and dest.stat().st_size > 0:
            return dest
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + f".{uuid.uuid4().hex}.tmp")
        with requests.get(_hf_url(vp), stream=True, timeout=180) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        tmp.replace(dest)
    return dest


def edited_video_path(row_id: str) -> Path:
    return OUT_ROOT / row_id / "final_spliced.mp4"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["baseline", "edited"], required=True)
    ap.add_argument("--out-dir", required=True, help="Directory for responses.json (+ baseline video cache)")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N rows (used for the smoke test)")
    ap.add_argument("--rows", nargs="*", default=None, help="Explicit row ids (default: all, sorted)")
    ap.add_argument("--seed", type=int, default=42, help="Seed for deterministic per-row option shuffling.")
    ap.add_argument("--concurrency", type=int, default=5, help="Max concurrent API calls (default 5).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "responses.json"
    video_cache = out_dir / "_orig_videos"

    # Fails fast if OPENROUTER_API_KEY is missing.
    client = build_instructor_client()
    models = load_enabled_models()
    rows = load_rows()

    # Row selection: need a manifest (for options); edited also needs final_spliced.mp4.
    candidate_ids = args.rows if args.rows else sorted(rows.keys())
    row_ids: list[str] = []
    skipped_no_manifest, skipped_no_edit = [], []
    for rid in candidate_ids:
        if rid not in rows or load_manifest(rid) is None:
            skipped_no_manifest.append(rid)
            continue
        if args.mode == "edited" and not edited_video_path(rid).is_file():
            skipped_no_edit.append(rid)
            continue
        row_ids.append(rid)
    if args.limit is not None:
        row_ids = row_ids[: args.limit]

    if skipped_no_edit:
        print(f"  (edited) skipping {len(skipped_no_edit)} rows with no final_spliced.mp4: {skipped_no_edit}")

    # Resume: load prior results, skip those already ok.
    results: dict[str, Any] = {}
    if results_path.is_file():
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
        except Exception:
            results = {}

    def save() -> None:
        tmp = results_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(results_path)

    print(f"mode={args.mode}  seed={args.seed}  concurrency={args.concurrency}  "
          f"models={[m['id'] for m in models]}  rows={len(row_ids)}  out={results_path}")

    def run_one(m: dict[str, Any], rid: str) -> tuple[str, dict[str, Any]]:
        """Worker: resolve video + run inference for one (model, row) and return
        (key, record). Pure w.r.t. `results` — it never reads/writes the shared dict,
        so only the main thread mutates results and no lock is needed."""
        mid = m["id"]
        model = m["openrouter_model"]
        max_tokens = int(m.get("max_tokens", 4000))
        temperature = float(m.get("temperature", 0.7))
        key = f"{mid}|{rid}"
        try:
            row = rows[rid]
            manifest = load_manifest(rid)
            opts, correct, wrong = build_options(manifest, row, rid, args.seed)
            question = str(row.get("question") or manifest.get("question") or "").strip()
            rec: dict[str, Any] = {
                "model_id": mid, "model": model, "row_id": rid,
                "split": row.get("split"), "mode": args.mode,
                "question": question, "options": opts,
                "correct": correct, "wrong": wrong,
                "correct_index": opts.index(correct) if correct in opts else None,
                "shuffle_seed": args.seed, "ts": _utc_now(),
            }
            try:
                if args.mode == "baseline":
                    video = download_baseline_video(row, video_cache)
                else:
                    video = edited_video_path(rid)
                    if not video.is_file():
                        raise FileNotFoundError(f"missing {video}")
                rec["video"] = str(video)
            except Exception as e:
                rec.update({"answer": None, "confidence": None, "raw": None,
                            "status": "video_error", "error": str(e)[:2000], "ts": _utc_now()})
                return key, rec
            try:
                data_url = encode_video_to_data_url(video)
                result, raw = ask_video_question_structured(
                    client=client, model=model, video_data_url=data_url,
                    question=question, options=opts,
                    max_tokens=max_tokens, temperature=temperature,
                    include_none_option=False,  # opts already contains a shuffled "none of these"
                )
                rec.update({"answer": result.answer, "confidence": result.confidence,
                            "raw": raw, "status": "ok", "error": None})
            except Exception as e:
                err = str(e) or type(e).__name__
                nm = type(e).__name__.lower()
                low = err.lower()
                if "incompleteoutput" in nm or "max_tokens" in low or "length" in low:
                    status = "parse_error"
                elif "instructorretry" in nm or "validation" in low or "invalid json" in low:
                    status = "validation_error"
                elif "timeout" in low or "timed out" in low:
                    status = "timeout"
                else:
                    status = "api_error"
                rec.update({"answer": None, "confidence": None, "raw": None,
                            "status": status, "error": err[:2000]})
            rec["ts"] = _utc_now()
            return key, rec
        except Exception as e:  # defensive: a worker must never crash the pool
            return key, {"model_id": mid, "model": model, "row_id": rid, "mode": args.mode,
                         "status": "error", "error": str(e)[:2000], "ts": _utc_now()}

    # Work list — skip rows already completed ok (resume); run the rest concurrently.
    worklist = [(m, rid) for m in models for rid in row_ids
                if results.get(f"{m['id']}|{rid}", {}).get("status") != "ok"]
    print(f"  {len(worklist)} calls to run "
          f"({len(models)} models x {len(row_ids)} rows, minus already-ok).")

    new_count = 0
    total = len(worklist)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futures = [ex.submit(run_one, m, rid) for (m, rid) in worklist]
        for fut in concurrent.futures.as_completed(futures):
            key, rec = fut.result()
            results[key] = rec   # main thread only — safe without a lock
            save()
            new_count += 1
            mark = "✓" if rec.get("status") == "ok" else "✗"
            print(f"[{rec.get('status', ''):14s}] {mark} ({new_count}/{total}) "
                  f"{rec.get('model_id')} {rec.get('row_id')} ans={rec.get('answer')!r}")

    # Summary
    print(f"\n{'='*60}\nDone. {new_count} new responses this run. Total saved: {len(results)} -> {results_path}")
    for m in models:
        mid = m["id"]
        sub = [v for v in results.values() if v.get("model_id") == mid]
        from collections import Counter
        c = Counter(v.get("status") for v in sub)
        print(f"  {mid:24s} {dict(c)}")


if __name__ == "__main__":
    main()
