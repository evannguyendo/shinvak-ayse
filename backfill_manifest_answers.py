#!/usr/bin/env python3
"""
Backfill the `answer` (correct option) field in each pipeline_manifest.json from
the authoritative metadata ground truth.

~Half the manifests were written with an empty `answer` (the `wrong_option` was
always recorded, but the correct answer was not). metadata.json (.meta_cache) has
the correct `answer` for every row and matches the manifest whenever both are set.

Idempotent: only fills manifests whose `answer` is empty/missing; never overwrites
a populated value. Run again safely.

  python backfill_manifest_answers.py            # apply
  python backfill_manifest_answers.py --dry-run  # report only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "hf_conflict_outputs"
META = OUT / ".meta_cache" / "metadata.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    args = ap.parse_args()

    if not META.is_file():
        raise SystemExit(f"metadata not found: {META}")
    meta = {str(r["id"]): r for r in json.loads(META.read_text(encoding="utf-8"))}

    changed, already_ok, problems = [], [], []
    for rid, row in sorted(meta.items()):
        mp = OUT / rid / "pipeline_manifest.json"
        if not mp.is_file():
            continue
        mf = json.loads(mp.read_text(encoding="utf-8"))
        cur = str(mf.get("answer", "")).strip()
        truth = str(row.get("answer", "")).strip()

        if cur:
            if cur != truth:
                problems.append(f"{rid}: manifest answer {cur!r} != metadata {truth!r} (left as-is)")
            else:
                already_ok.append(rid)
            continue
        if not truth:
            problems.append(f"{rid}: manifest answer empty AND metadata answer empty")
            continue

        mf["answer"] = truth
        if not args.dry_run:
            tmp = mp.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(mf, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(mp)
        changed.append(f"{rid} -> {truth!r}")

    verb = "Would fill" if args.dry_run else "Filled"
    print(f"{verb} {len(changed)} manifest(s); already populated: {len(already_ok)}; problems: {len(problems)}")
    for c in changed:
        print(f"  {verb.split()[0].lower()}: {c}")
    for p in problems:
        print(f"  PROBLEM: {p}")


if __name__ == "__main__":
    main()
