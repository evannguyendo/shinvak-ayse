#!/usr/bin/env python3
"""
Score the baseline & edited inference results and save an accuracy summary JSON.

Definitions (per the benchmark design):
  baseline accuracy = fraction of OK responses where the model's answer == the
                      correct option (it got the original video right).
  edited accuracy   = fraction of OK responses where the model's answer ==
                      "none of these" (it detected the temporal conflict / edit
                      instead of committing to the correct OR the wrong option).

Only `status == "ok"` responses are scored; api_error/timeout/etc. are excluded
from the denominator and reported separately. Matching is case/whitespace/period
insensitive, and "none of the above" is treated the same as "none of these".

  python evaluate_results.py
  python evaluate_results.py --baseline results/baseline_results/responses.json \
                             --edited   results/complete_edit_results/responses.json \
                             --out      results/evaluation_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _norm(s: object) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower()).strip().rstrip(".").strip()


def _is_none(ans: object) -> bool:
    a = _norm(ans)
    return a in {"none", "none of these", "none of the above"} or a.startswith("none of")


def _category(r: dict) -> str:
    """Which of the 3 options the answer landed on: 'correct', 'wrong', 'none', or 'other'."""
    ans = r.get("answer")
    if _is_none(ans):
        return "none"
    if _norm(ans) == _norm(r.get("correct")):
        return "correct"
    if _norm(ans) == _norm(r.get("wrong")):
        return "wrong"
    return "other"


def _breakdown(records: list[dict]) -> dict:
    """Per-model answer distribution across the 3 categories (+ 'other'), OK responses only."""
    out: dict[str, dict] = defaultdict(lambda: {"correct": 0, "wrong": 0, "none": 0, "other": 0, "total": 0})
    for r in records:
        if r.get("status") != "ok":
            continue
        o = out[r.get("model_id", "?")]
        o[_category(r)] += 1
        o["total"] += 1
    return out


def _with_pcts(c: dict) -> dict:
    t = c["total"] or 1
    d = {k: {"count": c[k], "pct": round(100 * c[k] / t, 2)} for k in ("correct", "wrong", "none", "other")}
    d["total"] = c["total"]
    return d


def _load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")).values())


def _score(records: list[dict], mode: str) -> dict:
    """mode='baseline' -> correct when answer==correct; mode='edited' -> correct when answer is 'none'."""
    per_model: dict[str, dict] = defaultdict(
        lambda: {"correct": 0, "total": 0, "unmatched": 0, "excluded": 0,
                 "by_split": defaultdict(lambda: {"correct": 0, "total": 0})}
    )
    for r in records:
        mid = r.get("model_id", "?")
        pm = per_model[mid]
        if r.get("status") != "ok":
            pm["excluded"] += 1
            continue
        ans = r.get("answer")
        split = r.get("split") or "?"
        pm["total"] += 1
        pm["by_split"][split]["total"] += 1

        if mode == "baseline":
            correct = _norm(ans) == _norm(r.get("correct"))
        else:  # edited
            correct = _is_none(ans)

        if _norm(ans) not in {_norm(o) for o in (r.get("options") or [])}:
            pm["unmatched"] += 1
        if correct:
            pm["correct"] += 1
            pm["by_split"][split]["correct"] += 1

    out = {}
    for mid, pm in per_model.items():
        tot, cor = pm["total"], pm["correct"]
        out[mid] = {
            "correct": cor,
            "total": tot,
            "accuracy_pct": round(100 * cor / tot, 2) if tot else None,
            "unmatched": pm["unmatched"],
            "excluded_non_ok": pm["excluded"],
            "by_split": {s: {"correct": v["correct"], "total": v["total"],
                             "accuracy_pct": round(100 * v["correct"] / v["total"], 2) if v["total"] else None}
                         for s, v in sorted(pm["by_split"].items())},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(ROOT / "results/baseline_results/responses.json"))
    ap.add_argument("--edited", default=str(ROOT / "results/complete_edit_results/responses.json"))
    ap.add_argument("--out", default=str(ROOT / "results/evaluation_summary.json"))
    args = ap.parse_args()

    base = _load(Path(args.baseline))
    edit = _load(Path(args.edited))
    base_scores = _score(base, "baseline")
    edit_scores = _score(edit, "edited")
    base_bd = _breakdown(base)
    edit_bd = _breakdown(edit)
    _ZERO = {"correct": 0, "wrong": 0, "none": 0, "other": 0, "total": 0}

    models = sorted(set(base_scores) | set(edit_scores))
    summary = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "definitions": {
            "baseline_correct": "answer == correct option",
            "edited_correct": "answer == 'none of these' (temporal conflict detected)",
            "answer_breakdown": ("distribution across the 3 options (count + pct of OK responses). "
                                 "For EDITED: none = conflict detected, wrong = answered the edited object "
                                 "(fooled by the edit), correct = answered the original answer (missed the edit)."),
        },
        "sources": {"baseline": args.baseline, "edited": args.edited},
        "per_model": {},
    }
    # pooled aggregates
    agg = {"baseline_correct": 0, "baseline_total": 0, "edited_correct": 0, "edited_total": 0}
    for m in models:
        b = base_scores.get(m, {})
        e = edit_scores.get(m, {})
        summary["per_model"][m] = {
            "baseline_correct_pct": b.get("accuracy_pct"),
            "baseline_correct": b.get("correct", 0),
            "baseline_total": b.get("total", 0),
            "edited_correct_pct": e.get("accuracy_pct"),
            "edited_correct": e.get("correct", 0),
            "edited_total": e.get("total", 0),
            "baseline_unmatched": b.get("unmatched", 0),
            "edited_unmatched": e.get("unmatched", 0),
            "baseline_excluded_non_ok": b.get("excluded_non_ok", 0),
            "edited_excluded_non_ok": e.get("excluded_non_ok", 0),
            "baseline_by_split": b.get("by_split", {}),
            "edited_by_split": e.get("by_split", {}),
            "answer_breakdown": {
                "baseline": _with_pcts(base_bd.get(m, _ZERO)),
                "edited": _with_pcts(edit_bd.get(m, _ZERO)),
            },
        }
        agg["baseline_correct"] += b.get("correct", 0)
        agg["baseline_total"] += b.get("total", 0)
        agg["edited_correct"] += e.get("correct", 0)
        agg["edited_total"] += e.get("total", 0)

    summary["overall"] = {
        "baseline_correct_pct": round(100 * agg["baseline_correct"] / agg["baseline_total"], 2) if agg["baseline_total"] else None,
        "edited_correct_pct": round(100 * agg["edited_correct"] / agg["edited_total"], 2) if agg["edited_total"] else None,
        **agg,
    }

    def _pool(bd: dict) -> dict:
        tot = {"correct": 0, "wrong": 0, "none": 0, "other": 0, "total": 0}
        for c in bd.values():
            for k in tot:
                tot[k] += c[k]
        return _with_pcts(tot)
    summary["overall"]["answer_breakdown"] = {"baseline": _pool(base_bd), "edited": _pool(edit_bd)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Console table
    print(f"\n{'Model':30s} {'baseline_correct%':>20s} {'edited(none)_correct%':>24s}")
    print("-" * 76)
    for m in models:
        pm = summary["per_model"][m]
        b = f"{pm['baseline_correct_pct']}% ({pm['baseline_correct']}/{pm['baseline_total']})" if pm["baseline_total"] else "n/a"
        e = f"{pm['edited_correct_pct']}% ({pm['edited_correct']}/{pm['edited_total']})" if pm["edited_total"] else "n/a"
        print(f"{m:30s} {b:>20s} {e:>24s}")
    ov = summary["overall"]
    print("-" * 76)
    print(f"{'OVERALL (pooled)':30s} "
          f"{str(ov['baseline_correct_pct'])+'%':>20s} {str(ov['edited_correct_pct'])+'%':>24s}")

    print("\n3-category answer breakdown (% of OK responses):")
    for mode_key in ("baseline", "edited"):
        note = "   (none=conflict detected, wrong=edited object, correct=original)" if mode_key == "edited" else ""
        print(f"  [{mode_key}]{note}")
        for m in models:
            bd = summary["per_model"][m]["answer_breakdown"][mode_key]
            line = (f"     {m:24s} correct={bd['correct']['pct']:5.1f}%  "
                    f"wrong={bd['wrong']['pct']:5.1f}%  none={bd['none']['pct']:5.1f}%")
            if bd["other"]["count"]:
                line += f"  other={bd['other']['pct']:.1f}%"
            print(line)

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
