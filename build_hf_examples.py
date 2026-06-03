"""
Build hf_examples.jsonl from completed hf_conflict_outputs.

Each row becomes one ExampleRecord with:
  control_path  = original HF video
  conflict_path = edited video at the chosen ablation percentage

Ablation variants (--edit-pct):
  100  →  final_spliced.mp4           (full edit, default)
  50   →  final_spliced_50pct.mp4
  30   →  final_spliced_30pct.mp4
  10   →  final_spliced_10pct.mp4
  1    →  final_spliced_1frame.mp4    (single-frame edit)

Usage:
  python build_hf_examples.py                        # 100% (full edit)
  python build_hf_examples.py --edit-pct 50          # 50% ablation
  python build_hf_examples.py --edit-pct 10 --output hf_examples_10pct.jsonl
  python build_hf_examples.py --hf-out-dir my_outputs --metadata path/to/metadata.json
"""
import json
import argparse
from pathlib import Path


def _sec_to_mmss(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = int(round(sec % 60))
    if s == 60:
        m += 1
        s = 0
    return f"{m:02d}:{s:02d}"


def _edit_timestamps_from_manifest(manifest: dict) -> list:
    """Extract the actual edit start time from the manifest segments."""
    segments = manifest.get("segments") or []
    if segments:
        interval = segments[0].get("source_interval_sec") or {}
        start = interval.get("start")
        if start is not None:
            return [_sec_to_mmss(float(start))]
    return ["00:00"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-out-dir",
        default="hf_conflict_outputs",
        help="Root dir with per-row pipeline outputs (default: hf_conflict_outputs).",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to metadata JSON. Defaults to <hf-out-dir>/.meta_cache/metadata.json.",
    )
    parser.add_argument(
        "--edit-pct",
        default="100",
        choices=["100", "50", "30", "10", "1"],
        help=(
            "Which ablation percentage of the edited video to use as conflict_path. "
            "100 = final_spliced.mp4 (default, full edit). "
            "Others require the corresponding file to exist, e.g. final_spliced_50pct.mp4."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output examples JSONL path. "
            "Defaults to hf_examples.jsonl (100%) or hf_examples_<pct>pct.jsonl for ablations."
        ),
    )
    args = parser.parse_args()

    # Map percentage → filename
    pct = args.edit_pct
    if pct == "100":
        edited_filename = "final_spliced.mp4"
    elif pct == "1":
        edited_filename = "final_spliced_1frame.mp4"
    else:
        edited_filename = f"final_spliced_{pct}pct.mp4"

    # Default output name encodes the percentage so files don't overwrite each other
    if args.output:
        default_output = args.output
    elif pct == "100":
        default_output = "hf_examples.jsonl"
    else:
        default_output = f"hf_examples_{pct}pct.jsonl"

    hf_out = Path(args.hf_out_dir)
    metadata_path = Path(args.metadata) if args.metadata else hf_out / ".meta_cache" / "metadata.json"

    if not metadata_path.exists():
        raise SystemExit(f"Metadata not found: {metadata_path}")

    meta_rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    meta_by_id = {str(r["id"]): r for r in meta_rows}

    examples = []
    skipped = []

    for row_id, row in sorted(meta_by_id.items()):
        manifest_path = hf_out / row_id / "pipeline_manifest.json"
        if not manifest_path.exists():
            skipped.append((row_id, "no manifest"))
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if manifest.get("dry_run"):
            skipped.append((row_id, "dry_run only"))
            continue

        orig_path = manifest.get("video_path", "")
        edited_path = str(hf_out / row_id / edited_filename)

        if not orig_path or not Path(orig_path).exists():
            skipped.append((row_id, f"original video missing: {orig_path}"))
            continue

        if not Path(edited_path).exists():
            skipped.append((row_id, f"{edited_filename} missing"))
            continue

        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            skipped.append((row_id, "missing question or answer"))
            continue

        # pick the wrong-answer option (first candidate that isn't the correct answer)
        candidates = row.get("candidates") or []
        wrong = next(
            (str(c) for c in candidates if str(c).strip().lower() != answer.lower()),
            answer,
        )

        examples.append({
            "video_id": row_id,
            "control_path": orig_path,
            "conflict_path": edited_path,
            "conflict_type": str(row.get("split", "unknown")),
            "edited_object": str(manifest.get("sam3_object_prompt") or row_id),
            "edit_timestamps": _edit_timestamps_from_manifest(manifest),
            "before_edit": answer,
            "after_edit": wrong,
            "questions": [{
                "question_id": "q1",
                "question_text": question,
                "expected_answer_control": answer,
                "expected_answer_conflict": wrong,
                "options": [str(c) for c in candidates],  # all answer choices passed to model
            }],
        })

    out_path = Path(default_output)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples → {out_path}")
    if skipped:
        print(f"Skipped {len(skipped)} rows:")
        for rid, reason in skipped:
            print(f"  {rid}: {reason}")


if __name__ == "__main__":
    main()
