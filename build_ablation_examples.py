#!/usr/bin/env python3
"""
build_ablation_examples.py — emit an inference-ready examples file for the frame-count ablation.

For every (sample, fraction) it writes one record mirroring hf_examples.jsonl, but with
`conflict_path` pointing at the spliced variant and a `variant`/`fraction` tag. Run frame_ablation.py
first. Output: ablation_outputs/ablation_examples.jsonl  (72 samples x 5 variants = 360 records).
"""
import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
HF_EX = os.path.join(ROOT, "hf_examples.jsonl")
MAN = os.path.join(ROOT, "ablation_outputs", "ablation_manifest.json")
OUT = os.path.join(ROOT, "ablation_outputs", "ablation_examples.jsonl")


def relativize(p):
    # hf_examples.jsonl stores absolute paths from the original author's machine; make repo-relative.
    if not p:
        return p
    i = p.find("hf_conflict_outputs/")
    return p[i:] if i != -1 else p


def main():
    hf = {}
    with open(HF_EX) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                hf[r["video_id"]] = r
    man = json.load(open(MAN))

    n = 0
    with open(OUT, "w") as out:
        for sid, meta in man["samples"].items():
            base = hf.get(sid)
            if base is None:
                continue
            for label, v in meta["variants"].items():
                rec = {
                    "video_id": sid,
                    "variant": f"splice_{label}",
                    "fraction_label": label,
                    "fraction": round(v["edited_frames"] / v["segment_frames"], 4),
                    "edited_frames": v["edited_frames"],
                    "segment_frames": v["segment_frames"],
                    "conflict_type": base.get("conflict_type"),
                    "control_path": relativize(base.get("control_path")),
                    "conflict_path": v["path"],
                    "edited_object": base.get("edited_object"),
                    "before_edit": base.get("before_edit"),
                    "after_edit": base.get("after_edit"),
                    "questions": base.get("questions", []),
                }
                out.write(json.dumps(rec) + "\n")
                n += 1
    print(f"Wrote {n} records -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
