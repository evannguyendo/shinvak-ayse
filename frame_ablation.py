#!/usr/bin/env python3
"""
frame_ablation.py  —  Edit-magnitude (frame-count) ablation for the temporal-conflict benchmark.

For each sample in hf_conflict_outputs/<id>/, build 5 versions of the video that vary HOW MUCH
of the edited segment is spliced in:

    splice_1frame  : 1 edited frame (minimal edit)
    splice_10      : first 10% of the edited-segment frames are edited, rest original
    splice_20      : 20%
    splice_50      : 50%
    splice_100     : 100%  (== the existing final_spliced.mp4)

This is PURE frame manipulation on clips that already exist (no AI generation, no API/credits).

Geometry (all parts are 1080x720 @ 25fps, "splice space"):
    final video = p0000a_orig  +  REGION  +  ptail_orig
    REGION has L_e frames (= len of p0000b_edit). For fraction f:
        n = max(1, round(f * L_e))
        REGION[i] = edited_frame[i]            if i < n
                  = original_segment_frame[i]  otherwise   (from source_clips/seg_000_src.mp4)
    anchor = start of the segment (first-n edited frames).  [decided per docs/tasks/frame-ablation.md]

Outputs:
    ablation_outputs/<id>/splice_{1frame,10,20,50,100}.mp4
    ablation_outputs/<id>/ablation_meta.json
    ablation_outputs/ablation_manifest.json   (index over all samples)

Usage:
    python frame_ablation.py                 # all samples
    python frame_ablation.py --ids moving_attribute_00 moving_count_03
    python frame_ablation.py --limit 3       # first 3 (smoke test)
"""
import argparse, json, os, sys
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "hf_conflict_outputs")
OUT_DIR = os.path.join(ROOT, "ablation_outputs")
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

# fraction label -> fraction value (None = single frame)
FRACTIONS = [("1frame", None), ("10", 0.10), ("20", 0.20), ("50", 0.50), ("100", 1.00)]


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return frames, fps


def write_frames(path, frames, fps, size):
    w = cv2.VideoWriter(path, FOURCC, fps, size)
    for fr in frames:
        if (fr.shape[1], fr.shape[0]) != size:
            fr = cv2.resize(fr, size)
        w.write(fr)
    w.release()


def n_edited(frac, L_e):
    if frac is None:
        return 1
    return max(1, min(L_e, round(frac * L_e)))


def process_sample(sample_id):
    d = os.path.join(SRC_DIR, sample_id)
    manifest = os.path.join(d, "pipeline_manifest.json")
    prefix_p = os.path.join(d, "splice_parts", "p0000a_orig.mp4")
    edit_p = os.path.join(d, "splice_parts", "p0000b_edit.mp4")
    tail_p = os.path.join(d, "splice_parts", "ptail_orig.mp4")
    src_p = os.path.join(d, "source_clips", "seg_000_src.mp4")

    # prefix (p0000a_orig) and tail (ptail_orig) are OPTIONAL: absent when the edit
    # starts at frame 0 or runs to the end of the clip. edit + source are required.
    for p in (manifest, edit_p, src_p):
        if not os.path.exists(p):
            return {"id": sample_id, "status": "skipped", "reason": f"missing {os.path.relpath(p, ROOT)}"}

    # Multi-segment samples would have p0001b_edit etc. — this ablation assumes the single-segment
    # subset (all 75 samples here). Flag if that assumption breaks.
    if os.path.exists(os.path.join(d, "splice_parts", "p0001b_edit.mp4")):
        return {"id": sample_id, "status": "skipped", "reason": "multi-segment not supported"}

    man = json.load(open(manifest))
    fps = float(man.get("fps", 25.0))

    prefix, _ = read_frames(prefix_p) if os.path.exists(prefix_p) else ([], None)
    edit, _ = read_frames(edit_p)
    tail, _ = read_frames(tail_p) if os.path.exists(tail_p) else ([], None)
    src, _ = read_frames(src_p)
    if not (edit and src):
        return {"id": sample_id, "status": "skipped", "reason": "empty edit/source clip"}

    L_e, L_o = len(edit), len(src)
    size = (edit[0].shape[1], edit[0].shape[0])  # (W,H) of splice space, e.g. 1080x720

    out_dir = os.path.join(OUT_DIR, sample_id)
    os.makedirs(out_dir, exist_ok=True)

    variants = {}
    for label, frac in FRACTIONS:
        n = n_edited(frac, L_e)
        region = []
        for i in range(L_e):
            if i < n:
                region.append(edit[i])
            else:
                region.append(src[min(i, L_o - 1)])  # original frame, clamped (L_o≈L_e)
        frames = prefix + region + tail
        out_path = os.path.join(out_dir, f"splice_{label}.mp4")
        write_frames(out_path, frames, fps, size)
        variants[label] = {
            "path": os.path.relpath(out_path, ROOT),
            "edited_frames": n,
            "segment_frames": L_e,
            "total_frames": len(frames),
        }

    meta = {
        "id": sample_id,
        "split": man.get("segments", [{}])[0].get("derived_from_presence_interval") and sample_id.rsplit("_", 1)[0],
        "fps": fps,
        "size": list(size),
        "question": man.get("question"),
        "answer": man.get("answer"),
        "wrong_option": man.get("wrong_option"),
        "edit_interval_sec": man.get("segments", [{}])[0].get("source_interval_sec"),
        "prefix_frames": len(prefix),
        "segment_frames_edited": L_e,
        "segment_frames_source": L_o,
        "tail_frames": len(tail),
        "anchor": "start",
        "variants": variants,
        "status": "ok",
    }
    json.dump(meta, open(os.path.join(out_dir, "ablation_meta.json"), "w"), indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", help="specific sample ids")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    all_ids = sorted(
        x for x in os.listdir(SRC_DIR)
        if os.path.isdir(os.path.join(SRC_DIR, x)) and x not in (".meta_cache", "hf_videos")
    )
    ids = args.ids or all_ids
    if args.limit:
        ids = ids[: args.limit]

    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    for i, sid in enumerate(ids, 1):
        r = process_sample(sid)
        results.append(r)
        st = r.get("status")
        extra = f" ({r.get('reason')})" if st != "ok" else f"  Le={r['segment_frames_edited']} total={r['variants']['100']['total_frames']}"
        print(f"[{i}/{len(ids)}] {sid}: {st}{extra}")

    ok = [r for r in results if r.get("status") == "ok"]
    manifest = {
        "generated_by": "frame_ablation.py",
        "fractions": [lbl for lbl, _ in FRACTIONS],
        "anchor": "start",
        "n_samples": len(ok),
        "n_skipped": len(results) - len(ok),
        "samples": {r["id"]: r for r in ok},
    }
    json.dump(manifest, open(os.path.join(OUT_DIR, "ablation_manifest.json"), "w"), indent=2)
    print(f"\nDone: {len(ok)} ok, {len(results)-len(ok)} skipped. Manifest -> ablation_outputs/ablation_manifest.json")
    if len(ok) != len(results):
        print("Skipped:", [(r["id"], r.get("reason")) for r in results if r.get("status") != "ok"])


if __name__ == "__main__":
    main()
