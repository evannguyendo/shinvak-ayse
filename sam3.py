#!/usr/bin/env python3
"""
Pipeline: SAM3 → key intervals where the target object appears → short clips (Kling limits)
→ Gemini-generated edit prompts (question + wrong option) → Kling video-to-video edit
→ ffmpeg concat → manifest with intervals, prompts, and output paths.

Requires: FAL_KEY, OPENROUTER_API_KEY.
Install: pip install fal-client opencv-python numpy requests

Example:
  export FAL_KEY=...
  export OPENROUTER_API_KEY=...
  python sam3.py --video data/conflict.mp4 \\
    --question "What color is the pen?" --wrong-option "blue" \\
    --object-prompt "pen"
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import requests

try:
    import fal_client
except ImportError as e:
    raise SystemExit(
        "Install fal-client: pip install fal-client\n" + str(e)
    ) from e

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


# --- SAM3 (fal) ---

def image_to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{b64}"


def segment_frame(
    frame_path: str,
    sam3_prompt: str,
    point_prompts: Optional[list],
    box_prompts: Optional[list],
) -> Optional[np.ndarray]:
    """Call fal-ai/sam-3/image; return boolean mask H×W or None."""
    args: dict[str, Any] = {
        "image_url": image_to_data_uri(frame_path),
        "prompt": sam3_prompt,
        "apply_mask": False,
    }
    if point_prompts:
        args["point_prompts"] = point_prompts
    if box_prompts:
        args["box_prompts"] = box_prompts

    try:
        result = fal_client.subscribe("fal-ai/sam-3/image", arguments=args)
    except Exception as e:
        print(f"  SAM3 error: {e}")
        return None

    mask_url = result.get("mask_url") or (
        result.get("masks", [{}])[0].get("url") if result.get("masks") else None
    )
    if not mask_url:
        return None

    r = requests.get(mask_url, timeout=60)
    r.raise_for_status()
    arr = np.frombuffer(r.content, dtype=np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    return gray.astype(bool)


def mask_fraction(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(mask.mean())


# --- Video / ffmpeg ---

def run_ffmpeg(args: list[str]) -> None:
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr or p.stdout}")


def probe_video(path: Path) -> tuple[float, float, int, int]:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    cap.release()
    duration = n / fps if fps > 0 else 0.0
    return fps, duration, w, h


def ensure_kling_geometry(src: Path, dst: Path) -> None:
    """
    Kling expects 720–2160 px sides, min 720×720, 24–60 fps; clip duration handled elsewhere.
    Scale so the shorter side is 720px; set 30 fps within allowed range.
    """
    run_ffmpeg(
        [
            "-i",
            str(src),
            "-vf",
            "scale='if(gt(iw,ih),-2,720)':'if(gt(iw,ih),720,-2)',fps=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(dst),
        ]
    )


def extract_clip(
    video: Path,
    start_sec: float,
    duration_sec: float,
    out_path: Path,
) -> None:
    run_ffmpeg(
        [
            "-ss",
            f"{start_sec:.4f}",
            "-i",
            str(video),
            "-t",
            f"{duration_sec:.4f}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(out_path),
        ]
    )


def concat_videos(clips: list[Path], out_path: Path) -> None:
    if not clips:
        raise ValueError("No clips to concatenate")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        list_path = Path(f.name)
        for c in clips:
            # concat demuxer needs escaped paths
            p = str(c.resolve()).replace("'", "'\\''")
            f.write(f"file '{p}'\n")
    try:
        run_ffmpeg(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                str(out_path),
            ]
        )
    finally:
        list_path.unlink(missing_ok=True)


# --- Intervals ---

@dataclass
class PresenceInterval:
    start_frame: int
    end_frame: int  # inclusive last frame index where object is seen (sampled)
    start_sec: float
    end_sec: float  # exclusive upper bound in time


def build_presence_intervals(
    sampled_hits: list[tuple[int, float]],
    fps: float,
    stride_frames: int,
    merge_gap_frames: int,
) -> list[PresenceInterval]:
    """Merge sampled frame hits into contiguous intervals (in frame index space)."""
    if not sampled_hits:
        return []
    sampled_hits = sorted(sampled_hits, key=lambda x: x[0])
    groups: list[list[tuple[int, float]]] = []
    cur = [sampled_hits[0]]
    for i in range(1, len(sampled_hits)):
        fi, _ = sampled_hits[i]
        prev_f, _ = sampled_hits[i - 1]
        if fi - prev_f <= merge_gap_frames:
            cur.append(sampled_hits[i])
        else:
            groups.append(cur)
            cur = [sampled_hits[i]]
    groups.append(cur)

    intervals: list[PresenceInterval] = []
    for g in groups:
        frames = [x[0] for x in g]
        sf, ef = min(frames), max(frames)
        start_sec = sf / fps
        end_sec = (ef + 1) / fps
        intervals.append(
            PresenceInterval(
                start_frame=sf,
                end_frame=ef,
                start_sec=start_sec,
                end_sec=end_sec,
                # note: mean mask frac available via fracs if needed
            )
        )
    return intervals


def split_for_kling(
    start_sec: float,
    end_sec: float,
    video_duration: float,
    min_d: float = 3.0,
    max_d: float = 10.0,
) -> list[tuple[float, float]]:
    """Return sub-intervals each within [min_d, max_d] seconds (Kling input constraints)."""
    s = max(0.0, min(start_sec, video_duration))
    e = max(s, min(end_sec, video_duration))
    if e - s <= 1e-9:
        return []

    # Pad segments shorter than min_d (centered), clamped to [0, video_duration]
    if e - s < min_d:
        mid = (s + e) / 2.0
        s = mid - min_d / 2.0
        e = mid + min_d / 2.0
        if s < 0:
            e -= s
            s = 0.0
        if e > video_duration:
            s -= e - video_duration
            e = video_duration
        s = max(0.0, s)
        if e - s < min_d - 1e-6:
            return [(s, e)]

    segs: list[tuple[float, float]] = []
    cur = s
    while cur < e - 1e-6:
        rem = e - cur
        if rem > max_d + 1e-6:
            segs.append((cur, cur + max_d))
            cur += max_d
            continue
        if rem >= min_d - 1e-6:
            segs.append((cur, e))
            break
        # rem < min_d: fold tail into previous segment or split previous to fit Kling max
        if not segs:
            segs.append((cur, e))
            break
        ps, pe = segs[-1]
        if e - ps <= max_d + 1e-6:
            segs[-1] = (ps, e)
            break
        new_pe = e - min_d
        if new_pe >= ps + min_d - 1e-6:
            segs[-1] = (ps, new_pe)
            segs.append((new_pe, e))
        else:
            segs.append((max(cur, e - min_d), e))
        break

    # Drop impossible zero-width (should not happen)
    return [(a, b) for a, b in segs if b - a >= 1e-6]


# --- Gemini ---

GEMINI_EDIT_SYSTEM = """You write precise instructions for Kling AI video-to-video editing.
The editor will receive a short clip cut from a longer video. Your job is to describe ONLY
the visual edits needed so that the clip misleadingly supports a specific wrong multiple-choice
answer, while staying physically plausible.

Rules:
- Output a single paragraph of concrete visual directions (subject appearance, colors, props,
  lighting, micro-motion). No markdown, no bullet labels.
- Reference the question and the wrong option explicitly at the start in one short clause,
  then describe the edit.
- Do NOT mention SAM3, masks, AI models, or datasets.
- Keep under 120 words.
"""


def build_gemini_edit_prompt(
    question: str,
    wrong_option: str,
    object_prompt: str,
    clip_start_sec: float,
    clip_end_sec: float,
    model_name: str,
) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY")

    user = (
        f"Target object (segmentation prompt): {object_prompt}\n"
        f"Clip time range in source video: {clip_start_sec:.2f}s – {clip_end_sec:.2f}s\n"
        f"Multiple-choice question: {question}\n"
        f"The distractor answer to make visually believable (incorrect ground truth): {wrong_option}\n"
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": GEMINI_EDIT_SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0.55,
        "max_tokens": 400,
    }
    resp = requests.post(
        OPENROUTER_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/evannguyendo",
            "X-Title": "Video Benchmark",
        },
        json=payload,
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"OpenRouter error {resp.status_code}: {resp.text[:400]}"
        )
    data = resp.json()
    text = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        or ""
    ).strip()
    if not text:
        raise RuntimeError("OpenRouter returned empty edit prompt")
    return text


# --- Kling ---

def kling_edit(
    video_url: str,
    prompt: str,
    endpoint: str,
    keep_audio: bool,
) -> str:
    args: dict[str, Any] = {
        "prompt": prompt,
        "video_url": video_url,
        "keep_audio": keep_audio,
    }
    result = fal_client.subscribe(endpoint, arguments=args)
    vid = result.get("video") or {}
    url = vid.get("url")
    if not url:
        raise RuntimeError(f"Kling returned no video URL: {result!r}")
    return url


def download_url(url: str, dest: Path) -> None:
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    dest.write_bytes(r.content)


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAM3 key intervals → Kling edit → concat + manifest"
    )
    parser.add_argument("--video", required=True, type=Path, help="Source video path")
    parser.add_argument("--question", required=True, help="Benchmark question text")
    parser.add_argument(
        "--wrong-option",
        required=True,
        help="The incorrect answer option to realize visually (distractor)",
    )
    parser.add_argument(
        "--object-prompt",
        default="person",
        help="SAM3 text prompt for the object to localize (e.g. pen, ball, car)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sam3_kling_output"),
        help="Directory for clips, frames, edited segments, manifest",
    )
    parser.add_argument(
        "--sample-stride-frames",
        type=int,
        default=15,
        help="Run SAM3 every N frames (cost control)",
    )
    parser.add_argument(
        "--merge-gap-frames",
        type=int,
        default=None,
        help="Merge presence runs if gap ≤ this many frames (default: 2× stride)",
    )
    parser.add_argument(
        "--min-mask-fraction",
        type=float,
        default=0.002,
        help="Treat frame as 'object present' if mask coverage ≥ this fraction",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=20,
        help="Max Kling calls (after splitting); caps cost",
    )
    parser.add_argument(
        "--kling-endpoint",
        default="fal-ai/kling-video/o1/standard/video-to-video/edit",
        help="fal model id for Kling video edit",
    )
    parser.add_argument(
        "--gemini-model",
        default="google/gemini-3-flash-preview",
        help="OpenRouter model id used to generate edit prompts",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Pass keep_audio=true to Kling (clips are extracted without audio by default)",
    )
    args = parser.parse_args()

    merge_gap = args.merge_gap_frames or (args.sample_stride_frames * 2)

    if not os.environ.get("FAL_KEY"):
        print("Error: set FAL_KEY for fal.ai", file=sys.stderr)
        sys.exit(1)
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Error: set OPENROUTER_API_KEY for OpenRouter (Gemini edit prompts)", file=sys.stderr)
        sys.exit(1)

    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        print(f"Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    out_root = args.output_dir.expanduser().resolve()
    frames_dir = out_root / "sampled_frames"
    clips_dir = out_root / "source_clips"
    kling_ready_dir = out_root / "kling_input_clips"
    edited_dir = out_root / "edited_clips"
    keyframes_dir = out_root / "keyframes"
    for d in (out_root, frames_dir, clips_dir, kling_ready_dir, edited_dir, keyframes_dir):
        d.mkdir(parents=True, exist_ok=True)

    fps, duration, vw, vh = probe_video(video_path)
    print(f"Video: {duration:.2f}s @ {fps:.2f} fps, {vw}×{vh}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Could not open video", file=sys.stderr)
        sys.exit(1)

    sampled_hits: list[tuple[int, float]] = []
    frame_idx = 0
    print(f"SAM3 sampling every {args.sample_stride_frames} frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.sample_stride_frames != 0:
            frame_idx += 1
            continue

        fp = frames_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(fp), frame)

        mask = segment_frame(
            str(fp),
            args.object_prompt,
            None,
            None,
        )
        if mask is not None:
            frac = mask_fraction(mask)
            if frac >= args.min_mask_fraction:
                sampled_hits.append((frame_idx, frac))
                print(f"  hit frame {frame_idx} mask={frac:.4f}")

        frame_idx += 1

    cap.release()

    if not sampled_hits:
        print(
            "No frames passed SAM3 threshold. Lower --min-mask-fraction or adjust --object-prompt.",
            file=sys.stderr,
        )
        sys.exit(2)

    intervals = build_presence_intervals(
        sampled_hits, fps, args.sample_stride_frames, merge_gap
    )
    print(f"Presence intervals (sampled): {len(intervals)}")

    # Expand intervals into Kling-sized clips
    raw_clips: list[tuple[float, float, PresenceInterval]] = []
    for it in intervals:
        for s, e in split_for_kling(it.start_sec, it.end_sec, duration):
            raw_clips.append((s, e, it))

    raw_clips = raw_clips[: args.max_segments]
    print(f"Clips after Kling length constraints: {len(raw_clips)}")

    manifest: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "video_path": str(video_path),
        "fps": fps,
        "duration_sec": duration,
        "question": args.question,
        "wrong_option": args.wrong_option,
        "sam3_object_prompt": args.object_prompt,
        "sample_stride_frames": args.sample_stride_frames,
        "merge_gap_frames": merge_gap,
        "min_mask_fraction": args.min_mask_fraction,
        "presence_intervals": [asdict(x) for x in intervals],
        "kling_endpoint": args.kling_endpoint,
        "gemini_model": args.gemini_model,
        "segments": [],
    }

    edited_paths: list[Path] = []

    for seg_i, (cs, ce, src_it) in enumerate(raw_clips):
        seg_label = f"seg_{seg_i:03d}"
        dur = ce - cs
        raw_clip = clips_dir / f"{seg_label}_src.mp4"
        kling_in = kling_ready_dir / f"{seg_label}_kling.mp4"
        edited_out = edited_dir / f"{seg_label}_edited.mp4"
        kf_path = keyframes_dir / f"{seg_label}_ref.png"

        print(f"\n[{seg_i + 1}/{len(raw_clips)}] clip {cs:.2f}–{ce:.2f}s (dur {dur:.2f}s)")

        extract_clip(video_path, cs, dur, raw_clip)
        ensure_kling_geometry(raw_clip, kling_in)

        # Keyframe: middle frame of this clip
        cap = cv2.VideoCapture(str(video_path))
        mid_t = (cs + ce) / 2.0
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_t * 1000.0)
        ok, fr = cap.read()
        cap.release()
        if ok and fr is not None:
            cv2.imwrite(str(kf_path), fr)

        edit_prompt = build_gemini_edit_prompt(
            args.question,
            args.wrong_option,
            args.object_prompt,
            cs,
            ce,
            args.gemini_model,
        )
        print(f"  Gemini prompt ({len(edit_prompt)} chars)")

        out_url: Optional[str] = None
        try:
            ku = fal_client.upload_file(str(kling_in))
            out_url = kling_edit(
                ku, edit_prompt, args.kling_endpoint, args.keep_audio
            )
            download_url(out_url, edited_out)
            status = "ok"
            err = None
        except Exception as e:
            status = "error"
            err = str(e)
            print(f"  ERROR: {e}")

        seg_record = {
            "segment_index": seg_i,
            "source_interval_sec": {"start": cs, "end": ce},
            "derived_from_presence_interval": asdict(src_it),
            "sampled_frame_range_hint": {
                "start_frame": src_it.start_frame,
                "end_frame": src_it.end_frame,
            },
            "keyframe_image_path": str(kf_path) if kf_path.exists() else None,
            "source_clip_path": str(raw_clip),
            "kling_input_clip_path": str(kling_in),
            "gemini_edit_prompt": edit_prompt,
            "edited_clip_path": str(edited_out) if status == "ok" else None,
            "kling_output_url": out_url,
            "status": status,
            "error": err,
        }
        manifest["segments"].append(seg_record)

        if status == "ok":
            edited_paths.append(edited_out)

    combined = out_root / "combined_edited.mp4"
    if len(edited_paths) >= 1:
        concat_videos(edited_paths, combined)
        manifest["combined_video_path"] = str(combined)
        print(f"\nCombined video: {combined}")
    else:
        manifest["combined_video_path"] = None
        print("\nNo successful edits; combined video skipped.")

    manifest_path = out_root / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
