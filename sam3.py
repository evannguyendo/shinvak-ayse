#!/usr/bin/env python3
"""
Pipeline
--------
1. Use SAM3 (fal.ai) to find every time interval in the video where the
   target object is actually visible.
2. Extract those intervals as short clips (3–10 s, Kling's input constraint).
3. Use Gemini (via OpenRouter) to auto-generate scene context and a detailed Kling
   video-editing instruction from the question, wrong answer option, and a reference keyframe.
4. Send each clip to Kling video-to-video edit (fal.ai) to produce an
   edited version that visually supports the wrong answer.
5. Splice each edited clip back into the ORIGINAL video at the exact time
   position it came from → final_spliced.mp4
   (original_before + edited_clip + original_after)
6. Save everything: edited clips, keyframes, manifest (intervals, prompts,
   paths, Kling URLs), plus raw_outputs/ (mirror of source extracts, Kling inputs,
   unmodified API downloads, and copies of final spliced / concat videos).

Requires: OPENROUTER_API_KEY and (for full runs) FAL_KEY — put them in a file named .env
          in the same folder as sam3.py (no need to export in the shell). Also needs ffmpeg.

Install:  pip install fal-client opencv-python numpy requests
          Optional: pip install python-dotenv  (otherwise .env is parsed with a tiny built-in reader)
          ffmpeg (macOS):  brew install ffmpeg

Run from any directory (use the path to this script); --video may be relative to the project
folder (e.g. data/control.mp4) or to your current working directory:

  python /path/to/shivank-ayse/sam3.py --video data/control.mp4 \\
    --question "What color is the pen?" --wrong-option "blue" --dry-run
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import requests

_PROJECT_ROOT = Path(__file__).resolve().parent


def _parse_env_file(path: Path) -> dict[str, str]:
    """Minimal KEY=value reader (used when python-dotenv is not installed)."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            out[key] = val
    return out


def _load_env_from_project() -> None:
    """
    Load OPENROUTER_API_KEY / FAL_KEY from <project>/.env so you do not need export in the shell.
    Existing environment variables always win (never overwritten).
    """
    env_path = _PROJECT_ROOT / ".env"
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        pass
    for key, val in _parse_env_file(env_path).items():
        if val and not (os.environ.get(key) or "").strip():
            os.environ[key] = val


def _resolve_input_path(user_path: Path) -> Path:
    """
    Resolve --video relative to CWD first, then the project directory (so you can run
    this script from anywhere with --video data/foo.mp4).
    """
    p = user_path.expanduser()
    if p.is_file():
        return p.resolve()
    cand = Path.cwd() / p
    if cand.is_file():
        return cand.resolve()
    cand = _PROJECT_ROOT / p
    if cand.is_file():
        return cand.resolve()
    return p.resolve()


def _fal_client():
    """Import fal_client only when SAM3/Kling runs (so --dry-run does not need the package)."""
    try:
        import fal_client as fc  # type: ignore[import-untyped]
        return fc
    except ImportError as e:
        raise SystemExit(
            "Missing package: fal-client (needed for SAM3 and Kling).\n"
            "  pip install fal-client\n"
            "Or use --dry-run to test without fal.ai.\n"
            + str(e)
        ) from e


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_HEADERS_BASE = {
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/evannguyendo",
    "X-Title": "Video Benchmark",
}


def _openrouter_post(payload: dict, api_key: str, timeout: int = 60) -> dict:
    headers = {**_OPENROUTER_HEADERS_BASE, "Authorization": f"Bearer {api_key}"}
    resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text[:400]}")
    return resp.json()


def _openrouter_text(response: dict) -> str:
    return (
        (response.get("choices") or [{}])[0]
        .get("message", {})
        .get("content", "")
        or ""
    ).strip()


# --- Object extraction from question ---

OBJECT_EXTRACT_SYSTEM = """\
You extract the single most visually distinctive physical object from a multiple-choice video \
question so it can be used as a segmentation target.

Rules:
- Return ONLY the object noun (1–4 words, lowercase), nothing else.
- Pick the object whose presence/absence/appearance the question is directly testing.
- Examples:
    Q: "What color is the pen?" → pen
    Q: "How many dogs are in the frame?" → dog
    Q: "Is the person holding an umbrella?" → umbrella
    Q: "What sport is being played?" → ball
"""


def extract_object_from_question(question: str, model_name: str) -> str:
    """
    Ask the LLM to pull the key visual object out of the benchmark question.
    Returns a short lowercase noun suitable for SAM3's text prompt.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY")

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": OBJECT_EXTRACT_SYSTEM},
            {"role": "user", "content": f"Question: {question}"},
        ],
        "temperature": 0.0,
        "max_tokens": 20,
    }
    data = _openrouter_post(payload, api_key)
    obj = _openrouter_text(data).lower().strip().strip(".")
    if not obj:
        raise RuntimeError("LLM returned empty object extraction")
    return obj


OBJECT_CONTEXT_SYSTEM = """\
You write a single-sentence scene hint for an AI video editor that will recolor or \
modify an object already in a clip.

Rules:
- Describe how the target object appears in the clip (color, material, pose, location).
- State the specific in-place visual change needed so the clip misleadingly supports \
the wrong multiple-choice answer.
- Stress: edit the existing object only — same instance, position, and motion; do NOT add, \
duplicate, or replace with a different object.
- One sentence, plain English, under 60 words. No markdown or labels.
"""


def generate_object_context(
    question: str,
    wrong_option: str,
    object_prompt: str,
    model_name: str,
    keyframe_path: Optional[Path] = None,
) -> str:
    """
    Ask the LLM for a scene hint describing the object in the clip and the edit to apply.
    When keyframe_path is set, uses the reference frame so hints match visible appearance.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY")

    user_text = (
        f"Target object: {object_prompt}\n"
        f"Question: {question}\n"
        f"Wrong answer option to make visually believable: {wrong_option}\n"
    )
    if keyframe_path and keyframe_path.is_file():
        user_text += (
            "Reference frame from the clip is attached — ground your description in what you see.\n"
        )
        user_content: Any = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_to_data_uri(str(keyframe_path))}},
        ]
    else:
        user_content = user_text

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": OBJECT_CONTEXT_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.25,
        "max_tokens": 120,
    }
    ctx = _openrouter_text(_openrouter_post(payload, api_key)).strip().strip('"')
    if not ctx:
        raise RuntimeError("LLM returned empty object context")
    return ctx


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
        result = _fal_client().subscribe("fal-ai/sam-3/image", arguments=args)
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

# Shared ffmpeg video-filter string: scale shortest side to 720 px, force 30 fps.
# Used for both Kling input prep and splicing normalization so all parts match.
VF_NORMALIZE = "scale='if(gt(iw,ih),-2,720)':'if(gt(iw,ih),720,-2)',fps=30"


# Resolved in main() before any ffmpeg call (default "ffmpeg" for PATH lookup).
_FFMPEG_CMD: list[str] = ["ffmpeg"]


def _find_ffmpeg() -> Optional[str]:
    """PATH first, then common Homebrew locations on macOS."""
    w = shutil.which("ffmpeg")
    if w:
        return w
    for cand in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if Path(cand).is_file():
            return cand
    return None


def run_ffmpeg(args: list[str]) -> None:
    p = subprocess.run(
        [_FFMPEG_CMD[0], "-hide_banner", "-loglevel", "error", "-y", *args],
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


def normalize_clip(src: Path, dst: Path) -> None:
    """Re-encode any clip to the common 720p/30fps/yuv420p spec."""
    run_ffmpeg([
        "-i", str(src),
        "-vf", VF_NORMALIZE,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
        str(dst),
    ])


def ensure_kling_geometry(src: Path, dst: Path) -> None:
    """Prepare a clip for Kling: shortest side 720 px, 30 fps, no audio."""
    normalize_clip(src, dst)


def extract_clip(
    video: Path,
    start_sec: float,
    duration_sec: float,
    out_path: Path,
) -> None:
    run_ffmpeg([
        "-ss", f"{start_sec:.4f}",
        "-i", str(video),
        "-t", f"{duration_sec:.4f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
        str(out_path),
    ])


def concat_videos(clips: list[Path], out_path: Path) -> None:
    """Join clips that already share codec/resolution/fps (uses -c copy)."""
    if not clips:
        raise ValueError("No clips to concatenate")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        list_path = Path(f.name)
        for c in clips:
            p = str(c.resolve()).replace("'", "'\\''")
            f.write(f"file '{p}'\n")
    try:
        run_ffmpeg([
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            str(out_path),
        ])
    finally:
        list_path.unlink(missing_ok=True)


def concat_videos_normalize(clips: list[Path], out_path: Path) -> None:
    """
    Concatenate clips that may differ in resolution/fps (e.g. raw Kling outputs).
    Each clip is normalized to the shared spec before joining.
    """
    if not clips:
        raise ValueError("No clips to concatenate")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        normed: list[Path] = []
        for i, c in enumerate(clips):
            n = tmp_dir / f"n{i:04d}.mp4"
            normalize_clip(c, n)
            normed.append(n)
        concat_videos(normed, out_path)


def build_spliced_video(
    original: Path,
    edits: list[tuple[float, float, Path]],
    out_path: Path,
    video_duration: float,
) -> None:
    """
    Rebuild the full video by replacing each [start_sec, end_sec) window with
    its edited clip.  All parts are normalized to the same spec before joining
    so the concat is seamless.

    Layout:  original[0 → edit1.start]
           + edited_clip_1
           + original[edit1.end → edit2.start]
           + edited_clip_2
           + ...
           + original[last_edit.end → video_end]
    """
    if not edits:
        raise ValueError("No edits provided for splice")

    edits = sorted(edits, key=lambda x: x[0])

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        parts: list[Path] = []
        cursor = 0.0

        for i, (start_sec, end_sec, edited_clip) in enumerate(edits):
            # --- original segment before this edit ---
            before_dur = start_sec - cursor
            if before_dur > 0.05:
                before = tmp_dir / f"p{i:04d}a_orig.mp4"
                run_ffmpeg([
                    "-ss", f"{cursor:.4f}",
                    "-i", str(original),
                    "-t", f"{before_dur:.4f}",
                    "-vf", VF_NORMALIZE,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
                    str(before),
                ])
                parts.append(before)

            # --- normalized edited clip ---
            edit_norm = tmp_dir / f"p{i:04d}b_edit.mp4"
            normalize_clip(edited_clip, edit_norm)
            parts.append(edit_norm)

            cursor = end_sec

        # --- tail of original after the last edit ---
        tail_dur = video_duration - cursor
        if tail_dur > 0.05:
            tail = tmp_dir / f"ptail_orig.mp4"
            run_ffmpeg([
                "-ss", f"{cursor:.4f}",
                "-i", str(original),
                "-t", f"{tail_dur:.4f}",
                "-vf", VF_NORMALIZE,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
                str(tail),
            ])
            parts.append(tail)

        if not parts:
            raise ValueError("Splice produced no output parts")

        concat_videos(parts, out_path)


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


def select_presence_interval(
    intervals: list[PresenceInterval],
    video_duration: float,
    pick: str,
    index: Optional[int],
) -> Optional[PresenceInterval]:
    """
    Choose one SAM3 presence interval to edit.
    pick: first | last | middle (closest midpoint to video center) | longest
    index: if set, overrides pick (0-based into intervals list)
    """
    if not intervals:
        return None
    if index is not None:
        if index < 0 or index >= len(intervals):
            raise ValueError(
                f"--interval-index {index} out of range; "
                f"found {len(intervals)} presence interval(s) (use 0..{len(intervals) - 1})"
            )
        return intervals[index]

    if pick == "first":
        return intervals[0]
    if pick == "last":
        return intervals[-1]
    if pick == "longest":
        return max(
            intervals,
            key=lambda it: it.end_sec - it.start_sec,
        )
    if pick == "middle":
        center = video_duration / 2.0
        return min(
            intervals,
            key=lambda it: abs((it.start_sec + it.end_sec) / 2.0 - center),
        )
    raise ValueError(f"Unknown interval pick mode: {pick}")


# --- Gemini ---

GEMINI_EDIT_SYSTEM = """You write precise instructions for Kling AI video-to-video editing.
The editor receives a short clip cut from a longer video. A reference keyframe from the clip \
is attached — ground all your directions in exactly what you see in that image.

Critical rules:
- RECOLOR or MODIFY the existing object already in the footage. Do NOT add a new object,
  duplicate, or replace with a different item. Same instance, same position, same motion —
  only change its appearance (e.g. color, finish, texture).
- Explicitly instruct Kling to PRESERVE unchanged everything else in the frame:
  name the exact colors of the background, surface/table, hands or skin tones, shadows,
  and any other visible objects as you see them in the keyframe.
- Use specific color descriptions (e.g. "warm ivory surface", "soft pink matte finish →
  solid deep cobalt blue with a subtle satin sheen") not vague terms.
- Require temporal consistency: the edit must look identical in every frame of the clip,
  with no flickering or gradual drift.
- Output one paragraph of concrete directions. No markdown, no bullet labels.
- Start with one short clause naming the question and wrong option, then the edit details,
  then an explicit list of what must not change.
- Do NOT mention SAM3, masks, AI models, or datasets.
- Keep under 200 words.
"""


def build_gemini_edit_prompt(
    question: str,
    wrong_option: str,
    object_prompt: str,
    clip_start_sec: float,
    clip_end_sec: float,
    model_name: str,
    object_context: Optional[str] = None,
    keyframe_path: Optional[Path] = None,
) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY")

    user_text = (
        f"Target object (segmentation prompt): {object_prompt}\n"
        f"Clip time range in source video: {clip_start_sec:.2f}s – {clip_end_sec:.2f}s\n"
        f"Multiple-choice question: {question}\n"
        f"The distractor answer to make visually believable (incorrect ground truth): {wrong_option}\n"
        "Edit type: in-place recolor/modify of the existing object in the clip — do not add objects.\n"
    )
    if object_context:
        user_text += f"Scene context (what is already in the video): {object_context}\n"

    if keyframe_path and keyframe_path.is_file():
        user_text += (
            "The reference keyframe from the clip is attached. "
            "Use it to ground every color, material, and background detail in your instruction.\n"
        )
        user_content: Any = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_to_data_uri(str(keyframe_path))}},
        ]
    else:
        user_content = user_text

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": GEMINI_EDIT_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.45,
        "max_tokens": 512,
    }
    text = _openrouter_text(_openrouter_post(payload, api_key))
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
    result = _fal_client().subscribe(endpoint, arguments=args)
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


def _validate_openrouter_key() -> None:
    key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    env_file = _PROJECT_ROOT / ".env"
    if not key:
        print(
            "Error: OPENROUTER_API_KEY is not set.\n"
            f"  Add this line to:  {env_file}\n"
            "    OPENROUTER_API_KEY=sk-or-v1-...\n"
            "  (get a key at https://openrouter.ai/keys )\n"
            "  Or export it in the shell before running.",
            file=sys.stderr,
        )
        sys.exit(1)
    low = key.lower()
    if "paste_your" in low or low in ("sk-or-v1-xxx", "your_key_here"):
        print(
            "Error: OPENROUTER_API_KEY looks like a placeholder, not a real key.\n"
            "  Replace it with your actual key from https://openrouter.ai/keys",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    _load_env_from_project()

    parser = argparse.ArgumentParser(
        description="SAM3 key intervals → Kling edit → concat + manifest"
    )
    parser.add_argument(
        "--video",
        required=True,
        type=Path,
        help="Source video path (relative paths: tried vs CWD, then vs project folder containing sam3.py)",
    )
    parser.add_argument("--question", required=True, help="Benchmark question text")
    parser.add_argument(
        "--wrong-option",
        required=True,
        help="The incorrect answer option to realize visually (distractor)",
    )
    parser.add_argument(
        "--object-prompt",
        default=None,
        help=(
            "SAM3 text prompt for the object to localize (e.g. pen, ball, car). "
            "If omitted, the LLM will extract it automatically from --question."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip SAM3 and Kling API calls (uses every stride frame as a mock hit, "
            "copies the source clip as the edited output). "
            "Still runs ffmpeg and the Gemini prompt call so you can inspect real outputs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for outputs (default: sam3_kling_output next to sam3.py). "
            "If you pass a relative path, it is resolved from your current working directory."
        ),
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
        default=1,
        help="Max Kling clips to generate from the chosen presence interval (default: 1)",
    )
    parser.add_argument(
        "--interval-pick",
        choices=["first", "last", "middle", "longest"],
        default="first",
        help=(
            "Which SAM3 presence interval to edit: first, last, middle (midpoint nearest "
            "video center), or longest. Use with --interval-index to pick a specific one."
        ),
    )
    parser.add_argument(
        "--interval-index",
        type=int,
        default=None,
        metavar="N",
        help="0-based index into presence_intervals (overrides --interval-pick). "
        "E.g. 1 = second detected block (~later in video for control.mp4).",
    )
    parser.add_argument(
        "--object-context",
        default=None,
        help=(
            "Optional override for the scene hint fed into the Kling edit prompt. "
            "If omitted, Gemini generates it from --question, --wrong-option, and a "
            "reference keyframe from the chosen presence interval."
        ),
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
    parser.add_argument(
        "--ffmpeg",
        default=None,
        metavar="PATH",
        help="Path to ffmpeg binary (default: look on PATH, then /opt/homebrew/bin/ffmpeg, /usr/local/bin/ffmpeg)",
    )
    args = parser.parse_args()

    ffmpeg_bin = args.ffmpeg or _find_ffmpeg()
    if not ffmpeg_bin:
        print(
            "Error: ffmpeg not found. This script needs ffmpeg to cut and splice video.\n"
            "  macOS:  brew install ffmpeg\n"
            "  Then ensure brew's bin is on your PATH, or pass e.g.\n"
            "    --ffmpeg /opt/homebrew/bin/ffmpeg",
            file=sys.stderr,
        )
        sys.exit(1)
    _FFMPEG_CMD[0] = ffmpeg_bin
    print(f"Using ffmpeg: {ffmpeg_bin}")

    _validate_openrouter_key()

    merge_gap = args.merge_gap_frames or (args.sample_stride_frames * 2)

    if not args.dry_run and not (os.environ.get("FAL_KEY") or "").strip():
        env_file = _PROJECT_ROOT / ".env"
        print(
            "Error: FAL_KEY is not set (required for SAM3 and Kling).\n"
            f"  Add to {env_file}:  FAL_KEY=...\n"
            "  Or use --dry-run to skip fal.ai.\n"
            "  Or export FAL_KEY in the shell.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Resolve object prompt (auto-extract if not supplied) ---
    object_prompt = args.object_prompt
    if not object_prompt:
        print(f"No --object-prompt supplied; asking LLM to extract object from question...")
        try:
            object_prompt = extract_object_from_question(args.question, args.gemini_model)
            print(f"  LLM extracted object: '{object_prompt}'")
        except Exception as e:
            print(f"  Object extraction failed: {e}\n  Falling back to 'person'.", file=sys.stderr)
            object_prompt = "person"
    else:
        print(f"Object prompt (manual): '{object_prompt}'")

    video_path = _resolve_input_path(args.video)
    if not video_path.is_file():
        print(f"Video not found: {video_path}", file=sys.stderr)
        print(
            f"  Tried: {args.video!s} as given, then under {Path.cwd()}, then under {_PROJECT_ROOT}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.output_dir is None:
        out_root = (_PROJECT_ROOT / "sam3_kling_output").resolve()
    else:
        p = args.output_dir.expanduser()
        out_root = p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()
    frames_dir = out_root / "sampled_frames"
    clips_dir = out_root / "source_clips"
    kling_ready_dir = out_root / "kling_input_clips"
    edited_dir = out_root / "edited_clips"
    keyframes_dir = out_root / "keyframes"
    raw_video_dir = out_root / "raw_outputs"
    edit_prompts_dir = out_root / "edit_prompts"
    for d in (
        out_root,
        frames_dir,
        clips_dir,
        kling_ready_dir,
        edited_dir,
        keyframes_dir,
        raw_video_dir,
        edit_prompts_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)

    fps, duration, vw, vh = probe_video(video_path)
    print(f"Video: {duration:.2f}s @ {fps:.2f} fps, {vw}×{vh}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Could not open video", file=sys.stderr)
        sys.exit(1)

    sampled_hits: list[tuple[int, float]] = []
    frame_idx = 0
    if args.dry_run:
        print(f"[DRY-RUN] Mocking SAM3 — treating every {args.sample_stride_frames}-th frame as a hit...")
    else:
        print(f"SAM3 sampling every {args.sample_stride_frames} frames for '{object_prompt}'...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.sample_stride_frames != 0:
            frame_idx += 1
            continue

        fp = frames_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(fp), frame)

        if args.dry_run:
            # Mock: every sampled frame is a hit with a synthetic mask fraction
            sampled_hits.append((frame_idx, 0.05))
            print(f"  [mock] hit frame {frame_idx}")
        else:
            mask = segment_frame(str(fp), object_prompt, None, None)
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
    for i, it in enumerate(intervals):
        print(
            f"  [{i}] frames {it.start_frame}–{it.end_frame}  "
            f"time {it.start_sec:.2f}s–{it.end_sec:.2f}s  "
            f"(dur {it.end_sec - it.start_sec:.2f}s)"
        )

    try:
        chosen = select_presence_interval(
            intervals, duration, args.interval_pick, args.interval_index
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    if chosen is None:
        print("No presence interval to edit.", file=sys.stderr)
        sys.exit(2)

    pick_label = (
        f"index {args.interval_index}"
        if args.interval_index is not None
        else args.interval_pick
    )
    print(
        f"\nChosen interval ({pick_label}): "
        f"{chosen.start_sec:.2f}s – {chosen.end_sec:.2f}s "
        f"(frames {chosen.start_frame}–{chosen.end_frame})"
    )

    # Expand only the chosen interval into Kling-sized clips (3–10 s each)
    raw_clips: list[tuple[float, float, PresenceInterval]] = []
    for s, e in split_for_kling(chosen.start_sec, chosen.end_sec, duration):
        raw_clips.append((s, e, chosen))

    raw_clips = raw_clips[: args.max_segments]
    print(f"Clips after Kling length constraints: {len(raw_clips)}")

    object_context = args.object_context
    object_context_source = "manual" if object_context else None
    context_kf = keyframes_dir / "context_ref.png"
    mid_sec = (chosen.start_sec + chosen.end_sec) / 2.0
    vcap_ctx = cv2.VideoCapture(str(video_path))
    vcap_ctx.set(cv2.CAP_PROP_POS_MSEC, mid_sec * 1000.0)
    ok_ctx, ref_frame = vcap_ctx.read()
    vcap_ctx.release()
    context_kf_path: Optional[Path] = None
    if ok_ctx and ref_frame is not None:
        cv2.imwrite(str(context_kf), ref_frame)
        context_kf_path = context_kf
        print(f"Context reference keyframe @ {mid_sec:.2f}s → {context_kf.name}")

    if not object_context:
        print("No --object-context supplied; generating scene hint via LLM...")
        try:
            object_context = generate_object_context(
                args.question,
                args.wrong_option,
                object_prompt,
                args.gemini_model,
                context_kf_path,
            )
            object_context_source = "llm_generated"
            print(f"  LLM object context: {object_context}")
        except Exception as e:
            print(f"  Object context generation failed: {e}", file=sys.stderr)
            object_context_source = "failed"

    manifest: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "project_root": str(_PROJECT_ROOT),
        "video_path": str(video_path),
        "fps": fps,
        "duration_sec": duration,
        "question": args.question,
        "wrong_option": args.wrong_option,
        "object_prompt_source": "manual" if args.object_prompt else "llm_extracted",
        "sam3_object_prompt": object_prompt,
        "dry_run": args.dry_run,
        "sample_stride_frames": args.sample_stride_frames,
        "merge_gap_frames": merge_gap,
        "min_mask_fraction": args.min_mask_fraction,
        "presence_intervals": [asdict(x) for x in intervals],
        "chosen_presence_interval": asdict(chosen),
        "interval_pick": pick_label,
        "object_context": object_context,
        "object_context_source": object_context_source,
        "context_reference_keyframe": str(context_kf_path) if context_kf_path else None,
        "kling_endpoint": args.kling_endpoint,
        "gemini_model": args.gemini_model,
        "segments": [],
        "final_spliced_video_path": None,
        "isolated_edits_concat_path": None,
        "raw_outputs_dir": str(raw_video_dir),
        "raw_outputs_final_spliced": None,
        "raw_outputs_edited_concat": None,
    }

    # Tracks (start_sec, end_sec, edited_clip_path) for the final splice step
    splice_edits: list[tuple[float, float, Path]] = []

    for seg_i, (cs, ce, src_it) in enumerate(raw_clips):
        seg_label = f"seg_{seg_i:03d}"
        dur = ce - cs
        raw_clip = clips_dir / f"{seg_label}_src.mp4"
        kling_in = kling_ready_dir / f"{seg_label}_kling.mp4"
        edited_out = edited_dir / f"{seg_label}_edited.mp4"
        kf_path = keyframes_dir / f"{seg_label}_ref.png"

        print(f"\n[{seg_i + 1}/{len(raw_clips)}] interval {cs:.2f}s – {ce:.2f}s  (duration {dur:.2f}s)")

        raw_source_saved = raw_video_dir / f"{seg_label}_01_source_interval_extract.mp4"
        raw_kling_in_saved = raw_video_dir / f"{seg_label}_02_kling_input_normalized.mp4"
        raw_kling_api_saved = raw_video_dir / f"{seg_label}_03_kling_api_raw_download.mp4"

        # 1. Extract the interval from the original video
        extract_clip(video_path, cs, dur, raw_clip)
        shutil.copy2(raw_clip, raw_source_saved)
        # 2. Normalize geometry for Kling
        ensure_kling_geometry(raw_clip, kling_in)
        shutil.copy2(kling_in, raw_kling_in_saved)

        # 3. Save a keyframe (middle of the interval) for reference
        vcap = cv2.VideoCapture(str(video_path))
        vcap.set(cv2.CAP_PROP_POS_MSEC, ((cs + ce) / 2.0) * 1000.0)
        ok, fr = vcap.read()
        vcap.release()
        if ok and fr is not None:
            cv2.imwrite(str(kf_path), fr)
            print(f"  Keyframe saved: {kf_path.name}")

        # 4. Generate Kling editing instruction via Gemini (with keyframe for visual grounding)
        edit_prompt = build_gemini_edit_prompt(
            args.question,
            args.wrong_option,
            object_prompt,
            cs,
            ce,
            args.gemini_model,
            object_context=object_context,
            keyframe_path=kf_path if kf_path.exists() else None,
        )
        prompt_path = edit_prompts_dir / f"{seg_label}_edit_instruction.txt"
        prompt_path.write_text(edit_prompt, encoding="utf-8")
        print(f"  Edit instruction ({len(edit_prompt)} chars) — saved: {prompt_path.name}")
        for line in textwrap.wrap(edit_prompt, width=88):
            print(f"    {line}")

        # 5. Edit the clip with Kling (or mock for dry-run)
        out_url: Optional[str] = None
        if args.dry_run:
            shutil.copy2(kling_in, edited_out)
            shutil.copy2(kling_in, raw_kling_api_saved)
            print(f"  [DRY-RUN] Skipped Kling — using source clip as stand-in")
            status = "dry_run"
            err = None
        else:
            try:
                ku = _fal_client().upload_file(str(kling_in))
                out_url = kling_edit(ku, edit_prompt, args.kling_endpoint, args.keep_audio)
                # Save exact bytes from Kling URL into raw_outputs, then mirror to edited_clips/
                download_url(out_url, raw_kling_api_saved)
                shutil.copy2(raw_kling_api_saved, edited_out)
                print(f"  Kling edit saved: {edited_out.name} (raw API file: {raw_kling_api_saved.name})")
                status = "ok"
                err = None
            except Exception as e:
                status = "error"
                err = str(e)
                print(f"  ERROR (Kling): {e}")

        seg_record = {
            "segment_index": seg_i,
            "source_interval_sec": {"start": cs, "end": ce},
            "duration_sec": dur,
            "derived_from_presence_interval": asdict(src_it),
            "keyframe_image_path": str(kf_path) if kf_path.exists() else None,
            "source_clip_path": str(raw_clip),
            "kling_input_clip_path": str(kling_in),
            "gemini_edit_instruction": edit_prompt,
            "gemini_edit_instruction_path": str(prompt_path),
            "edited_clip_path": str(edited_out) if status in ("ok", "dry_run") else None,
            "kling_output_url": out_url,
            "status": status,
            "error": err,
            "raw_outputs": {
                "source_interval_extract": str(raw_source_saved),
                "kling_input_normalized": str(raw_kling_in_saved),
                "kling_api_raw_download": str(raw_kling_api_saved)
                if status in ("ok", "dry_run")
                else None,
            },
        }
        manifest["segments"].append(seg_record)

        if status in ("ok", "dry_run"):
            splice_edits.append((cs, ce, edited_out))

    # ------------------------------------------------------------------ #
    # 6. Splice edited intervals back into the original video
    # ------------------------------------------------------------------ #
    print(f"\n{'─'*60}")
    spliced_path = out_root / "final_spliced.mp4"
    isolated_path = out_root / "edited_clips_only.mp4"

    if splice_edits:
        print(f"Splicing {len(splice_edits)} edited interval(s) back into original video...")
        try:
            build_spliced_video(video_path, splice_edits, spliced_path, duration)
            manifest["final_spliced_video_path"] = str(spliced_path)
            print(f"  → final_spliced.mp4  (original video with edits applied in-place)")
            spliced_raw = raw_video_dir / "final_spliced_full_video.mp4"
            shutil.copy2(spliced_path, spliced_raw)
            manifest["raw_outputs_final_spliced"] = str(spliced_raw)
        except Exception as e:
            print(f"  Splice ERROR: {e}")

        # Also save just the edited clips concatenated (useful for quick review)
        try:
            concat_videos_normalize([ep for _, _, ep in splice_edits], isolated_path)
            manifest["isolated_edits_concat_path"] = str(isolated_path)
            print(f"  → edited_clips_only.mp4  (just the edited intervals, concatenated)")
            concat_raw = raw_video_dir / "edited_intervals_concat.mp4"
            shutil.copy2(isolated_path, concat_raw)
            manifest["raw_outputs_edited_concat"] = str(concat_raw)
        except Exception as e:
            print(f"  Concat ERROR: {e}")
    else:
        print("No successful edits — skipping splice and concat.")

    manifest_path = out_root / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest: {manifest_path}")
    print(f"Output directory: {out_root}")


if __name__ == "__main__":
    main()
