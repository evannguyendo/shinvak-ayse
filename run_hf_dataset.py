#!/usr/bin/env python3
"""
run_hf_dataset.py — general-purpose HuggingFace dataset runner for sam3.py
---------------------------------------------------------------------------
Loads any HF dataset that has videos + question/answer metadata, downloads
each video, and runs sam3.py on it exactly as if you had passed a single MP4
by hand.  Defaults are wired to shivank21/mvbench-temporal-conflict-subset
so it works out of the box for that dataset with no extra flags.

HOW VIDEO PATHS ARE RESOLVED
-----------------------------
The --col-video-path column value is interpreted in priority order:
  1. Already an existing local file path → used directly, no download.
  2. Starts with "http" → downloaded from that URL directly.
  3. Everything else → treated as a path relative to the HF repo root and
     downloaded from:
       https://huggingface.co/datasets/<repo>/resolve/<branch>/<value>

COLUMN DEFAULTS (shivank21 dataset)
------------------------------------
  --col-id           id
  --col-video-path   video_path        (e.g.  "videos/video_10929.mp4")
  --col-question     question
  --col-candidates   candidates        (JSON list of MCQ strings)
  --col-answer       answer            (correct answer text)
  --col-split        split             (optional — used for --split-filter)
  --col-notes        editability_notes (optional edit hint → --object-context)
  --col-start        accurate_start    (optional clip-window start in seconds)
  --col-end          accurate_end      (optional clip-window end in seconds)

EXAMPLES
--------
  # Run full shivank21 dataset (default), dry-run, verbose:
  python run_hf_dataset.py --dry-run --verbose

  # Same but limit to 3 rows for a quick smoke test:
  python run_hf_dataset.py --limit 3 --dry-run --verbose

  # Full production run (needs FAL_KEY + OPENROUTER_API_KEY in .env):
  python run_hf_dataset.py --out-dir hf_conflict_outputs

  # Resume after interruption:
  python run_hf_dataset.py --skip-existing --out-dir hf_conflict_outputs

  # Filter to one split:
  python run_hf_dataset.py --split-filter object_interaction --dry-run

  # Any other public HF dataset (map its columns to the expected roles):
  python run_hf_dataset.py \\
    --dataset some-user/some-dataset \\
    --col-video-path file_name \\
    --col-question   question_text \\
    --col-candidates options \\
    --col-answer     correct \\
    --col-notes      edit_hint \\
    --out-dir my_outputs
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import load_dotenv

from models_config import DEFAULT_MODELS_PATH, default_openrouter_model

# Re-use sam3 object-prompt logic so the runner passes an explicit --object-prompt.
# Import may fail if opencv is not installed in the runner's Python env.
try:
    from sam3 import derive_object_prompt_heuristic as _derive_object_prompt
except ImportError:
    def _derive_object_prompt(  # type: ignore[misc]
        question: str,
        answer: Optional[str] = None,
        object_context: Optional[str] = None,
    ) -> Optional[str]:
        notes = (object_context or "").strip()
        m = re.search(r"Swap\s+`([^`]+)`", notes, re.IGNORECASE)
        if m:
            return m.group(1).strip().lower()
        m2 = re.search(
            r"\bon the\s+([\w\s-]+?\s+object)\b",
            notes,
            re.IGNORECASE,
        )
        if m2:
            return m2.group(1).strip().lower()
        if answer:
            a = answer.strip().lower().strip(".")
            a = re.sub(r"^the\s+", "", a)
            colors = {
                "red", "blue", "green", "brown", "yellow", "cyan", "purple",
                "gray", "grey", "orange", "pink", "white", "black",
            }
            if (
                a
                and a not in ("yes", "no", "not sure")
                and a not in colors
                and a not in ("rubber", "metal", "sphere", "cube", "cylinder")
                and not re.fullmatch(r"[\d.]+", a)
            ):
                return a.split("/")[0].strip()
        return None

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)

# ---------------------------------------------------------------------------
# Defaults for the shivank21 dataset
# ---------------------------------------------------------------------------
_DEFAULT_DATASET     = "shivank21/mvbench-temporal-conflict-subset"
_DEFAULT_BRANCH      = "main"
_DEFAULT_META_FILE   = "metadata.json"   # JSON file in the repo with all rows
_DEFAULT_COL_ID      = "id"
_DEFAULT_COL_VIDEO   = "video_path"      # repo-relative path, e.g. videos/foo.mp4
_DEFAULT_COL_QUESTION = "question"
_DEFAULT_COL_CANDIDATES = "candidates"
_DEFAULT_COL_ANSWER  = "answer"
_DEFAULT_COL_SPLIT   = "split"
_DEFAULT_COL_NOTES   = "editability_notes"
_DEFAULT_COL_START   = "accurate_start"
_DEFAULT_COL_END     = "accurate_end"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _http_download(url: str, dest: Path, timeout: int = 180) -> None:
    """Stream-download url → dest atomically (writes to .tmp then renames)."""
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _hf_raw_url(repo: str, branch: str, path_in_repo: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{branch}/{path_in_repo}"


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_metadata_from_json(
    repo: str,
    branch: str,
    metadata_filename: str,
    cache_dir: Path,
) -> list[dict[str, Any]]:
    """
    Download <metadata_filename> from the HF repo (or use local cache) and
    return the list of row dicts.  Only fetches once per output directory.
    """
    cached = cache_dir / metadata_filename
    if cached.is_file():
        return json.loads(cached.read_text(encoding="utf-8"))

    url = _hf_raw_url(repo, branch, metadata_filename)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {metadata_filename} from {repo} …", end=" ", flush=True)
    _http_download(url, cached)
    print("done")
    data = json.loads(cached.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            f"{metadata_filename} must be a JSON array of row objects; got {type(data).__name__}"
        )
    return data


def load_metadata_from_parquet(
    repo: str,
    non_video_cols: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    Fallback: load the dataset via the HuggingFace datasets library.
    Only selects the columns we actually need so video-byte columns are skipped,
    avoiding the SIGSEGV crash that occurs when loading raw video data into RAM.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError(
            "The 'datasets' package is required for parquet fallback.\n"
            "  pip install datasets"
        )
    ds = load_dataset(repo, split="train", columns=non_video_cols)
    return list(ds)


# ---------------------------------------------------------------------------
# Video resolution
# ---------------------------------------------------------------------------

def resolve_video(
    col_value: str,
    repo: str,
    branch: str,
    videos_cache: Path,
) -> Path:
    """
    Turn the raw column value into a local file path, downloading if needed.

    Priority:
      1. col_value is an existing local path → use as-is.
      2. col_value starts with "http" → download from that URL.
      3. Anything else → treat as a repo-relative path, download from HF CDN.
    """
    # 1. Already a local file?
    local_direct = Path(col_value)
    if local_direct.is_file():
        return local_direct.resolve()

    # Derive a clean filename for caching (last path component).
    filename = Path(col_value).name
    local_cached = videos_cache / filename

    if local_cached.is_file():
        return local_cached

    videos_cache.mkdir(parents=True, exist_ok=True)

    # 2. Full URL?
    if col_value.startswith("http"):
        url = col_value
    else:
        # 3. Repo-relative path.
        url = _hf_raw_url(repo, branch, col_value)

    print(f"    Downloading {filename} …", end=" ", flush=True)
    t0 = time.time()
    _http_download(url, local_cached)
    elapsed = time.time() - t0
    kb = local_cached.stat().st_size // 1024
    print(f"done ({elapsed:.1f}s  {kb} KB)")
    return local_cached


# ---------------------------------------------------------------------------
# Candidate / wrong-option parsing
# ---------------------------------------------------------------------------

def parse_candidates(raw: Any) -> list[str]:
    """
    Normalise the candidates column value to a plain Python list of strings.
    Handles: list, JSON string, comma-separated string.
    """
    if isinstance(raw, list):
        return [str(c) for c in raw]
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("["):
            try:
                return [str(c) for c in json.loads(stripped)]
            except json.JSONDecodeError:
                pass
        return [c.strip() for c in re.split(r",\s*", stripped)]
    return [str(raw)]


def pick_wrong_option(candidates: list[str], answer: str) -> str:
    """Return the first candidate that does not match the correct answer."""
    ans_lower = answer.strip().lower()
    for c in candidates:
        if c.strip().lower() != ans_lower:
            return c
    return candidates[-1]


def _row_already_complete(output_dir: Path) -> bool:
    """
    True when a prior non-dry-run pipeline finished successfully.
    Dry-run manifests alone do not count (so production can be retried).
    """
    manifest_path = output_dir / "pipeline_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if manifest.get("dry_run"):
        return False
    return (output_dir / "final_spliced.mp4").is_file()


# ---------------------------------------------------------------------------
# sam3.py command builder
# ---------------------------------------------------------------------------

def _should_skip_sam3(split: str, editability_notes: Optional[str]) -> bool:
    """
    Return True when the row's edit type does not require SAM3 to locate an
    existing object — either because the target object must be ADDED (it is not
    yet in the clip) or because the edit is a global count change (clone/remove)
    that spans the whole scene rather than a single tracked object.

    Heuristics:
    • 'object_existence' rows whose editability_notes start with 'ADD edit' —
      the target object is absent, so SAM3 can never find it.
    • 'moving_count' rows — clone/removal edits must cover the full scene;
      segmenting one object instance is not meaningful.
    """
    notes_upper = (editability_notes or "").strip().upper()
    if split == "object_existence" and notes_upper.startswith("ADD EDIT"):
        return True
    if split == "moving_count":
        return True
    return False


def build_sam3_command(
    *,
    video_path: Path,
    question: str,
    answer: str,
    wrong_option: str,
    output_dir: Path,
    object_context: Optional[str],
    object_prompt: Optional[str],
    skip_sam3: bool,
    clip_start_sec: Optional[float],
    clip_end_sec: Optional[float],
    interval_pick: str,
    dry_run: bool,
    max_interval_sec: float,
    gemini_model: str,
    kling_endpoint: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "sam3.py"),
        "--video",            str(video_path),
        "--question",         question,
        "--answer",           answer,
        "--wrong-option",     wrong_option,
        "--output-dir",       str(output_dir),
        "--interval-pick",    interval_pick,
        "--max-segments",     "1",
        "--max-interval-sec", str(max_interval_sec),
        "--gemini-model",     gemini_model,
        "--kling-endpoint",   kling_endpoint,
    ]
    if object_context:
        cmd.extend(["--object-context", object_context])
    if object_prompt:
        cmd.extend(["--object-prompt", object_prompt])
    if skip_sam3:
        cmd.append("--skip-sam3")
    if clip_start_sec is not None:
        cmd.extend(["--clip-start-sec", str(clip_start_sec)])
    if clip_end_sec is not None:
        cmd.extend(["--clip-end-sec", str(clip_end_sec)])
    if dry_run:
        cmd.append("--dry-run")
    return cmd


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _record(
    progress: list[dict[str, Any]],
    path: Path,
    entry: dict[str, Any],
) -> None:
    """Append entry and write progress JSON atomically."""
    progress.append(entry)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    # ── Dataset source ──────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset",
        default=_DEFAULT_DATASET,
        metavar="USER/REPO",
        help=f"HuggingFace dataset repo id (default: {_DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--branch",
        default=_DEFAULT_BRANCH,
        help="Git branch / revision of the HF repo to download from (default: main).",
    )
    parser.add_argument(
        "--metadata-file",
        default=_DEFAULT_META_FILE,
        metavar="FILENAME",
        help=(
            "JSON file at the repo root that contains all row metadata as a list "
            f"(default: {_DEFAULT_META_FILE!r}).  Set to '' to skip and use the "
            "parquet-based fallback (requires the 'datasets' package)."
        ),
    )

    # ── Column mapping ───────────────────────────────────────────────────────
    parser.add_argument("--col-id",         default=_DEFAULT_COL_ID,
                        help=f"Row identifier column (default: {_DEFAULT_COL_ID!r}).")
    parser.add_argument("--col-video-path", default=_DEFAULT_COL_VIDEO,
                        help=f"Video path column (default: {_DEFAULT_COL_VIDEO!r}). "
                             "Value may be a local path, a full URL, or a repo-relative path.")
    parser.add_argument("--col-question",   default=_DEFAULT_COL_QUESTION,
                        help=f"Question column (default: {_DEFAULT_COL_QUESTION!r}).")
    parser.add_argument("--col-candidates", default=_DEFAULT_COL_CANDIDATES,
                        help=f"MCQ candidates column — list or JSON string "
                             f"(default: {_DEFAULT_COL_CANDIDATES!r}).")
    parser.add_argument("--col-answer",     default=_DEFAULT_COL_ANSWER,
                        help=f"Correct answer column (default: {_DEFAULT_COL_ANSWER!r}).")
    parser.add_argument("--col-split",      default=_DEFAULT_COL_SPLIT,
                        help=f"Optional split/category column (default: {_DEFAULT_COL_SPLIT!r}).")
    parser.add_argument("--col-notes",      default=_DEFAULT_COL_NOTES,
                        help=f"Optional edit-hint column → sam3 --object-context "
                             f"(default: {_DEFAULT_COL_NOTES!r}).")
    parser.add_argument("--col-start",      default=_DEFAULT_COL_START,
                        help=f"Optional clip-window start column in seconds "
                             f"(default: {_DEFAULT_COL_START!r}).")
    parser.add_argument("--col-end",        default=_DEFAULT_COL_END,
                        help=f"Optional clip-window end column in seconds "
                             f"(default: {_DEFAULT_COL_END!r}).")

    # ── Filtering ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--split-filter",
        default=None,
        metavar="VALUE",
        help="Only process rows where --col-split equals this value.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N rows (useful for smoke tests).",
    )

    # ── Output / resume ──────────────────────────────────────────────────────
    parser.add_argument(
        "--out-dir",
        default="hf_conflict_outputs",
        help="Root directory for per-row outputs (default: hf_conflict_outputs/).",
    )
    parser.add_argument(
        "--videos-dir",
        default=None,
        help="Cache directory for downloaded videos (default: <out-dir>/hf_videos/).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip rows that already finished a non-dry-run pipeline "
            "(pipeline_manifest.json with dry_run=false and final_spliced.mp4 present)."
        ),
    )

    # ── sam3.py settings ─────────────────────────────────────────────────────
    parser.add_argument(
        "--interval-pick",
        choices=["first", "last", "middle", "longest"],
        default="middle",
        help="Which SAM3 presence interval to edit (default: middle).",
    )
    parser.add_argument(
        "--max-interval-sec",
        type=float,
        default=5.0,
        help="Cap for the Kling edit interval in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run to sam3.py — mock SAM3/Kling, no API cost.",
    )
    parser.add_argument(
        "--models-config",
        type=Path,
        default=DEFAULT_MODELS_PATH,
        help="Path to models.json (default model = first enabled entry).",
    )
    parser.add_argument(
        "--gemini-model",
        default=None,
        help="OpenRouter model for edit-prompt generation. "
             "Default: first enabled model in --models-config.",
    )
    parser.add_argument(
        "--kling-endpoint",
        default="fal-ai/kling-video/o1/standard/video-to-video/edit",
        help="fal.ai Kling model endpoint.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream sam3.py output live (default: capture to per-row sam3_run.log).",
    )

    args = parser.parse_args()

    # Resolve model default
    gemini_model = args.gemini_model or default_openrouter_model(args.models_config)
    if not args.gemini_model:
        print(f"Edit-prompt model (from models.json): {gemini_model}")

    out_root   = Path(args.out_dir).resolve()
    videos_dir = Path(args.videos_dir).resolve() if args.videos_dir else out_root / "hf_videos"
    meta_cache = out_root / ".meta_cache"
    for d in (out_root, videos_dir, meta_cache):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load row metadata
    # ------------------------------------------------------------------
    repo   = args.dataset
    branch = args.branch
    print(f"\nLoading metadata from {repo} …")

    rows: list[dict[str, Any]]
    if args.metadata_file:
        try:
            rows = load_metadata_from_json(repo, branch, args.metadata_file, meta_cache)
        except Exception as exc:
            print(
                f"  metadata.json load failed ({exc}), falling back to parquet …",
                file=sys.stderr,
            )
            cols_needed = [
                c for c in [
                    args.col_id, args.col_video_path, args.col_question,
                    args.col_candidates, args.col_answer, args.col_split,
                    args.col_notes, args.col_start, args.col_end,
                ] if c
            ]
            rows = load_metadata_from_parquet(repo, non_video_cols=cols_needed or None)
    else:
        cols_needed = [
            c for c in [
                args.col_id, args.col_video_path, args.col_question,
                args.col_candidates, args.col_answer, args.col_split,
                args.col_notes, args.col_start, args.col_end,
            ] if c
        ]
        rows = load_metadata_from_parquet(repo, non_video_cols=cols_needed or None)

    print(f"  Total rows: {len(rows)}")

    if args.split_filter:
        split_col = args.col_split
        rows = [r for r in rows if str(r.get(split_col, "")) == args.split_filter]
        print(f"  After --split-filter={args.split_filter!r}: {len(rows)} rows")

    if args.limit:
        rows = rows[: args.limit]
        print(f"  After --limit={args.limit}: {len(rows)} rows")

    if not rows:
        print("No rows to process.")
        return

    # ------------------------------------------------------------------
    # Progress log
    # ------------------------------------------------------------------
    progress_path = out_root / "run_progress.json"
    progress: list[dict[str, Any]] = []
    if progress_path.is_file():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception:
            progress = []

    stats: dict[str, int] = {"ok": 0, "failed": 0, "error": 0, "skipped": 0}

    # ------------------------------------------------------------------
    # Per-row processing
    # ------------------------------------------------------------------
    for i, row in enumerate(rows):
        # ── Extract fields using configured column names ─────────────────
        row_id    = str(row.get(args.col_id, f"row_{i:04d}"))
        vid_path_val = str(row.get(args.col_video_path, "") or "")
        question  = str(row.get(args.col_question, "") or "")
        raw_cands = row.get(args.col_candidates)
        answer    = str(row.get(args.col_answer, "") or "")
        split_val = str(row.get(args.col_split, "") or "") if args.col_split else ""
        notes_val = row.get(args.col_notes) if args.col_notes else None
        start_val = row.get(args.col_start) if args.col_start else None
        end_val   = row.get(args.col_end)   if args.col_end   else None

        editability_notes: Optional[str] = str(notes_val).strip() or None if notes_val else None
        clip_start: Optional[float] = float(start_val) if start_val is not None else None
        clip_end:   Optional[float] = float(end_val)   if end_val   is not None else None

        print(f"\n{'─'*60}")
        print(f"[{i+1}/{len(rows)}] {row_id}" + (f"  ({split_val})" if split_val else ""))
        print(f"  video    : {vid_path_val}")
        print(f"  question : {question}")
        print(f"  answer   : {answer!r}")

        if not vid_path_val or not question:
            print("  ERROR: missing video path or question — skipping row.", file=sys.stderr)
            _record(progress, progress_path, {
                "id": row_id, "status": "error", "error": "missing video_path or question",
            })
            stats["error"] += 1
            continue

        output_dir    = out_root / row_id
        manifest_path = output_dir / "pipeline_manifest.json"
        log_path      = output_dir / "sam3_run.log"

        # ── skip-existing ────────────────────────────────────────────────
        if args.skip_existing and _row_already_complete(output_dir):
            print("  → Skipping (final_spliced.mp4 already exists from a prior run)")
            stats["skipped"] += 1
            continue

        # ── candidates / wrong option ────────────────────────────────────
        candidates = parse_candidates(raw_cands) if raw_cands is not None else []
        if candidates:
            wrong_option = pick_wrong_option(candidates, answer)
            print(f"  candidates   : {candidates}")
            print(f"  wrong-option : {wrong_option!r}")
        else:
            # Dataset doesn't have a candidates column — ask sam3.py to derive
            # wrong-option from the question itself via the LLM.
            wrong_option = answer  # placeholder; sam3 will overwrite via LLM if needed
            print("  No candidates column — wrong-option will be LLM-derived by sam3.py")

        if editability_notes:
            print(f"  edit hint    : {editability_notes}")
        if clip_start is not None and clip_end is not None:
            print(f"  clip window  : {clip_start:.2f}s – {clip_end:.2f}s")

        # ── resolve / download video ─────────────────────────────────────
        try:
            video_path = resolve_video(vid_path_val, repo, branch, videos_dir)
        except Exception as exc:
            print(f"  ERROR fetching video: {exc}", file=sys.stderr)
            _record(progress, progress_path, {
                "id": row_id, "split": split_val, "video": vid_path_val,
                "status": "error", "error": str(exc),
            })
            stats["error"] += 1
            continue

        # ── decide whether to skip SAM3 for this row ─────────────────────
        skip_sam3 = _should_skip_sam3(split_val, editability_notes)
        if skip_sam3:
            print(f"  skip-sam3: True (split={split_val!r}, edit type requires full-clip)")

        # ── derive SAM3 object prompt from dataset metadata ──────────────
        object_prompt: Optional[str] = None
        if not skip_sam3 and _derive_object_prompt is not None:
            object_prompt = _derive_object_prompt(
                question, answer=answer, object_context=editability_notes
            )
            if object_prompt:
                print(f"  object-prompt: {object_prompt!r} (from dataset metadata)")

        # ── build and run sam3.py ────────────────────────────────────────
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_sam3_command(
            video_path=video_path,
            question=question,
            answer=answer,
            wrong_option=wrong_option,
            output_dir=output_dir,
            object_context=editability_notes,
            object_prompt=object_prompt,
            skip_sam3=skip_sam3,
            clip_start_sec=clip_start,
            clip_end_sec=clip_end,
            interval_pick=args.interval_pick,
            dry_run=args.dry_run,
            max_interval_sec=args.max_interval_sec,
            gemini_model=gemini_model,
            kling_endpoint=args.kling_endpoint,
        )

        print("  Running sam3.py …")
        t0 = time.time()
        try:
            if args.verbose:
                proc = subprocess.run(cmd, text=True)
            else:
                with log_path.open("w", encoding="utf-8") as fh:
                    proc = subprocess.run(cmd, text=True, stdout=fh, stderr=subprocess.STDOUT)
            elapsed = time.time() - t0
            has_final = (output_dir / "final_spliced.mp4").is_file()
            if proc.returncode == 0 and has_final:
                status = "ok"
            elif proc.returncode == 0:
                status = "failed"
                print(
                    "  sam3.py exited 0 but final_spliced.mp4 is missing "
                    "(Kling/edit step failed) — marking failed.",
                    file=sys.stderr,
                )
            else:
                status = "failed"
        except Exception as exc:
            elapsed = time.time() - t0
            status  = "error"
            proc    = None
            print(f"  ERROR launching sam3.py: {exc}", file=sys.stderr)

        print(
            f"  Status: {status}  ({elapsed:.1f}s)"
            + ("" if args.verbose else f"   log → {log_path.name}")
        )

        # On failure, tail the log so problems are visible without --verbose.
        if status == "failed" and not args.verbose and log_path.is_file():
            lines = log_path.read_text(encoding="utf-8").splitlines()
            print("  ── last lines of sam3.py output ──")
            for ln in lines[-20:]:
                print(f"    {ln}")

        _record(progress, progress_path, {
            "id":           row_id,
            "split":        split_val,
            "video":        vid_path_val,
            "wrong_option": wrong_option,
            "status":       status,
            "elapsed_sec":  round(elapsed, 1),
            "output_dir":   str(output_dir),
            "log":          str(log_path) if not args.verbose else None,
            "returncode":   proc.returncode if proc is not None else None,
        })
        stats[status] += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Dataset : {repo}")
    print(f"Rows    : {len(rows)} processed")
    print(
        f"  ok={stats['ok']}  failed={stats['failed']}  "
        f"error={stats['error']}  skipped={stats['skipped']}"
    )
    print(f"Progress log : {progress_path}")
    print(f"Outputs root : {out_root}")


if __name__ == "__main__":
    main()
