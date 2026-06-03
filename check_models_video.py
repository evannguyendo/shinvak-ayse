#!/usr/bin/env python3
"""Smoke-test each enabled model in models.json with one local video via run_single_video_question."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

from models_config import load_enabled_models
from run_single_video_question import (
    OPENROUTER_API_KEY,
    ask_video_question,
    encode_video_to_data_url,
    require_api_key,
)


def _short_error(exc: BaseException) -> str:
    msg = str(exc).replace("\n", " ")
    if len(msg) > 220:
        msg = msg[:220] + "..."
    return msg


def main() -> int:
    if not OPENROUTER_API_KEY:
        print("OPENROUTER_API_KEY is not set in .env")
        return 1

    video = _PROJECT_ROOT / "hf_conflict_outputs/moving_attribute_00/final_spliced.mp4"
    if not video.is_file():
        video = _PROJECT_ROOT / "hf_conflict_outputs/hf_videos/video_10929.mp4"
    if not video.is_file():
        print("No test video found under hf_conflict_outputs/")
        return 1

    question = "What shape is the cyan object that is moving?"
    print(f"Test video: {video}")
    print(f"Question: {question}\n")

    require_api_key()
    data_url = encode_video_to_data_url(video)

    results = []
    for cfg in load_enabled_models():
        model_id = cfg["id"]
        model = cfg["openrouter_model"]
        print(f"--- {model_id} ({model}) ---")
        try:
            parsed, raw = ask_video_question(
                model=model,
                video_data_url=data_url,
                question=question,
                max_tokens=400,
                temperature=0.1,
            )
            results.append(
                {
                    "id": model_id,
                    "model": model,
                    "runnable": True,
                    "status": "ok",
                    "answer": parsed.answer,
                }
            )
            print(f"  OK — answer: {parsed.answer!r}\n")
        except Exception as e:
            results.append(
                {
                    "id": model_id,
                    "model": model,
                    "runnable": False,
                    "status": "failed",
                    "error": _short_error(e),
                }
            )
            print(f"  FAIL — {_short_error(e)}\n")

    ok = sum(1 for r in results if r["runnable"])
    print("=" * 60)
    print(f"Runnable with video: {ok}/{len(results)}")
    for r in results:
        mark = "yes" if r["runnable"] else "no "
        print(f"  [{mark}] {r['id']}: {r['model']}")
        if not r["runnable"]:
            print(f"         {r.get('error', '')}")

    out = _PROJECT_ROOT / "runs" / "model_video_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
