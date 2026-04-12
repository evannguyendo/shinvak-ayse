import argparse
import os
import base64
import zipfile
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import instructor
import litellm
from openai import OpenAI
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from pydantic import BaseModel, Field, ConfigDict, field_validator


_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)


def _openrouter_api_key() -> str:
    key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            f"Add OPENROUTER_API_KEY=<your key> to {_PROJECT_ROOT / '.env'} "
            "or export it in your shell."
        )
    return key


_OPENROUTER_API_KEY = _openrouter_api_key()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
APP_URL = "https://github.com/evannguyendo"
APP_NAME = "MVBench Eval"

# Provider prefixes that LiteLLM recognises natively (not OpenRouter pass-throughs).
# Any model string that does NOT start with one of these will be auto-prefixed with
# "openrouter/" so LiteLLM routes it through OpenRouter.
_LITELLM_NATIVE_PREFIXES: frozenset[str] = frozenset({
    "gemini/", "anthropic/", "openai/", "vertex_ai/", "azure/",
    "groq/", "mistral/", "cohere/", "together_ai/", "huggingface/",
    "ollama/", "bedrock/", "openrouter/",
})


def _litellm_model_name(model: str) -> str:
    """Prepend 'openrouter/' when the model string has no LiteLLM provider prefix."""
    if any(model.startswith(p) for p in _LITELLM_NATIVE_PREFIXES):
        return model
    return f"openrouter/{model}"


def build_instructor_client(backend: str) -> instructor.Instructor:
    """Return an Instructor-patched client for the requested backend.

    backend="openrouter"  – thin OpenAI-compatible client pointed at OpenRouter.
    backend="litellm"     – LiteLLM completion function (supports 100+ providers).
    """
    if backend == "litellm":
        # Expose the key so LiteLLM can reach OpenRouter (and other providers).
        os.environ.setdefault("OPENROUTER_API_KEY", _OPENROUTER_API_KEY)
        litellm.drop_params = True   # silently drop unknown provider params
        return instructor.from_litellm(litellm.completion, mode=instructor.Mode.JSON)

    # Default: openrouter via the OpenAI-compatible endpoint
    return instructor.from_openai(
        OpenAI(
            api_key=_OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            default_headers={"HTTP-Referer": APP_URL, "X-Title": APP_NAME},
        ),
        mode=instructor.Mode.JSON,
    )

TASK_VIDEO_SOURCES: Dict[str, Dict[str, str]] = {
    "action_sequence":     {"type": "zip", "file": "video/star.zip"},
    "action_antonym":      {"type": "zip", "file": "video/ssv2_video.zip"},
    "action_count":        {"type": "zip", "file": "video/perception.zip"},
    "action_localization": {"type": "zip", "file": "video/sta.zip"},
}

_SEP = "=" * 60


class VideoEvalOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chain_of_thought: str = Field(
        min_length=1,
        description=(
            "Step-by-step reasoning: what happens in the video and why it supports "
            "exactly one of the given answer choices."
        ),
    )
    choice_index: int = Field(
        description="Index of the chosen option only: 0 = first listed choice, 1 = second, etc.",
    )
    confidence_score: int = Field(ge=1, le=10, description="Integer confidence from 1 to 10 only.")

    @field_validator("choice_index", mode="before")
    @classmethod
    def coerce_choice_index(cls, v: Any) -> int:
        if v is None:
            raise ValueError("choice_index is required (0 .. N-1 for the listed options).")
        try:
            return int(round(float(v)))
        except (TypeError, ValueError) as e:
            raise ValueError("choice_index must be an integer.") from e

    @field_validator("confidence_score", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> int:
        """Models sometimes return out-of-range numbers; coerce and clamp to 1–10."""
        if v is None:
            return 5
        try:
            n = int(round(float(v)))
        except (TypeError, ValueError):
            return 5
        return max(1, min(10, n))


def answer_from_choice_index(choice_index: int, candidates: List[str]) -> str:
    if not candidates:
        raise RuntimeError("No candidates provided for this question.")
    if not (0 <= choice_index < len(candidates)):
        raise RuntimeError(
            f"choice_index={choice_index} is invalid for {len(candidates)} options: {candidates!r}"
        )
    return candidates[choice_index]


def encode_video_to_data_url(video_path: Path) -> str:
    """Encode the full video file as a base64 data URL."""
    mime = {
        ".mp4": "video/mp4", ".mov": "video/quicktime",
        ".webm": "video/webm", ".avi": "video/x-msvideo",
    }.get(video_path.suffix.lower(), "video/mp4")
    return f"data:{mime};base64,{base64.b64encode(video_path.read_bytes()).decode()}"


def ask_video_question(
    *,
    client: instructor.Instructor,
    model: str,
    backend: str,
    video_path: Path,
    question: str,
    candidates: List[str],
    max_tokens: int = 8192,
    temperature: float = 0.1,
) -> VideoEvalOutput:
    last = len(candidates) - 1
    numbered = "\n".join(f"  [{i}] {opt}" for i, opt in enumerate(candidates))

    resolved_model = _litellm_model_name(model) if backend == "litellm" else model

    video_content: List[Dict[str, Any]] = [
        {"type": "video_url", "video_url": {"url": encode_video_to_data_url(video_path)}}
    ]

    prompt = (
        f"You are an expert video analysis AI. You are given a full video clip. "
        f"Answer the multiple-choice question based only on what is visually observable.\n\n"
        f"Question:\n{question}\n\n"
        f"Answer choices (select by index only):\n{numbered}\n\n"
        f"Instructions:\n"
        f"1. In \"chain_of_thought\", explain step by step what you observe and how it leads to one choice.\n"
        f"2. Set \"choice_index\" to the integer index (0–{last}) of your answer.\n\n"
        f"Rules:\n"
        f"- Base everything only on visual observation.\n"
        f"- choice_index must be 0–{last} inclusive.\n"
        f"- confidence_score is an integer 1–10 (not a percentage)."
    )

    # Instructor enforces the VideoEvalOutput schema via JSON mode for both backends.
    return client.chat.completions.create(
        model=resolved_model,
        response_model=VideoEvalOutput,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}] + video_content}],
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=3,
    )


def _index_zip(zip_path: str) -> Dict[str, str]:
    """Build a {basename → member_path} index once so video lookups are O(1)."""
    with zipfile.ZipFile(zip_path, "r") as z:
        return {os.path.basename(m): m for m in z.namelist() if not m.endswith("/")}


def get_video_from_zip(
    zip_path: str, zip_index: Dict[str, str], video_name: str, temp_dir: str
) -> Optional[str]:
    member = zip_index.get(os.path.basename(video_name))
    if member is None:
        return None
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extract(member, temp_dir)
    return os.path.join(temp_dir, member)


def get_video_from_ssv2(video_name: str) -> str:
    return hf_hub_download(
        repo_id="OpenGVLab/MVBench",
        repo_type="dataset",
        filename=f"video/ssv2_video_mp4/{os.path.basename(video_name)}",
    )


def _print_sample_header(
    idx: int, video_name: str, question: str, candidates: List[str], ground_truth: str
) -> None:
    print(f"\n  --- Sample {idx + 1} ---")
    print(f"  Video     : {video_name}")
    print(f"  Question  : {question}")
    print(f"  Choices   : {candidates}")
    print(f"  Truth     : {ground_truth}")


def _print_model_output(result: VideoEvalOutput, candidates: List[str], ground_truth: str) -> bool:
    predicted = answer_from_choice_index(result.choice_index, candidates)
    correct = predicted.strip().lower() == ground_truth.strip().lower()
    print(f"  Chain of thought:\n    {result.chain_of_thought}")
    print(f"  Chosen answer: [{result.choice_index}] {predicted}")
    print(f"  Confidence: {result.confidence_score}/10")
    print(f"  Result    : {'✅ CORRECT' if correct else '❌ INCORRECT'}")
    return correct


# Type alias for a job tuple
_Job = Tuple[int, str, str, List[str], str, str]


def evaluate_task(
    task_name: str,
    source_cfg: Dict[str, str],
    model: str,
    backend: str = "openrouter",
    num_samples: int = 3,
    max_tokens: int = 8192,
    max_workers: int = 1,
) -> Tuple[int, int]:
    print(f"\n{_SEP}\nTASK: {task_name}  [backend={backend}]\n{_SEP}")

    client = build_instructor_client(backend)
    dataset = load_dataset("OpenGVLab/MVBench", task_name, split="train")
    temp_dir = tempfile.mkdtemp()
    correct_predictions = 0
    total_processed = 0

    try:
        zip_path: Optional[str] = None
        zip_index: Dict[str, str] = {}
        if source_cfg["type"] == "zip":
            print(f"  Downloading/verifying archive: {source_cfg['file']} ...")
            zip_path = hf_hub_download(
                repo_id="OpenGVLab/MVBench",
                repo_type="dataset",
                filename=source_cfg["file"],
            )
            zip_index = _index_zip(zip_path)

        jobs: List[_Job] = []
        for i in range(min(num_samples, len(dataset))):
            row = dataset[i]
            video_name: str = row["video"]
            local_path = (
                get_video_from_zip(zip_path, zip_index, video_name, temp_dir)
                if source_cfg["type"] == "zip"
                else get_video_from_ssv2(video_name)
            )
            if not local_path or not os.path.exists(local_path):
                print(f"\n  --- Sample {i+1} ---\n  Video     : {video_name}\n  [SKIP] Video not found.")
                continue
            jobs.append((i, video_name, row["question"], row["candidates"], row["answer"], local_path))

        def _run_one(job: _Job) -> Tuple[int, Optional[str], Optional[VideoEvalOutput], str, str, List[str], str]:
            idx, vn, q, cand, gt, path = job
            try:
                result = ask_video_question(
                    client=client, model=model, backend=backend,
                    video_path=Path(path), question=q, candidates=cand,
                    max_tokens=max_tokens,
                )
                return (idx, None, result, vn, q, cand, gt)
            except Exception as e:
                return (idx, str(e), None, vn, q, cand, gt)

        if not jobs:
            return correct_predictions, total_processed

        if max_workers <= 1:
            for job in jobs:
                idx, vn, q, cand, gt, _ = job
                _print_sample_header(idx, vn, q, cand, gt)
                print(f"  Querying {backend} ({model})...")
                _, err, ev, *_ = _run_one(job)
                if err or ev is None:
                    print(f"  [ERROR] {err}")
                else:
                    if _print_model_output(ev, cand, gt):
                        correct_predictions += 1
                    total_processed += 1
        else:
            print(f"  Querying {backend} ({model}) on {len(jobs)} sample(s) (up to {max_workers} concurrent)...")
            collected: Dict[int, Tuple[Optional[str], Optional[VideoEvalOutput], str, str, List[str], str]] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_map = {pool.submit(_run_one, job): job for job in jobs}
                for fut in as_completed(future_map):
                    job = future_map[fut]
                    try:
                        idx, err, ev, vn, q, cand, gt = fut.result()
                    except Exception as e:
                        idx, err, ev, vn, q, cand, gt = job[0], str(e), None, job[1], job[2], job[3], job[4]
                    collected[idx] = (err, ev, vn, q, cand, gt)

            for idx in sorted(collected):
                err, ev, vn, q, cand, gt = collected[idx]
                _print_sample_header(idx, vn, q, cand, gt)
                if err or ev is None:
                    print(f"  [ERROR] {err}")
                else:
                    if _print_model_output(ev, cand, gt):
                        correct_predictions += 1
                    total_processed += 1

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return correct_predictions, total_processed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MVBench evaluation via OpenRouter (video + multiple choice)."
    )
    parser.add_argument(
        "--model", default="google/gemini-3.1-pro-preview",
        help="OpenRouter model id (video-capable, e.g. google/gemini-3.1-pro-preview).",
    )
    parser.add_argument(
        "--num-samples", type=int, default=3,
        help="Number of dataset rows to evaluate per task.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=8192,
        help="Max completion tokens. Reasoning models consume most tokens internally; 8192 is a safe floor.",
    )
    parser.add_argument(
        "--max-workers", type=int, default=1,
        help="Concurrent OpenRouter calls per task (1 = sequential). Try 2–3 to overlap slow uploads.",
    )
    parser.add_argument(
        "--backend", default="openrouter", choices=["openrouter", "litellm"],
        help=(
            "Inference backend for structured JSON output via Instructor. "
            "'openrouter' uses the OpenAI-compatible client pointed at openrouter.ai. "
            "'litellm' uses LiteLLM (supports 100+ providers; OpenRouter models are "
            "auto-prefixed with 'openrouter/' if no provider prefix is detected)."
        ),
    )
    args = parser.parse_args()

    results: Dict[str, Tuple[int, int]] = {}
    for task_name, source_cfg in TASK_VIDEO_SOURCES.items():
        try:
            results[task_name] = evaluate_task(
                task_name, source_cfg,
                model=args.model,
                backend=args.backend,
                num_samples=args.num_samples,
                max_tokens=args.max_tokens,
                max_workers=args.max_workers,
            )
        except Exception as e:
            print(f"\n  [TASK ERROR] {task_name}: {e}")
            results[task_name] = (0, 0)

    print(f"\n{_SEP}\nEVALUATION SUMMARY\n{_SEP}")
    overall_correct = overall_total = 0

    for task_name, (correct, total) in results.items():
        if total > 0:
            acc = correct / total * 100
            marks = "✅" * correct + "❌" * (total - correct)
            print(f"  {task_name:<25} {acc:5.1f}%  ({correct}/{total})  {marks}")
        else:
            print(f"  {task_name:<25}  N/A   (0 samples processed)")
        overall_correct += correct
        overall_total += total

    if overall_total > 0:
        overall_acc = overall_correct / overall_total * 100
        overall_marks = "✅" * overall_correct + "❌" * (overall_total - overall_correct)
        print(f"\n  {'OVERALL':<25} {overall_acc:5.1f}%  ({overall_correct}/{overall_total})  {overall_marks}")


if __name__ == "__main__":
    main()
