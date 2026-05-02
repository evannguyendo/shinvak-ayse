import argparse
import base64
import json
import os
import re
import shutil
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import instructor
import litellm
import requests
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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
OPENROUTER_URL = f"{OPENROUTER_BASE_URL}/chat/completions"
APP_URL = "https://github.com/evannguyendo"
APP_NAME = "Video Benchmark"

DEFAULT_MODEL = "google/gemini-3-flash-preview"


def _openrouter_provider_extensions(model: str) -> Dict[str, Any]:
    """Extra OpenRouter JSON fields so Z.AI–hosted models (e.g. z-ai/glm-5v-turbo) use the z-ai provider.

    Without this, OpenRouter may route to disallowed hosts and return 404 for Z.AI-only models.
    LiteLLM passes models like openrouter/z-ai/... — treat any path segment z-ai as Z.AI.
    """
    m = (model or "").strip()
    if m.startswith("z-ai/") or "/z-ai/" in m:
        return {
            "provider": {
                "only": ["z-ai"],
                "allow_fallbacks": False,
            }
        }
    return {}

TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}$")

# Provider prefixes that LiteLLM recognises natively (not OpenRouter pass-throughs).
# Any model string that does NOT start with one of these will be auto-prefixed with
# "openrouter/" so LiteLLM routes it through OpenRouter.
_LITELLM_NATIVE_PREFIXES: frozenset[str] = frozenset({
    "gemini/", "anthropic/", "openai/", "vertex_ai/", "azure/",
    "groq/", "mistral/", "cohere/", "together_ai/", "huggingface/",
    "ollama/", "bedrock/", "openrouter/",
})

TASK_VIDEO_SOURCES: Dict[str, Dict[str, str]] = {
    "action_sequence":     {"type": "zip", "file": "video/star.zip"},
    "action_antonym":      {"type": "zip", "file": "video/ssv2_video.zip"},
    "action_count":        {"type": "zip", "file": "video/perception.zip"},
    "action_localization": {"type": "zip", "file": "video/sta.zip"},
}

_SEP = "=" * 60


def _litellm_model_name(model: str) -> str:
    """Prepend 'openrouter/' when the model string has no LiteLLM provider prefix."""
    if any(model.startswith(p) for p in _LITELLM_NATIVE_PREFIXES):
        return model
    return f"openrouter/{model}"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class VideoEvalOutput(BaseModel):
    """Structured output for MVBench multiple-choice evaluation."""

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


class QAResult(BaseModel):
    """Open-ended video Q&A result with confidence and timestamp evidence."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    confidence: Literal["high", "medium", "low", "unknown"]
    evidence: List[str] = Field(default_factory=list)

    @field_validator("evidence")
    @classmethod
    def validate_timestamps(cls, v: List[str]) -> List[str]:
        for item in v:
            if not TIMESTAMP_RE.match(item):
                raise ValueError(f"Invalid timestamp '{item}'. Expected MM:SS format.")
        return v


class QuestionSpec(BaseModel):
    """Single question specification for a temporal conflict benchmark example."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    question_text: str = Field(min_length=1)
    expected_answer_control: str = Field(min_length=1)
    expected_answer_conflict: str = Field(min_length=1)


class ExampleRecord(BaseModel):
    """A video pair (control + conflict) with associated Q&A specs."""

    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(min_length=1)
    control_path: str = Field(min_length=1)
    conflict_path: str = Field(min_length=1)
    conflict_type: str = Field(min_length=1)
    edited_object: str = Field(min_length=1)
    edit_timestamps: List[str] = Field(default_factory=list)
    before_edit: str = Field(min_length=1)
    after_edit: str = Field(min_length=1)
    questions: List[QuestionSpec] = Field(min_length=1)

    @field_validator("edit_timestamps")
    @classmethod
    def validate_edit_timestamps(cls, v: List[str]) -> List[str]:
        for item in v:
            if not TIMESTAMP_RE.match(item):
                raise ValueError(f"Invalid edit timestamp '{item}'. Expected MM:SS format.")
        return v


class ParsedResult(BaseModel):
    """Parsed model answer stored inside a ResponseRecord."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    confidence: Literal["high", "medium", "low", "unknown"]
    evidence: List[str] = Field(default_factory=list)

    @field_validator("evidence")
    @classmethod
    def validate_evidence_timestamps(cls, v: List[str]) -> List[str]:
        for item in v:
            if not TIMESTAMP_RE.match(item):
                raise ValueError(f"Invalid evidence timestamp '{item}'. Expected MM:SS format.")
        return v


class LightEvalOutput(BaseModel):
    """Lightweight single-inference output for a video + question evaluation."""

    model_config = ConfigDict(extra="ignore")

    answer: str = Field(min_length=1, description="The model's answer to the question.")
    reasoning: str = Field(min_length=1, description="Step-by-step reasoning behind the answer.")
    confidence: int = Field(ge=1, le=10, description="Confidence score 1–10.")

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> int:
        if v is None:
            return 5
        try:
            n = int(round(float(v)))
        except (TypeError, ValueError):
            return 5
        return max(1, min(10, n))


class ResponseRecord(BaseModel):
    """Full audit record for a single model response to one video/question/variant."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    run_timestamp: str = Field(min_length=1)
    model: str = Field(min_length=1)

    video_id: str = Field(min_length=1)
    variant: Literal["control", "conflict"]
    video_path: str = Field(min_length=1)
    conflict_type: str = Field(min_length=1)

    question_id: str = Field(min_length=1)
    question_text: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)

    raw_response_text: str = Field(min_length=1)
    parsed_result: Optional[ParsedResult] = None

    status: Literal["ok", "parse_error", "api_error", "timeout", "validation_error"]
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Client & API Utilities
# ---------------------------------------------------------------------------

def build_instructor_client(backend: str) -> instructor.Instructor:
    """Return an Instructor-patched client for the requested backend.

    backend="openrouter"  – thin OpenAI-compatible client pointed at OpenRouter.
    backend="litellm"     – LiteLLM completion function (supports 100+ providers).
    """
    if backend == "litellm":
        os.environ.setdefault("OPENROUTER_API_KEY", _OPENROUTER_API_KEY)
        litellm.drop_params = True
        return instructor.from_litellm(litellm.completion, mode=instructor.Mode.JSON)

    return instructor.from_openai(
        OpenAI(
            api_key=_OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            default_headers={"HTTP-Referer": APP_URL, "X-Title": APP_NAME},
        ),
        mode=instructor.Mode.JSON,
    )


def _openrouter_chat(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 1200,
    temperature: float = 0.1,
    response_format: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Low-level OpenRouter REST call used for interactive video analysis."""
    headers = {
        "Authorization": f"Bearer {_OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": APP_URL,
        "X-Title": APP_NAME,
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    payload.update(_openrouter_provider_extensions(model))
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=300)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter error {resp.status_code}\n{resp.text[:4000]}")
    return resp.json()


def _extract_message_text(response_json: Dict[str, Any]) -> str:
    """Extract plain text content from an OpenRouter chat completion response."""
    choices = response_json.get("choices", [])
    if not choices:
        raise RuntimeError(
            "OpenRouter returned no choices.\n"
            f"Response:\n{json.dumps(response_json, indent=2)[:4000]}"
        )
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


def _parse_json_text(text: str) -> Dict[str, Any]:
    """Parse JSON from model output, stripping markdown code fences if present."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
        return json.loads(cleaned)


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _make_run_id(
    *,
    model: str,
    video_id: str,
    variant: str,
    question_id: str,
    run_timestamp: str,
) -> str:
    safe_model = model.replace("/", "_").replace(":", "_")
    return f"{run_timestamp}__{safe_model}__{video_id}__{variant}__{question_id}"


# ---------------------------------------------------------------------------
# Video Utilities
# ---------------------------------------------------------------------------

def encode_video_to_data_url(video_path: Path) -> str:
    """Encode a local video file as a base64 data URL for API submission."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    mime = {
        ".mp4": "video/mp4", ".mov": "video/quicktime",
        ".webm": "video/webm", ".avi": "video/x-msvideo",
    }.get(video_path.suffix.lower(), "video/mp4")
    return f"data:{mime};base64,{base64.b64encode(video_path.read_bytes()).decode()}"


def _index_zip(zip_path: str) -> Dict[str, str]:
    """Build a {basename → member_path} index once so video lookups are O(1)."""
    with zipfile.ZipFile(zip_path, "r") as z:
        return {os.path.basename(m): m for m in z.namelist() if not m.endswith("/")}


def _get_video_from_zip(
    zip_path: str, zip_index: Dict[str, str], video_name: str, temp_dir: str
) -> Optional[str]:
    member = zip_index.get(os.path.basename(video_name))
    if member is None:
        return None
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extract(member, temp_dir)
    return os.path.join(temp_dir, member)


def _get_video_from_ssv2(video_name: str) -> str:
    return hf_hub_download(
        repo_id="OpenGVLab/MVBench",
        repo_type="dataset",
        filename=f"video/ssv2_video_mp4/{os.path.basename(video_name)}",
    )


# ---------------------------------------------------------------------------
# MVBench Evaluation
# ---------------------------------------------------------------------------

def _answer_from_choice_index(choice_index: int, candidates: List[str]) -> str:
    if not candidates:
        raise RuntimeError("No candidates provided for this question.")
    if not (0 <= choice_index < len(candidates)):
        raise RuntimeError(
            f"choice_index={choice_index} is invalid for {len(candidates)} options: {candidates!r}"
        )
    return candidates[choice_index]


def _print_sample_header(
    idx: int, video_name: str, question: str, candidates: List[str], ground_truth: str
) -> None:
    print(f"\n  --- Sample {idx + 1} ---")
    print(f"  Video     : {video_name}")
    print(f"  Question  : {question}")
    print(f"  Choices   : {candidates}")
    print(f"  Truth     : {ground_truth}")


def _print_model_output(result: VideoEvalOutput, candidates: List[str], ground_truth: str) -> bool:
    predicted = _answer_from_choice_index(result.choice_index, candidates)
    correct = predicted.strip().lower() == ground_truth.strip().lower()
    print(f"  Chain of thought:\n    {result.chain_of_thought}")
    print(f"  Chosen answer: [{result.choice_index}] {predicted}")
    print(f"  Confidence: {result.confidence_score}/10")
    print(f"  Result    : {'✅ CORRECT' if correct else '❌ INCORRECT'}")
    return correct


_MvBenchJob = Tuple[int, str, str, List[str], str, str]


def ask_video_question_mvbench(
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
    """Submit a multiple-choice video question to the model and return a structured output."""
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
    z_ai_extra = _openrouter_provider_extensions(resolved_model)
    create_kw: Dict[str, Any] = dict(
        model=resolved_model,
        response_model=VideoEvalOutput,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}] + video_content}],
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=3,
    )
    if z_ai_extra:
        create_kw["extra_body"] = z_ai_extra
    return client.chat.completions.create(**create_kw)


def evaluate_mvbench_task(
    task_name: str,
    source_cfg: Dict[str, str],
    model: str,
    backend: str = "openrouter",
    num_samples: int = 3,
    max_tokens: int = 8192,
    max_workers: int = 1,
) -> Tuple[int, int]:
    """Evaluate a single MVBench task and return (correct, total) counts."""
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

        jobs: List[_MvBenchJob] = []
        for i in range(min(num_samples, len(dataset))):
            row = dataset[i]
            video_name: str = row["video"]
            local_path = (
                _get_video_from_zip(zip_path, zip_index, video_name, temp_dir)
                if source_cfg["type"] == "zip"
                else _get_video_from_ssv2(video_name)
            )
            if not local_path or not os.path.exists(local_path):
                print(f"\n  --- Sample {i+1} ---\n  Video     : {video_name}\n  [SKIP] Video not found.")
                continue
            jobs.append((i, video_name, row["question"], row["candidates"], row["answer"], local_path))

        def _run_one(
            job: _MvBenchJob,
        ) -> Tuple[int, Optional[str], Optional[VideoEvalOutput], str, str, List[str], str]:
            idx, vn, q, cand, gt, path = job
            try:
                result = ask_video_question_mvbench(
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


# ---------------------------------------------------------------------------
# Temporal Conflict Benchmark
# ---------------------------------------------------------------------------

def _load_examples_jsonl(path: str | Path) -> List[ExampleRecord]:
    """Load and validate example records from a JSONL file, rejecting duplicates."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Examples file not found: {path}")

    examples: List[ExampleRecord] = []
    seen_video_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Invalid JSON in {path} at line {line_number}: {e}"
                ) from e
            try:
                example = ExampleRecord.model_validate(raw)
            except ValidationError as e:
                raise RuntimeError(
                    f"Example validation failed in {path} at line {line_number}:\n{e}"
                ) from e
            if example.video_id in seen_video_ids:
                raise RuntimeError(
                    f"Duplicate video_id '{example.video_id}' in {path} line {line_number}"
                )
            seen_video_ids.add(example.video_id)
            qids = [q.question_id for q in example.questions]
            if len(qids) != len(set(qids)):
                raise RuntimeError(f"Duplicate question_id in example '{example.video_id}'")
            examples.append(example)

    return examples


def _append_response_record(path: str | Path, record: ResponseRecord) -> None:
    """Append a ResponseRecord as a JSON line to the given file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")


def _build_response_record(
    *,
    model: str,
    video_id: str,
    variant: str,
    video_path: str,
    conflict_type: str,
    question_id: str,
    question_text: str,
    expected_answer: str,
    raw_response_text: str,
    parsed_result: Optional[ParsedResult],
    status: str,
    error: Optional[str],
) -> ResponseRecord:
    run_timestamp = _utc_now_iso()
    run_id = _make_run_id(
        model=model,
        video_id=video_id,
        variant=variant,
        question_id=question_id,
        run_timestamp=run_timestamp,
    )
    return ResponseRecord(
        run_id=run_id,
        run_timestamp=run_timestamp,
        model=model,
        video_id=video_id,
        variant=variant,
        video_path=video_path,
        conflict_type=conflict_type,
        question_id=question_id,
        question_text=question_text,
        expected_answer=expected_answer,
        raw_response_text=raw_response_text,
        parsed_result=parsed_result,
        status=status,
        error=error,
    )


def _get_expected_answer(question: QuestionSpec, variant: str) -> str:
    if variant == "control":
        return question.expected_answer_control
    if variant == "conflict":
        return question.expected_answer_conflict
    raise ValueError(f"Unknown variant: {variant}")


def ask_single_video_question(
    *,
    client: instructor.Instructor,
    model: str,
    backend: str,
    video_path: Path,
    question: str,
    max_tokens: int = 1200,
    temperature: float = 0.1,
) -> QAResult:
    """Ask an open-ended question about a video and return a structured QAResult."""
    resolved_model = _litellm_model_name(model) if backend == "litellm" else model
    video_content: List[Dict[str, Any]] = [
        {"type": "video_url", "video_url": {"url": encode_video_to_data_url(video_path)}}
    ]
    prompt = (
        "Answer the user's question about the video using only what is visually observable.\n\n"
        f"Question:\n{question}\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "answer": "your answer",\n'
        '  "confidence": "high|medium|low|unknown",\n'
        '  "evidence": ["timestamps in MM:SS format"]\n'
        "}\n\n"
        "Rules:\n"
        "- Base the answer only on the video.\n"
        "- Evidence must be timestamps in MM:SS format.\n"
        "- Do not include any other keys."
    )
    z_ai_extra = _openrouter_provider_extensions(resolved_model)
    create_kw = dict(
        model=resolved_model,
        response_model=QAResult,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}] + video_content}],
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=3,
    )
    if z_ai_extra:
        create_kw["extra_body"] = z_ai_extra
    return client.chat.completions.create(**create_kw)


def run_example(
    *,
    example: ExampleRecord,
    client: instructor.Instructor,
    model: str,
    backend: str,
    responses_path: str | Path,
    max_tokens: int = 1200,
    temperature: float = 0.1,
) -> None:
    """Run all variant/question combinations for one example and record results."""
    for variant in ["control", "conflict"]:
        video_path_str = example.control_path if variant == "control" else example.conflict_path

        for question in example.questions:
            expected_answer = _get_expected_answer(question, variant)

            try:
                result = ask_single_video_question(
                    client=client,
                    model=model,
                    backend=backend,
                    video_path=Path(video_path_str),
                    question=question.question_text,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                raw_text = result.model_dump_json()
                parsed = ParsedResult.model_validate(result.model_dump())
                record = _build_response_record(
                    model=model,
                    video_id=example.video_id,
                    variant=variant,
                    video_path=video_path_str,
                    conflict_type=example.conflict_type,
                    question_id=question.question_id,
                    question_text=question.question_text,
                    expected_answer=expected_answer,
                    raw_response_text=raw_text,
                    parsed_result=parsed,
                    status="ok",
                    error=None,
                )

            except Exception as e:
                error_text = str(e)
                lowered = error_text.lower()
                if "invalid json" in lowered or "validation failed" in lowered:
                    status = "parse_error"
                elif "timeout" in lowered:
                    status = "timeout"
                elif "openrouter error" in lowered:
                    status = "api_error"
                else:
                    status = "validation_error"
                record = _build_response_record(
                    model=model,
                    video_id=example.video_id,
                    variant=variant,
                    video_path=video_path_str,
                    conflict_type=example.conflict_type,
                    question_id=question.question_id,
                    question_text=question.question_text,
                    expected_answer=expected_answer,
                    raw_response_text=error_text[:4000],
                    parsed_result=None,
                    status=status,
                    error=error_text[:4000],
                )

            _append_response_record(responses_path, record)
            mark = ""
            if record.status == "ok" and record.parsed_result:
                correct = record.parsed_result.answer.strip().lower() in expected_answer.strip().lower() or \
                          expected_answer.strip().lower() in record.parsed_result.answer.strip().lower()
                mark = " ✅" if correct else " ❌"
            print(
                f"[{record.status}]{mark} "
                f"video={example.video_id} variant={variant} "
                f"q={question.question_id} model={model}"
            )


def run_batch_examples(
    *,
    examples_path: str | Path,
    responses_path: str | Path,
    model: str,
    backend: str = "openrouter",
    max_tokens: int = 1200,
    temperature: float = 0.1,
    limit: Optional[int] = None,
) -> None:
    """Load examples from a JSONL file and run the temporal conflict benchmark."""
    examples = _load_examples_jsonl(examples_path)
    if limit is not None:
        examples = examples[:limit]

    client = build_instructor_client(backend)
    print(f"Loaded {len(examples)} examples")
    print(f"Running model: {model}")

    for example in examples:
        run_example(
            example=example,
            client=client,
            model=model,
            backend=backend,
            responses_path=responses_path,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    print("Done.")


# ---------------------------------------------------------------------------
# Interactive Video Analysis
# ---------------------------------------------------------------------------

def analyze_video_init(*, model: str, video_path: Path) -> Dict[str, Any]:
    """Perform a structured initial analysis of a video: summary, timeline, objects, etc."""
    video_data_url = encode_video_to_data_url(video_path)
    prompt = (
        "Analyze this video and what happens over time. Focus on what is visually observable.\n\n"
        "Return ONLY valid JSON with this exact structure:\n"
        "{\n"
        '  "summary": "a concise summary of the video",\n'
        '  "timeline": [{"time": "approximate time or \'unknown\'", "event": "what happens"}],\n'
        '  "objects": ["important visible objects"],\n'
        '  "people": ["people or roles if present"],\n'
        '  "locations": ["visible settings"],\n'
        '  "uncertainties": ["things you are not sure about"]\n'
        "}"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video_url", "video_url": {"url": video_data_url}},
            ],
        }
    ]
    response = _openrouter_chat(
        model=model, messages=messages, max_tokens=1800,
        temperature=0.1,
    )
    text = _extract_message_text(response)
    return _parse_json_text(text)


def ask_question_interactive(
    *,
    model: str,
    video_path: Path,
    cached_analysis: Dict[str, Any],
    question: str,
) -> Dict[str, Any]:
    """Answer an interactive question about a video, using cached analysis as context."""
    video_data_url = encode_video_to_data_url(video_path)
    prompt = (
        "Answer the user's question about the video. "
        "You may use the cached analysis below as context.\n\n"
        f"Cached analysis:\n{json.dumps(cached_analysis, indent=2)}\n\n"
        f"Question:\n{question}\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "answer": "your answer",\n'
        '  "confidence": "high|medium|low|unknown",\n'
        '  "evidence": ["short evidence"]\n'
        "}"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video_url", "video_url": {"url": video_data_url}},
            ],
        }
    ]
    response = _openrouter_chat(
        model=model, messages=messages, max_tokens=1200,
        temperature=0.1,
    )
    text = _extract_message_text(response)
    return _parse_json_text(text)


def run_interactive_session(*, model: str, video_path: Path) -> None:
    """Launch an interactive Q&A session for a video, logging all exchanges to disk."""
    print(f"Reading video: {video_path}")
    print("Running initial analysis...\n")
    analysis = analyze_video_init(model=model, video_path=video_path)
    Path("video_analysis.json").write_text(json.dumps(analysis, indent=2))

    print("Summary:\n")
    print(analysis.get("summary", ""))
    print("\nReady for questions. Type 'exit' or 'quit' to stop.\n")

    while True:
        q = input("Question: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        result = ask_question_interactive(
            model=model,
            video_path=video_path,
            cached_analysis=analysis,
            question=q,
        )

        with open("qa_log.jsonl", "a") as f:
            f.write(json.dumps({"question": q, "result": result}) + "\n")

        print(f"\n{_SEP}")
        print(f"  Answer    : {result.get('answer', '')}")
        print(f"  Confidence: {result.get('confidence', '')}")
        evidence = result.get("evidence", [])
        if evidence:
            print(f"  Evidence  : {', '.join(evidence)}")
        print(_SEP + "\n")


# ---------------------------------------------------------------------------
# Multi-model Suite
# ---------------------------------------------------------------------------

def _load_models(path: str | Path) -> List[Dict[str, Any]]:
    """Load and return only enabled model configurations from a JSON file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    models = data.get("models", [])
    if not isinstance(models, list):
        raise RuntimeError("models.json must contain a top-level 'models' list")
    enabled = [m for m in models if m.get("enabled", True)]
    if not enabled:
        raise RuntimeError("No enabled models found in models config")
    return enabled


def run_suite(
    *,
    models_config: str | Path,
    examples_path: str | Path,
    out_dir: str | Path,
    backend: str = "openrouter",
    limit: Optional[int] = None,
) -> None:
    """Run the temporal conflict benchmark across all enabled models in a config file."""
    models = _load_models(models_config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(models)} enabled models")

    for model_cfg in models:
        model_id = model_cfg["id"]
        model_name = model_cfg["openrouter_model"]
        max_tokens = int(model_cfg.get("max_tokens", 1200))
        temperature = float(model_cfg.get("temperature", 0.1))

        model_dir = out_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        responses_path = model_dir / "responses.jsonl"

        print(f"\n{_SEP}\nRunning {model_id} ({model_name})\n{_SEP}")

        try:
            run_batch_examples(
                examples_path=examples_path,
                responses_path=responses_path,
                model=model_name,
                backend=backend,
                max_tokens=max_tokens,
                temperature=temperature,
                limit=limit,
            )
            print(f"[done] model={model_id} responses={responses_path}")
        except Exception as e:
            print(f"[failed] model={model_id} error={e}")


# ---------------------------------------------------------------------------
# Lightweight Evaluation  (single inference run, direct args)
# ---------------------------------------------------------------------------

def _answers_match(predicted: str, expected: str) -> bool:
    """Fuzzy match: True when either string contains the other (case-insensitive)."""
    p, e = predicted.strip().lower(), expected.strip().lower()
    return p == e or e in p or p in e


def run_evaluate(
    *,
    client: instructor.Instructor,
    model: str,
    backend: str,
    video_path: Path,
    question: str,
    options: Optional[List[str]] = None,
    expected: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.1,
) -> None:
    """Single inference call on one video — prints answer, reasoning, confidence, and accuracy if expected is given."""
    resolved_model = _litellm_model_name(model) if backend == "litellm" else model
    options_block = ""
    if options:
        options_block = "\n\nAnswer options:\n" + "\n".join(
            f"  [{i}] {o}" for i, o in enumerate(options)
        )

    prompt = (
        "Answer the following question about the video based solely on visual observation.\n\n"
        f"Question:\n{question}"
        f"{options_block}\n\n"
        "Rules:\n"
        "- Base your answer only on the video.\n"
        "- If options are provided, use the exact option text as your answer.\n"
        "- 'confidence' is an integer 1–10."
    )
    video_content: List[Dict[str, Any]] = [
        {"type": "video_url", "video_url": {"url": encode_video_to_data_url(video_path)}}
    ]
    z_ai_extra = _openrouter_provider_extensions(resolved_model)
    create_kw = dict(
        model=resolved_model,
        response_model=LightEvalOutput,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}] + video_content}],
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=3,
    )
    if z_ai_extra:
        create_kw["extra_body"] = z_ai_extra
    result = client.chat.completions.create(**create_kw)

    print(f"\n{_SEP}")
    print(f"  Video     : {video_path}")
    print(f"  Question  : {question}")
    if options:
        print(f"  Options   : {options}")
    print(f"\n  Answer    : {result.answer}")
    print(f"  Reasoning : {result.reasoning}")
    print(f"  Confidence: {result.confidence}/10")
    if expected is not None:
        correct = _answers_match(result.answer, expected)
        print(f"\n  Expected  : {expected}")
        print(f"  Retrieval : {'✅ CORRECT' if correct else '❌ INCORRECT'}")
    print(_SEP)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified video benchmark — runs MVBench evaluation by default.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""modes:
  mvbench     (default) Evaluate all MVBench tasks — multiple-choice, HuggingFace dataset
  evaluate    Single-inference eval on one video — answer + optional accuracy check
  batch       Run the temporal conflict benchmark on a JSONL examples file
  suite       Run batch across multiple models defined in a JSON config file
  interactive Launch an interactive Q&A session for a local video

examples:
  python unified_video_benchmark.py
  python unified_video_benchmark.py --mode evaluate --video data/conflict.mp4 --question "What color is the pen?"
  python unified_video_benchmark.py --mode evaluate --video data/conflict.mp4 --question "What color is the pen?" --options "blue|red|green" --expected "red"
  python unified_video_benchmark.py --mode batch --examples examples.jsonl --responses out.jsonl
  python unified_video_benchmark.py --mode suite --models-config models.json --examples examples.jsonl --out-dir runs/
  python unified_video_benchmark.py --mode interactive --video video.mp4

default model: {DEFAULT_MODEL}
""",
    )

    # Shared flags
    parser.add_argument(
        "--mode", default="mvbench",
        choices=["mvbench", "batch", "suite", "interactive", "evaluate"],
        help="Run mode (default: mvbench).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model id.")
    parser.add_argument(
        "--backend", default="openrouter", choices=["openrouter", "litellm"],
        help="Inference backend (default: openrouter).",
    )

    # mvbench flags
    parser.add_argument("--num-samples", type=int, default=3, help="[mvbench] Dataset rows per task.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max completion tokens.")
    parser.add_argument("--max-workers", type=int, default=1, help="[mvbench] Concurrent calls per task.")

    # evaluate / interactive flags
    parser.add_argument("--video", default=None, help="[evaluate/interactive] Path to video file.")
    parser.add_argument("--question", default=None, help="[evaluate] Question to ask.")
    parser.add_argument("--options", default=None, help="[evaluate] Pipe-separated answer options, e.g. 'blue|red|green'.")
    parser.add_argument("--expected", default=None, help="[evaluate] Expected answer — prints ✅/❌ correctness check.")

    # batch / suite flags
    parser.add_argument("--examples", default=None, help="[batch/suite] Path to examples.jsonl.")
    parser.add_argument("--responses", default=None, help="[batch] Path to output responses.jsonl.")
    parser.add_argument("--temperature", type=float, default=0.1, help="[batch] Sampling temperature.")
    parser.add_argument("--limit", type=int, default=None, help="[batch/suite] Max examples to run.")

    # suite flags
    parser.add_argument("--models-config", default=None, help="[suite] Path to models.json.")
    parser.add_argument("--out-dir", default=None, help="[suite] Directory for run outputs.")

    args = parser.parse_args()

    if args.mode == "mvbench":
        results: Dict[str, Tuple[int, int]] = {}
        for task_name, source_cfg in TASK_VIDEO_SOURCES.items():
            try:
                results[task_name] = evaluate_mvbench_task(
                    task_name, source_cfg,
                    model=args.model,
                    backend=args.backend,
                    num_samples=args.num_samples,
                    max_tokens=args.max_tokens or 8192,
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

    elif args.mode == "batch":
        if not args.examples or not args.responses:
            parser.error("--examples and --responses are required for --mode batch")
        run_batch_examples(
            examples_path=args.examples,
            responses_path=args.responses,
            model=args.model,
            backend=args.backend,
            max_tokens=args.max_tokens or 1200,
            temperature=args.temperature,
            limit=args.limit,
        )

    elif args.mode == "suite":
        if not args.models_config or not args.examples or not args.out_dir:
            parser.error("--models-config, --examples, and --out-dir are required for --mode suite")
        run_suite(
            models_config=args.models_config,
            examples_path=args.examples,
            out_dir=args.out_dir,
            backend=args.backend,
            limit=args.limit,
        )

    elif args.mode == "interactive":
        if not args.video:
            parser.error("--video is required for --mode interactive")
        run_interactive_session(model=args.model, video_path=Path(args.video))

    elif args.mode == "evaluate":
        if not args.video or not args.question:
            parser.error("--video and --question are required for --mode evaluate")
        client = build_instructor_client(args.backend)
        options = args.options.split("|") if args.options else None
        run_evaluate(
            client=client,
            model=args.model,
            backend=args.backend,
            video_path=Path(args.video),
            question=args.question,
            options=options,
            expected=args.expected,
            max_tokens=args.max_tokens or 2048,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()
