import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, ValidationError, field_validator
import re

from run_single_video_question import (
    QAResult,
    ask_video_question,
    encode_video_to_data_url,
)

TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}$")

# schema for the question
class QuestionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    question_text: str = Field(min_length=1)
    expected_answer_control: str = Field(min_length=1)
    expected_answer_conflict: str = Field(min_length=1)


class ExampleRecord(BaseModel):
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
                raise ValueError(
                    f"Invalid edit timestamp '{item}'. Expected MM:SS format."
                )
        return v


class ParsedResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    confidence: Literal["high", "medium", "low", "unknown"]
    evidence: List[str] = Field(default_factory=list)

    @field_validator("evidence")
    @classmethod
    def validate_evidence_timestamps(cls, v: List[str]) -> List[str]:
        for item in v:
            if not TIMESTAMP_RE.match(item):
                raise ValueError(
                    f"Invalid evidence timestamp '{item}'. Expected MM:SS format."
                )
        return v


# schema for the response
class ResponseRecord(BaseModel):
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

# timestamp for the run
def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

def make_run_id(
    *,
    model: str,
    video_id: str,
    variant: str,
    question_id: str,
    run_timestamp: str,
) -> str:
    safe_model = model.replace("/", "_").replace(":", "_")
    return f"{run_timestamp}__{safe_model}__{video_id}__{variant}__{question_id}"

# load examples, no duplicates videos or question ids
def load_examples_jsonl(path: str | Path) -> List[ExampleRecord]:
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
                raise RuntimeError(
                    f"Duplicate question_id in example '{example.video_id}'"
                )

            examples.append(example)

    return examples


def append_response_record(path: str | Path, record: ResponseRecord) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")


def build_response_record(
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
    run_timestamp = utc_now_iso()
    run_id = make_run_id(
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

# switch between control and conflict
def get_expected_answer(question: QuestionSpec, variant: str) -> str:
    if variant == "control":
        return question.expected_answer_control
    if variant == "conflict":
        return question.expected_answer_conflict
    raise ValueError(f"Unknown variant: {variant}")


# parse the response into the schema
def to_parsed_result(result: QAResult) -> ParsedResult:
    return ParsedResult.model_validate(result.model_dump())

# loop over variants, choose video, loop over questions, ask the question, build the response record, append the response record
def run_example(
    *,
    example: ExampleRecord,
    model: str,
    responses_path: str | Path,
    max_tokens: int = 1200,
    temperature: float = 0.1,
) -> None:
    for variant in ["control", "conflict"]:
        video_path = example.control_path if variant == "control" else example.conflict_path

        for question in example.questions:
            expected_answer = get_expected_answer(question, variant)

            try:
                video_data_url = encode_video_to_data_url(Path(video_path))
                result, raw_text = ask_video_question(
                    model=model,
                    video_data_url=video_data_url,
                    question=question.question_text,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                record = build_response_record(
                    model=model,
                    video_id=example.video_id,
                    variant=variant,
                    video_path=video_path,
                    conflict_type=example.conflict_type,
                    question_id=question.question_id,
                    question_text=question.question_text,
                    expected_answer=expected_answer,
                    raw_response_text=raw_text,
                    parsed_result=to_parsed_result(result),
                    status="ok",
                    error=None,
                )

            except RuntimeError as e:
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

                record = build_response_record(
                    model=model,
                    video_id=example.video_id,
                    variant=variant,
                    video_path=video_path,
                    conflict_type=example.conflict_type,
                    question_id=question.question_id,
                    question_text=question.question_text,
                    expected_answer=expected_answer,
                    raw_response_text=error_text[:4000],
                    parsed_result=None,
                    status=status,
                    error=error_text[:4000],
                )

            append_response_record(responses_path, record)
            print(
                f"[{record.status}] "
                f"model={model} video_id={example.video_id} "
                f"variant={variant} question_id={question.question_id}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", required=True, help="Path to examples.jsonl")
    parser.add_argument("--responses", required=True, help="Path to responses.jsonl")
    parser.add_argument("--model", required=True, help="Model name, e.g. google/gemini-2.5-pro")
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of examples to run",
    )
    args = parser.parse_args()

    examples = load_examples_jsonl(args.examples)

    if args.limit is not None:
        examples = examples[: args.limit]

    print(f"Loaded {len(examples)} examples")
    print(f"Running model: {args.model}")

    for example in examples:
        run_example(
            example=example,
            model=args.model,
            responses_path=args.responses,
        )

    print("Done.")


if __name__ == "__main__":
    main()


'''
python run_batch.py \
  --examples examples.jsonl \
  --responses responses.jsonl \
  --model google/gemini-2.5-pro
'''