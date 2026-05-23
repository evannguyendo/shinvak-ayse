import os
import json
import base64
import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar

import requests
# use Pydantic for validation and parsing of the response
from pydantic import BaseModel, Field, ConfigDict, ValidationError, field_validator

from models_config import default_openrouter_model

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODEL = default_openrouter_model()

APP_URL = "http://localhost"
APP_NAME = "Temporal Conflict Benchmark"

# to match MM:SS format for evidence later
TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}$")
ModelT = TypeVar("ModelT", bound=BaseModel)

# schema for the response
class QAResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    confidence: Literal["high", "medium", "low", "unknown"]
    evidence: List[str] = Field(default_factory=list)

    # validate the evidence timestamps
    @field_validator("evidence")
    @classmethod
    def validate_timestamps(cls, v: List[str]) -> List[str]:
        for item in v:
            if not TIMESTAMP_RE.match(item):
                raise ValueError(
                    f"Invalid timestamp '{item}'. Expected MM:SS format."
                )
        return v


def require_api_key() -> None:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")


# local video to data URL for OpenRouter
def encode_video_to_data_url(video_path: Path) -> str:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    suffix = video_path.suffix.lower()
    mime = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".avi": "video/x-msvideo",
    }.get(suffix, "video/mp4")

    video_bytes = video_path.read_bytes()
    encoded = base64.b64encode(video_bytes).decode("utf-8")
    return f"data:{mime};base64,{encoded}"

# normalize the response text from OpenRouter - get text
def extract_message_text(response_json: Dict[str, Any]) -> str:
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


def openrouter_chat(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 1200,
    temperature: float = 0.1,
    response_format: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
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

    resp = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=300,
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"OpenRouter error {resp.status_code}\n{resp.text[:4000]}"
        )

    return resp.json()

# for JSON formatting - could possibly use Instructor library later if failing to parse
def parse_json_text(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = text.strip()

        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

        return json.loads(cleaned)

# validate the response against the schema
def validate_model(model_cls: Type[ModelT], text: str) -> ModelT:
    try:
        data = parse_json_text(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            "Model returned invalid JSON.\n"
            f"Raw text:\n{text[:4000]}"
        ) from e

    try:
        return model_cls.model_validate(data)
    except ValidationError as e:
        raise RuntimeError(
            f"{model_cls.__name__} validation failed.\n"
            f"Parsed data:\n{json.dumps(data, indent=2)[:4000]}\n\n"
            f"Validation error:\n{e}"
        ) from e


def ask_video_question(
    *,
    model: str,
    video_data_url: str,
    question: str,
    max_tokens: int = 1200,
    temperature: float = 0.1,
) -> tuple[QAResult, str]:
    prompt = f"""
Answer the user's question about the video using only what is visually observable.

Question:
{question}

Return ONLY valid JSON:
{{
  "answer": "your answer",
  "confidence": "high|medium|low|unknown",
  "evidence": ["timestamps in MM:SS format"]
}}

Rules:
- Base the answer only on the video
- Evidence must be timestamps in MM:SS format
- Do not include any other keys
""".strip()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "video_url",
                    "video_url": {"url": video_data_url},
                },
            ],
        }
    ]

    response = openrouter_chat(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"},
    )

    raw_text = extract_message_text(response)
    parsed = validate_model(QAResult, raw_text)
    return parsed, raw_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--question", required=True, help="Question to ask")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model name")
    args = parser.parse_args()

    require_api_key()

    video_path = Path(args.video)
    print(f"Reading video: {video_path}")

    video_data_url = encode_video_to_data_url(video_path)

    result, raw_text = ask_video_question(
        model=args.model,
        video_data_url=video_data_url,
        question=args.question,
    )

    print("\nRaw response:")
    print(raw_text)

    print("\nParsed result:")
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()

'''
python run_single_video_question.py \
  --video data/control.mp4 \
  --question "What color is the pen?" \
  --model google/gemini-2.5-pro
'''