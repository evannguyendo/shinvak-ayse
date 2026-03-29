import os
import json
import base64
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODEL = "google/gemini-2.5-pro"

APP_URL = "http://localhost"
APP_NAME = "Temporal Conflict Demo"


def require_api_key() -> None:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")


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

    data = resp.json()

    return data


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

def analyze_video_init(
    *,
    model: str,
    video_data_url: str,
) -> Dict[str, Any]:
    prompt = """
Analyze this video and what happens over time. Focus on what is visually observable.

Return ONLY valid JSON with this exact structure:
{
  "summary": "a concise summary of the video",
  "timeline": [
    {
      "time": "approximate time or 'unknown'",
      "event": "what happens"
    }
  ],
  "objects": ["important visible objects"],
  "people": ["people or roles if present"],
  "locations": ["visible settings"],
  "uncertainties": ["things you are not sure about"]
}

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
        max_tokens=1800,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    text = extract_message_text(response)
    return parse_json_text(text)


def ask_question(
    *,
    model: str,
    video_data_url: str,
    cached_analysis: Dict[str, Any],
    question: str,
) -> Dict[str, Any]:
    prompt = f"""
Answer the user's question about the video. You may use the cached analysis below as context. 

Cached analysis:
{json.dumps(cached_analysis, indent=2)}

Question:
{question}

Return ONLY valid JSON:
{{
  "answer": "your answer",
  "confidence": "high|medium|low|unknown",
  "evidence": ["short evidence"],
}}
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
        max_tokens=1200,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    text = extract_message_text(response)
    return parse_json_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    require_api_key()

    video_path = Path(args.video)
    print(f"Reading video: {video_path}")

    video_data_url = encode_video_to_data_url(video_path)

    print("Running analysis...\n")
    analysis = analyze_video_init(
        model=args.model,
        video_data_url=video_data_url,
    )

    Path("video_analysis.json").write_text(
        json.dumps(analysis, indent=2)
    )

    print("Summary:\n")
    print(analysis.get("summary", ""))
    print("\nReady for questions.\n")

    while True:
        q = input("Question: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        result = ask_question(
            model=args.model,
            video_data_url=video_data_url,
            cached_analysis=analysis,
            question=q,
        )

        with open("qa_log.jsonl", "a") as f:
            f.write(json.dumps({
                "question": q,
                "result": result
            }) + "\n")

        print("\nAnswer:")
        print(result.get("answer", ""))
        print("Confidence:", result.get("confidence", ""))
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()