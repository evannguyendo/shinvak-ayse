"""Load enabled OpenRouter models from models.json (shared default for scripts)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_PATH = _PROJECT_ROOT / "models.json"
_FALLBACK_MODEL = "google/gemini-3-flash-preview"


def load_enabled_models(path: Path | None = None) -> list[dict[str, Any]]:
    """Return enabled model entries from models.json."""
    cfg = path or DEFAULT_MODELS_PATH
    with cfg.open("r", encoding="utf-8") as f:
        data = json.load(f)

    models = data.get("models", [])
    if not isinstance(models, list):
        raise RuntimeError(f"{cfg}: top-level 'models' must be a list")

    enabled = [m for m in models if m.get("enabled", True)]
    if not enabled:
        raise RuntimeError(f"{cfg}: no enabled models found")
    return enabled


def default_openrouter_model(path: Path | None = None) -> str:
    """First enabled model's openrouter_model (used as script default)."""
    try:
        return str(load_enabled_models(path)[0]["openrouter_model"])
    except (OSError, json.JSONDecodeError, KeyError, RuntimeError, IndexError):
        return _FALLBACK_MODEL


def openrouter_provider_extensions(model: str) -> dict[str, Any]:
    """OpenRouter routing hints so video requests hit capable providers."""
    m = (model or "").strip().lower()
    if m.startswith("z-ai/") or "/z-ai/" in m:
        return {"provider": {"only": ["z-ai"], "allow_fallbacks": False}}
    # Gemma on DeepInfra/Novita often rejects video_url; skip those hosts.
    if "gemma" in m:
        return {"provider": {"ignore": ["DeepInfra", "Novita"], "allow_fallbacks": True}}
    return {}
