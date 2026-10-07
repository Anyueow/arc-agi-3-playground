"""Minimal Ollama client: send chat messages, get a JSON object back."""

import json
import os
import time

import requests

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
DEFAULT_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


class Ollama:
    def __init__(self, model: str = DEFAULT_MODEL, url: str = DEFAULT_URL, temperature: float = 0.3) -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.temperature = temperature
        self.calls = 0
        self.seconds = 0.0

    def check(self) -> None:
        """Fail fast with a clear message if Ollama isn't running or the model is missing."""
        try:
            tags = requests.get(f"{self.url}/api/tags", timeout=5).json()
        except requests.RequestException as e:
            raise SystemExit(f"Can't reach Ollama at {self.url} ({e}). Start it with `ollama serve`.")
        names = {m["name"] for m in tags.get("models", [])}
        if self.model not in names:
            raise SystemExit(f"Model {self.model!r} not pulled. Run `ollama pull {self.model}` (have: {sorted(names)})")

    def chat_json(self, system: str, user: str) -> tuple[dict, str]:
        """Returns (parsed JSON, raw text). Parsed is {} if the model didn't return valid JSON."""
        start = time.monotonic()
        response = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "stream": False,
                "format": "json",
                "options": {"temperature": self.temperature, "num_ctx": 8192},
            },
            timeout=180,
        )
        response.raise_for_status()
        raw = response.json()["message"]["content"]
        self.calls += 1
        self.seconds += time.monotonic() - start
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}
        return (parsed if isinstance(parsed, dict) else {}), raw
