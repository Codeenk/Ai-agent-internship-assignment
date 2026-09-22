"""Minimal OpenAI-compatible chat client (stdlib only, no dependencies).

Uses urllib against /chat/completions. Works with OpenAI, Google AI Studio's
OpenAI-compatible endpoint (Gemini / Gemma), OpenRouter, Groq, Together, or
any compatible local server (Ollama, vLLM, LM Studio).

Reliability details:
- Thinking models (e.g. Gemma 4) wrap reasoning in <thought>...</thought>
  blocks; these are stripped so traces and evals only see the final answer.
- Transient provider errors (HTTP 429/5xx) are retried with backoff.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

from .config import CONFIG

_THOUGHT_RE = re.compile(
    r"<thought>.*?</thought>\s*|<think(?:ing)?>.*?</think(?:ing)?>\s*",
    re.DOTALL | re.IGNORECASE,
)
# Unterminated thought block (hit the token cap mid-reasoning).
_OPEN_THOUGHT_RE = re.compile(r"<(?:thought|think(?:ing)?)>.*\Z", re.DOTALL | re.IGNORECASE)


def strip_thoughts(text: str) -> str:
    out = _THOUGHT_RE.sub("", text or "")
    out = _OPEN_THOUGHT_RE.sub("", out)
    return out.strip()


class LLMError(RuntimeError):
    pass


class LLMUnavailable(LLMError):
    """Provider failed after retries — caller should fall back gracefully."""


def _post_once(url: str, api_key: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        if exc.code in (429, 500, 502, 503, 504):
            raise LLMUnavailable(f"LLM HTTP {exc.code} (transient): {detail}") from exc
        raise LLMError(f"LLM HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMUnavailable(f"LLM connection error: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LLMUnavailable("LLM request timed out") from exc


class LLMClient:
    """Thin chat-completions client with retries and thought-stripping."""

    def __init__(self, config=None) -> None:
        self.config = config or CONFIG.llm

    @property
    def available(self) -> bool:
        return bool(self.config.api_key)

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        if not self.config.api_key:
            raise LLMError("No API key configured (set OPENAI_API_KEY).")
        payload: dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        url = self.config.base_url.rstrip("/") + "/chat/completions"

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                data = _post_once(
                    url, self.config.api_key, payload, self.config.timeout_seconds
                )
                choice = data["choices"][0]
                content = strip_thoughts(choice["message"].get("content") or "")
                return {
                    "content": content,
                    "finish_reason": choice.get("finish_reason", ""),
                }
            except LLMUnavailable as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    time.sleep(1.5 * (attempt + 1))  # linear backoff
        raise LLMUnavailable(str(last_error))
