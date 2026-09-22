"""Central configuration.

All knobs live here so the rest of the code never reads os.environ
directly. Secrets are read from the environment (optionally via a .env
file); no credentials are ever hardcoded or logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_ROOT / "knowledge-base"
ORDERS_PATH = PROJECT_ROOT / "data" / "orders.json"
EVAL_VISIBLE_PATH = PROJECT_ROOT / "evaluation" / "visible-cases.json"
EVAL_ORIGINAL_PATH = PROJECT_ROOT / "evaluation" / "original-cases.json"
TRACE_DIR = PROJECT_ROOT / "traces"


def _load_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env loader (KEY=VALUE lines, # comments). Never overrides
    variables that are already set in the real environment."""
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


_load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class LLMConfig:
    """Settings for the chat model provider (OpenAI-compatible)."""

    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
    )
    model: str = field(default_factory=lambda: os.environ.get("MODEL_NAME", "gpt-4o-mini"))
    temperature: float = 0.1
    max_tokens: int = field(
        default_factory=lambda: int(os.environ.get("MAX_TOKENS", "900"))
    )
    timeout_seconds: float = field(
        default_factory=lambda: float(os.environ.get("LLM_TIMEOUT", "90"))
    )
    max_retries: int = field(
        default_factory=lambda: int(os.environ.get("LLM_MAX_RETRIES", "3"))
    )


@dataclass
class EmbeddingConfig:
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    model: str = field(default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"))
    timeout_seconds: float = 30.0


@dataclass
class RetrievalConfig:
    candidate_k: int = 24          # BM25 candidates before re-ranking
    final_k: int = 7               # passages sent to the model
    min_score_ratio: float = 0.18  # drop passages below this fraction of the best score
    dominance_ratio: float = 0.45  # reserved: topic-absence threshold
    conflict_overlap: int = 4      # top-N passages checked for active-source conflicts


@dataclass
class SessionConfig:
    max_history_turns: int = 12    # kept turns per session
    session_ttl_seconds: int = 3600


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    log_dir: Path = TRACE_DIR


CONFIG = AppConfig()
