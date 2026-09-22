"""Lightweight structured tracing.

Every agent run produces a JSON-lines trace file under traces/ capturing the
user message, relevant history, retrieved passages with scores, tool calls and
sanitized results, and the final answer. No secrets (API keys) are ever
written; PII from orders is never present because the tool result is
sanitized before it ever reaches the trace.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import threading
from pathlib import Path

from .config import CONFIG

_lock = threading.Lock()


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


class Trace:
    """Collects a structured, secret-free record of one agent turn."""

    def __init__(self, enabled: bool = False, session_id: str = "default") -> None:
        self.enabled = enabled
        self.session_id = session_id
        self.events: list[dict] = []
        self.records: list[str] = []  # human-readable lines for console debug
        self._console = enabled and os.environ.get("TRACE_CONSOLE", "") == "1"

    def log(self, event_type: str, **data) -> None:
        event = {"ts": _now(), "type": event_type, **data}
        self.events.append(event)
        line = self._render(event)
        if line:
            self.records.append(line)
        if self._console:
            print(line)

    def _render(self, event: dict) -> str:
        t = event.get("type", "")
        if t == "user_message":
            return f"[trace] user: {event.get('message', '')!r}"
        if t == "history":
            return f"[trace] history turns used: {event.get('turns', 0)}"
        if t == "retrieval":
            parts = []
            for p in event.get("passages", []):
                parts.append(
                    f"  - {p.get('source')} § {p.get('heading')} (score {p.get('score'):.3f}, "
                    f"status={p.get('status')}, authority={p.get('policy_authority')})"
                )
            head = f"[trace] retrieval: {len(event.get('passages', []))} passage(s)"
            if event.get("topic_supported") is False:
                head += " -> topic NOT supported (abstain path)"
            if event.get("conflicts"):
                head += f" | conflicts: {event['conflicts']}"
            return "\n".join([head] + parts)
        if t == "tool_call":
            return f"[trace] tool call: {event.get('tool')}({event.get('arguments')}) -> {event.get('result')}"
        if t == "tool_error":
            return f"[trace] tool error: {event.get('tool')} :: {event.get('error')}"
        if t == "llm_call":
            return f"[trace] llm call: model={event.get('model')} purpose={event.get('purpose')}"
        if t == "llm_error":
            return f"[trace] llm error: {event.get('error')}"
        if t == "handoff":
            return f"[trace] handoff recommended: {event.get('reason')}"
        if t == "final":
            return f"[trace] final: {event.get('response')!r}"
        if t == "error":
            return f"[trace] error: {event.get('error')}"
        return f"[trace] {event}"

    def flush(self) -> None:
        """Append the trace to traces/<session>.jsonl (never raises)."""
        if not self.enabled or not self.events:
            return
        try:
            log_dir = Path(CONFIG.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            path = log_dir / f"{self.session_id}.jsonl"
            with _lock, path.open("a", encoding="utf-8") as fh:
                for event in self.events:
                    fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            pass
