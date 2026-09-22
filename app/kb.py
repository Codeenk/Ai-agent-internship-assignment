"""Markdown parsing: YAML front matter, headings, heading-aware chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .docmeta import DocMeta

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")
_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass(frozen=True)
class Chunk:
    """One retrieval passage: a markdown section with useful context."""

    chunk_id: str          # "<doc>#<slugified-heading>"
    source: str            # filename
    heading: str           # section heading ("(document intro)" if none)
    text: str              # passage body text
    meta: DocMeta
    index: int             # position within the document

    @property
    def ref(self) -> str:
        return f"{self.source} — {self.meta.title} § {self.heading}"


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Split a markdown file into (front-matter dict, body)."""
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    raw = match.group(1)
    body = text[match.end():]
    fm: dict = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        fm[key.strip()] = value.strip().strip('"').strip("'")
    return fm, body


def split_markdown_sections(body: str) -> list[tuple[str, str]]:
    """Split markdown body into (heading, text) sections.

    The top-level ``# Title`` becomes the document intro section; each
    ``##``/``###`` heading starts a new section. Lists and paragraphs stay
    with their heading.
    """
    sections: list[tuple[str, str]] = []
    current_heading = "(document intro)"
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append((current_heading, text))

    for line in body.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            flush()
            buffer = []
            level, title = len(m.group(1)), m.group(2).strip()
            if level == 1:
                # Document title: keep content flowing under the intro.
                current_heading = "(document intro)"
            else:
                current_heading = title
        else:
            buffer.append(line)
    flush()
    return sections


def load_document(path: Path) -> tuple[DocMeta, list[Chunk]]:
    """Parse one markdown file into metadata plus heading-aware chunks."""
    text = path.read_text(encoding="utf-8")
    fm, body = parse_front_matter(text)
    meta = DocMeta(
        document_id=str(fm.get("document_id", path.stem)),
        title=str(fm.get("title", path.stem)),
        status=str(fm.get("status", "active")).lower(),
        audience=str(fm.get("audience", "customer")).lower(),
        policy_authority=str(fm.get("policy_authority", "official")).lower(),
        effective_date=str(fm.get("effective_date", "")),
        last_reviewed=str(fm.get("last_reviewed", "")),
        path=path.name,
        superseded_by=str(fm.get("superseded_by", "")),
        supersedes=str(fm.get("supersedes", "")),
        extra=fm,
    )
    chunks: list[Chunk] = []
    for i, (heading, section_text) in enumerate(split_markdown_sections(body)):
        # Prepend the doc title to short sections so retrieval still has topic context.
        prefix = f"{meta.title}. " if len(section_text) < 200 else ""
        chunks.append(
            Chunk(
                chunk_id=f"{path.name}#{heading.lower().replace(' ', '-')}",
                source=path.name,
                heading=heading,
                text=f"{prefix}{section_text}",
                meta=meta,
                index=i,
            )
        )
    return meta, chunks
