"""Document front-matter metadata model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocMeta:
    """Front-matter metadata for one knowledge-base document."""

    document_id: str
    title: str
    status: str          # active | superseded | draft
    audience: str        # customer | internal
    policy_authority: str  # official | none
    effective_date: str
    last_reviewed: str
    path: str            # filename, e.g. "01-returns-policy-current.md"
    superseded_by: str = ""
    supersedes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def is_active_official(self) -> bool:
        return self.status == "active" and self.policy_authority == "official"
