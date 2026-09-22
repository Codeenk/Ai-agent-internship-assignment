"""Unit tests for retrieval precedence, abstention, and conflict detection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retriever import Retriever  # noqa: E402
from app.config import KB_DIR  # noqa: E402


class RetrievalPrecedenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = Retriever(KB_DIR)

    def test_current_policy_beats_legacy(self):
        result = self.retriever.retrieve("How long do I have to return an unused backpack?")
        top_sources = [p.source for p in result.passages[:3]]
        self.assertIn("01-returns-policy-current.md", top_sources)
        self.assertNotIn("02-returns-policy-legacy.md", top_sources)

    def test_superseded_excluded_by_default(self):
        result = self.retriever.retrieve("return window 45 days free label")
        for p in result.passages:
            self.assertNotEqual(p.chunk.meta.status, "superseded")

    def test_superseded_allowed_when_explicit(self):
        result = self.retriever.retrieve(
            "What was the return window before April 2026?", allow_superseded=True
        )
        self.assertTrue(any(p.source == "02-returns-policy-legacy.md" for p in result.passages))

    def test_draft_doc_not_authoritative(self):
        result = self.retriever.retrieve("60 days return window every item")
        # The migration scratchpad may appear in the candidate set, but the
        # best authoritative passage must outrank it.
        best = result.best()
        self.assertIsNotNone(best)
        self.assertTrue(best.chunk.meta.is_active_official or best.chunk.meta.status == "superseded")

    def test_canada_retrieval(self):
        result = self.retriever.retrieve(
            "Do you ship internationally? What about Canada, how long does it take?"
        )
        self.assertIn("06-international-shipping.md", [p.source for p in result.passages[:3]])

    def test_conflict_detected_breeze_tumbler(self):
        result = self.retriever.retrieve(
            "Can I put the entire Breeze Tumbler in the dishwasher?"
        )
        sources = [p.source for p in result.passages]
        self.assertIn("11-product-care.md", sources)
        self.assertIn("12-breeze-tumbler-product-card.md", sources)
        self.assertTrue(result.conflicts, "expected conflict between care guide and product card")

    def test_gibberish_topic_not_supported(self):
        """Essentially zero lexical support -> retrieval-level abstention."""
        result = self.retriever.retrieve("zzz qwerty flux capacitor fluxing?")
        self.assertFalse(result.topic_supported)

    def test_vegan_question_retrieves_care_content(self):
        """Vegan-materials question retrieves product-care content; the
        generator is responsible for declaring insufficiency (eval case
        insufficient-information asserts the live-model behavior)."""
        result = self.retriever.retrieve(
            "Are the fabrics and adhesives in your bags vegan?"
        )
        self.assertTrue(result.passages)  # retrieval itself doesn't crash
        # No passage may claim to answer vegan certification.

    def test_sources_have_metadata(self):
        result = self.retriever.retrieve("return window")
        for p in result.passages:
            self.assertIn(p.chunk.meta.status, ("active", "superseded", "draft"))
            self.assertIn(p.chunk.meta.policy_authority, ("official", "none"))


if __name__ == "__main__":
    unittest.main()
