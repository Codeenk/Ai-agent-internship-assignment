"""Unit tests for the BM25 index and tokenizer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.bm25 import build_index, tokenize  # noqa: E402


class FakeChunk:
    def __init__(self, heading: str, text: str, source: str = "doc.md"):
        self.heading = heading
        self.text = text
        self.source = source


class TokenizeTests(unittest.TestCase):
    def test_lowercase_and_split(self):
        self.assertEqual(tokenize("Hello, World!"), ["hello", "world"])

    def test_suffix_folding(self):
        self.assertEqual(tokenize("returns"), ["return"])
        self.assertEqual(tokenize("returns returning"), ["return", "return"])
        self.assertEqual(tokenize("canadian"), ["canada"])

    def test_generic_plural_fold(self):
        self.assertEqual(tokenize("days ships"), ["day", "ship"])

    def test_numbers_kept(self):
        self.assertEqual(tokenize("ORD-1007 30 days"), ["ord", "1007", "30", "day"])


    def test_synonym_canonicalization(self):
        self.assertEqual(tokenize("where is my delivery?"), ["where", "is", "my", "deliver"])


class BM25Tests(unittest.TestCase):
    def setUp(self):
        self.index = build_index([
            FakeChunk("Standard return window", "Customers may request a return within 30 calendar days of delivery."),
            FakeChunk("Supported destinations", "We ship internationally only to Canada. Other countries are not available."),
            FakeChunk("Warranty periods", "Bags and backpacks have a 2 year warranty. Drinkware has 1 year."),
        ])

    def test_ranking_relevant_first(self):
        hits = self.index.search("return window days", k=3)
        self.assertEqual(hits[0][0].heading, "Standard return window")

    def test_no_match_empty(self):
        hits = self.index.search("zebra quantum banana", k=3)
        self.assertEqual(hits, [])

    def test_deterministic(self):
        a = self.index.search("canada shipping", k=3)
        b = self.index.search("canada shipping", k=3)
        self.assertEqual([h[0].heading for h in a], [h[0].heading for h in b])


if __name__ == "__main__":
    unittest.main()
