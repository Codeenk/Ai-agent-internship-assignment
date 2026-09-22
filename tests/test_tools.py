"""Unit tests for the order lookup tool (offline, deterministic)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.tools import (  # noqa: E402
    OrderLookupError,
    OrderStore,
    normalize_order_id,
)


def _write_orders(tmp: Path, orders: list[dict]) -> Path:
    path = tmp / "orders.json"
    path.write_text(json.dumps({"snapshot_at": "2026-08-15T12:00:00Z", "orders": orders}))
    return path


ORDER_1007 = {
    "order_id": "ORD-1007",
    "customer": {
        "name": "Ava Morgan",
        "email": "ava.morgan@example.test",
        "shipping_address": "220 King Street West, Toronto, ON M5V 3M2",
    },
    "membership_tier": "standard",
    "items": [{"sku": "PACK-ATLAS-BLK", "name": "Atlas Weekender",
               "quantity": 1, "final_sale": False}],
    "placed_at": "2026-08-11T15:05:00Z",
    "status": "shipped",
    "status_updated_at": "2026-08-14T20:40:00Z",
    "shipped_at": "2026-08-14T20:40:00Z",
    "delivered_at": None,
    "carrier": "UPS",
    "tracking_number": "1ZAR100700000007",
    "estimated_delivery": "2026-08-22",
    "customer_safe_message": "In transit with UPS.",
    "internal": {"risk_score": 82, "warehouse_note": "Manual fraud review cleared.",
                 "support_tags": ["international"]},
}


class NormalizeOrderIdTests(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(normalize_order_id("ORD-1007"), "ORD-1007")

    def test_lowercase(self):
        self.assertEqual(normalize_order_id("ord-1007"), "ORD-1007")

    def test_whitespace(self):
        self.assertEqual(normalize_order_id("  ORD-1007  "), "ORD-1007")

    def test_inner_space(self):
        self.assertEqual(normalize_order_id("ORD 1007"), "ORD-1007")

    def test_punctuation(self):
        self.assertEqual(normalize_order_id("\"ord-1007,\""), "ORD-1007")
        self.assertEqual(normalize_order_id("ord.1007"), "ORD-1007")

    def test_malformed_raises(self):
        with self.assertRaises(OrderLookupError):
            normalize_order_id("ORD_@@!!")

    def test_non_string_raises(self):
        with self.assertRaises(OrderLookupError):
            normalize_order_id(None)  # type: ignore[arg-type]


class OrderStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = _write_orders(Path(self.tmp.name), [ORDER_1007])
        self.store = OrderStore(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_found_returns_safe_fields(self):
        result = self.store.lookup("ord-1007")
        self.assertTrue(result.found)
        self.assertEqual(result.fields["status"], "shipped")
        self.assertEqual(result.fields["carrier"], "UPS")
        self.assertEqual(result.fields["estimated_delivery"], "2026-08-22")

    def test_no_internal_fields(self):
        prompt = self.store.lookup("ORD-1007").to_prompt()
        for forbidden in ("ava.morgan@example.test", "Ava Morgan", "220 King",
                          "82", "fraud", "warehouse", "support_tags", "risk"):
            self.assertNotIn(forbidden, prompt, f"leaked: {forbidden}")

    def test_unknown_order(self):
        result = self.store.lookup("ORD-9999")
        self.assertFalse(result.found)
        self.assertEqual(result.error, "not_found")

    def test_cancelled_stale_fields_removed(self):
        order = dict(ORDER_1007)
        order["order_id"] = "ORD-1004"
        order["status"] = "cancelled"
        order["estimated_delivery"] = "2026-08-16"
        path = _write_orders(Path(self.tmp.name), [order])
        store = OrderStore(path)
        result = store.lookup("ORD-1004")
        self.assertTrue(result.found)
        self.assertNotIn("estimated_delivery", result.fields)
        self.assertNotIn("carrier", result.fields)
        self.assertNotIn("tracking_number", result.fields)
        self.assertTrue(result.stale_delivery_fields)
        prompt = result.to_prompt()
        self.assertNotIn("2026-08-16", prompt)

    def test_shipped_without_eta(self):
        order = json.loads(json.dumps(ORDER_1007))
        order["order_id"] = "ORD-1011"
        order["carrier"] = "Canada Post"
        order["estimated_delivery"] = None
        path = _write_orders(Path(self.tmp.name), [order])
        store = OrderStore(path)
        result = store.lookup("ORD-1011")
        self.assertIsNone(result.fields.get("estimated_delivery"))


if __name__ == "__main__":
    unittest.main()
