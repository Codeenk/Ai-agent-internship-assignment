"""Order lookup tool.

Implements the data dictionary exactly: normalization of harmless input
differences, a strict customer-safe field allowlist, and status-driven rules
for stale delivery fields. The tool result is the ONLY order data that ever
reaches the model — the full orders file is never placed in the prompt.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .config import ORDERS_PATH

_ID_RE = re.compile(r"^[A-Z0-9-]+$")

# Fields the tool may return to the model (per data/orders-data-dictionary.md).
SAFE_ITEM_FIELDS = ("name", "quantity", "final_sale")
SAFE_TOP_FIELDS = (
    "order_id",
    "membership_tier",
    "placed_at",
    "status",
    "status_updated_at",
    "shipped_at",
    "delivered_at",
    "carrier",
    "tracking_number",
    "estimated_delivery",
    "customer_safe_message",
)


class OrderLookupError(Exception):
    """Raised when a lookup cannot be performed (malformed/unknown id)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class OrderLookupResult:
    """Sanitized, model-facing result of an order lookup."""

    found: bool
    order_id: str = ""
    fields: dict = field(default_factory=dict)
    stale_delivery_fields: bool = False
    error: str = ""  # "not_found" | "malformed" | ""

    def to_prompt(self) -> str:
        """Render the tool result as untrusted tool-output text for the model."""
        if not self.found:
            return (
                f"ORDER LOOKUP RESULT: no order found for id '{self.order_id}'. "
                f"Reason: {self.error}. Do not invent any order details."
            )
        lines = [f"ORDER LOOKUP RESULT (sanitized; internal fields removed) for {self.order_id}:"]
        for key in SAFE_TOP_FIELDS:
            if key in self.fields:
                lines.append(f"- {key}: {self.fields[key]}")
        if "items" in self.fields:
            for item in self.fields["items"]:
                safe = ", ".join(
                    f"{k}: {item[k]}" for k in SAFE_ITEM_FIELDS if k in item
                )
                lines.append(f"- item: {safe}")
        if self.stale_delivery_fields:
            lines.append(
                "NOTE: delivery fields above are STALE and must NOT be reported "
                "as upcoming delivery (order is cancelled/returned)."
            )
        return "\n".join(lines)


def normalize_order_id(raw: str) -> str:
    """Normalize harmless differences: trim, drop punctuation, uppercase."""
    if not isinstance(raw, str):
        raise OrderLookupError("malformed", "Order IDs must be text.")
    cleaned = raw.strip().strip("\"'.,!?;:")
    cleaned = cleaned.upper().replace(" ", "")
    # Remove common separators/punctuation people add: ord.1007 / ord 1007 / ord- 1007
    cleaned = cleaned.replace("_", "-")
    cleaned = re.sub(r"^ORD[-.\s]*", "ORD-", cleaned)
    cleaned = cleaned.replace("ORD--", "ORD-")
    if not _ID_RE.match(cleaned):
        raise OrderLookupError(
            "malformed", f"'{raw.strip()}' is not a valid order ID format."
        )
    return cleaned


def load_orders(path=None) -> dict:
    path = path or ORDERS_PATH
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class OrderStore:
    """In-memory view over data/orders.json with sanitized lookups."""

    def __init__(self, path=None) -> None:
        data = load_orders(path)
        self.snapshot_at: str = data.get("snapshot_at", "")
        self._orders: dict[str, dict] = {
            o["order_id"]: o for o in data.get("orders", [])
        }

    def lookup(self, raw_id: str) -> OrderLookupResult:
        order_id = normalize_order_id(raw_id)
        order = self._orders.get(order_id)
        if order is None:
            return OrderLookupResult(found=False, order_id=order_id, error="not_found")

        status = str(order.get("status", "")).lower()
        stale = status in ("cancelled", "returned")

        fields: dict = {}
        for key in SAFE_TOP_FIELDS:
            if key == "order_id":
                fields[key] = order_id
                continue
            # Stale delivery fields are dropped for cancelled/returned orders.
            if stale and key in ("carrier", "tracking_number", "estimated_delivery"):
                continue
            if order.get(key) is not None:
                fields[key] = order.get(key)
        fields["items"] = [
            {k: item[k] for k in SAFE_ITEM_FIELDS if k in item}
            for item in order.get("items", [])
        ]
        return OrderLookupResult(
            found=True, order_id=order_id, fields=fields, stale_delivery_fields=stale
        )


def lookup_order(raw_id: str, store: OrderStore | None = None) -> OrderLookupResult:
    store = store or OrderStore()
    return store.lookup(raw_id)
