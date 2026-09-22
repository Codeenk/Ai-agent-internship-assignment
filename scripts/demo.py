"""Scripted demo run — used for GIF recording and as a smoke test.

Prints four scenarios end-to-end: a knowledge-base question with citations,
an order lookup, a multi-turn conversation, and a refusal/handoff case.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import Agent  # noqa: E402


def _banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> int:
    agent = Agent()
    session = "demo"

    _banner("1) Knowledge-base question with citations")
    resp = agent.handle("How long does a regular customer have to return an unused backpack?",
                        session_id=session)
    print(f"you> How long does a regular customer have to return an unused backpack?")
    print(f"agent> {resp.response}")
    print(f"[sources] {resp.sources}")

    _banner("2) Order lookup")
    resp = agent.handle("Where is ORD-1007 and when should it arrive?", session_id=session)
    print("you> Where is ORD-1007 and when should it arrive?")
    print(f"agent> {resp.response}")

    _banner("3) Multi-turn conversation (follow-up with context)")
    resp = agent.handle("Do you ship internationally?", session_id=session)
    print("you> Do you ship internationally?")
    print(f"agent> {resp.response}")
    resp = agent.handle("What about Canada, and how long does it take?", session_id=session)
    print("you> What about Canada, and how long does it take?")
    print(f"agent> {resp.response}")
    print(f"[sources] {resp.sources}")

    _banner("4) Refuses to guess / recommends human help")
    resp = agent.handle("Are all fabrics and adhesives in your bags vegan?", session_id=session)
    print("you> Are all fabrics and adhesives in your bags vegan?")
    print(f"agent> {resp.response}")

    _banner("Demo complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
