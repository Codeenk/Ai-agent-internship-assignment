"""Command-line interface for the Aster & Row support agent.

Usage:
    python -m app.cli                 # interactive chat
    python -m app.cli --debug         # chat with per-turn trace output
    python -m app.cli --session demo1 # named session
"""

from __future__ import annotations

import argparse
import sys

from .agent import Agent
from .trace import Trace


def _print_turn(resp) -> None:
    print("\n--- Aster & Row Support ---")
    print(resp.response)
    if resp.sources:
        print("\nSources:")
        for s in resp.sources:
            print(f"  - {s}")
    if resp.handoff:
        print("\n[Human handoff recommended]"
              + (f" Reason: {resp.handoff_reason}" if resp.handoff_reason else ""))
    print("---------------------------\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aster & Row support agent CLI")
    parser.add_argument("--debug", action="store_true", help="show structured trace per turn")
    parser.add_argument("--session", default="default", help="session id (isolates history)")
    args = parser.parse_args()

    print("Aster & Row Support Agent — type 'exit' to quit, 'new' to start a new session.")
    agent = Agent()

    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not message:
            continue
        if message.lower() == "exit":
            break
        if message.lower() == "new":
            import uuid

            args.session = f"session-{uuid.uuid4().hex[:8]}"
            print(f"[started new session: {args.session}]")
            continue

        trace = Trace(enabled=True, session_id=args.session)
        try:
            resp = agent.handle(message, session_id=args.session, trace=trace)
        except Exception as exc:  # keep the loop alive no matter what
            print(f"[error] {exc}")
            trace.log("error", error=str(exc))
            trace.flush()
            continue
        if args.debug:
            for line in trace.records:
                print(line)
        _print_turn(resp)
        trace.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
