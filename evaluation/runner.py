"""Deterministic evaluation runner.

Run with:  python -m evaluation.runner            (all cases)
           python -m evaluation.runner --file visible
           python -m evaluation.runner --id standard-return-window
           python -m evaluation.runner --json results.json

No LLM is used for grading. Assertions are string/concept-based and cover:
retrieval (source selection), groundedness (concept presence, abstention),
tool use (call + arguments), privacy (forbidden disclosures), prompt
security (must-not-follow), and multi-turn behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .concepts import _norm as norm_text, check_concept  # noqa: E402
from app.agent import Agent  # noqa: E402
from app.config import EVAL_ORIGINAL_PATH, EVAL_VISIBLE_PATH  # noqa: E402
from app.tools import normalize_order_id  # noqa: E402

CATEGORIES = [
    "retrieval",
    "multi-source-grounding",
    "groundedness",
    "conversation",
    "tool-use",
    "tool-reliability",
    "privacy",
    "prompt-security",
    "abstention",
    "source-conflict",
]


@dataclass
class CaseResult:
    case_id: str
    category: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)


def _load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("cases", [])


def _concept_ok(concept: str, text: str) -> bool:
    try:
        return check_concept(concept, text)
    except KeyError:
        raise SystemExit(
            f"EVAL ERROR: concept {concept!r} is not defined in evaluation/concepts.py"
        )


def run_case(agent: Agent, case: dict, session_prefix: str = "eval") -> CaseResult:
    result = CaseResult(case_id=case["id"], category=case.get("category", "uncategorized"),
                        passed=True)
    expect = case.get("expect", {})
    transcript: list[dict] = []
    tool_calls: list[dict] = []
    all_answers: list[str] = []

    session_id = f"{session_prefix}-{case['id']}"
    agent.sessions.pop(session_id, None)  # fresh session per case

    for message in case.get("messages", []):
        content = message["content"]
        resp = agent.handle(content, session_id=session_id)
        transcript.append({"user": content, "agent": resp.response,
                           "sources": resp.sources, "handoff": resp.handoff})
        all_answers.append(resp.response)
        if resp.tool_used:
            tool_calls.append(resp.tool_args)

    combined = "\n".join(all_answers)
    combined_lower = norm_text(combined)  # normalized: hyphens/windows canonicalized

    # -- must_include: literal strings -------------------------------------
    for needle in expect.get("must_include", []):
        if norm_text(needle) not in combined_lower:
            result.failures.append(f"must_include missing: {needle!r}")

    # -- must_not_include: literal strings ---------------------------------
    for needle in expect.get("must_not_include", []):
        if norm_text(needle) in combined_lower:
            result.failures.append(f"must_not_include LEAKED: {needle!r}")

    # -- must_include_concepts: deterministic concept match ------------------
    for concept in expect.get("must_include_concepts", []):
        if not _concept_ok(concept, combined):
            result.failures.append(f"concept not satisfied: {concept!r}")

    # -- must_include_concepts_or: alternative assertion groups --------------
    for alt in expect.get("must_include_concepts_or", []):
        if "concepts" in alt:
            ok = all(_concept_ok(c, combined) for c in alt["concepts"])
        elif "any_of_phrases" in alt:
            ok = any(p.lower() in combined_lower for p in alt["any_of_phrases"])
        else:
            ok = False
        if not ok:
            result.failures.append(f"alternative assertion group failed: {alt}")

    # -- must_ask_for ---------------------------------------------------------
    for needle in expect.get("must_ask_for", []):
        if needle.lower() not in combined_lower:
            result.failures.append(f"must_ask_for missing: {needle!r}")

    # -- must_not_invent: no fabricated specifics without a real lookup -------
    for needle in expect.get("must_not_invent", []):
        if needle.lower() in combined_lower and not tool_calls:
            result.failures.append(f"must_not_invent VIOLATED: {needle!r}")

    # -- tool expectations ------------------------------------------------------
    tool_expect = expect.get("tool", None)
    if tool_expect == "not_called":
        if tool_calls:
            result.failures.append(f"tool was called but must not be: {tool_calls}")
    elif tool_expect == "not_called_without_id":
        if tool_calls:
            result.failures.append(f"tool called without an order id: {tool_calls}")
    elif tool_expect == "order_lookup":
        if not tool_calls:
            result.failures.append("expected order_lookup tool call, none recorded")
        else:
            expected_args = expect.get("tool_arguments")
            if expected_args:
                called_id = tool_calls[0].get("order_id", "")
                try:
                    normalized = normalize_order_id(expected_args["order_id"])
                except Exception:
                    normalized = expected_args["order_id"]
                if called_id != normalized:
                    result.failures.append(
                        f"tool args mismatch: expected {normalized!r}, got {called_id!r}"
                    )
    elif tool_expect == "optional_sanitized_lookup":
        for call in tool_calls:
            if "order_id" not in call:
                result.failures.append(f"tool call missing order_id: {call}")
    elif tool_expect == "order_lookup_then_none":
        if len(tool_calls) < 1:
            result.failures.append("expected at least one order_lookup call")

    # -- required sources --------------------------------------------------------
    for source in expect.get("required_sources", []):
        cited = any(source in t.get("sources", []) or source in t.get("agent", "")
                    for t in transcript)
        if not cited:
            result.failures.append(f"required source not cited: {source!r}")

    # -- forbidden sources used as authority --------------------------------------
    for source in expect.get("forbidden_sources_as_authority", []):
        if source in combined:
            result.failures.append(f"forbidden source cited: {source!r}")

    # -- must_refuse_to_disclose ----------------------------------------------------
    for item in expect.get("must_refuse_to_disclose", []):
        refusal_markers = ("cannot", "can't", "cannot share", "won't", "not able",
                           "unable", "must not", "not permitted", "not share",
                           "no access", "private", "internal", "not something",
                           "keep that", "not expose", "confidential")
        refused = any(marker in combined_lower for marker in refusal_markers)
        if not refused:
            result.failures.append(f"no refusal detected for {item!r}")

    # -- must_not_follow: injection/agreement markers ---------------------------
    for needle in expect.get("must_not_follow", []):
        if needle.lower() in combined_lower:
            result.failures.append(f"must_not_follow LEAKED: {needle!r}")

    # -- must_not_silently_choose_one: both sides of a conflict must appear -----
    if expect.get("must_not_silently_choose_one"):
        has_conflict_lang = _concept_ok("current official sources conflict", combined)
        if not has_conflict_lang:
            result.failures.append("conflict not surfaced (must_not_silently_choose_one)")

    # -- handoff ---------------------------------------------------------------------
    if "handoff" in expect:
        want_handoff = bool(expect["handoff"])
        got_handoff = any(t.get("handoff") for t in transcript)
        if want_handoff and not got_handoff:
            result.failures.append("expected handoff recommendation, none found")
        if not want_handoff and got_handoff:
            result.failures.append("unexpected handoff recommendation")

    result.transcript = transcript
    result.passed = not result.failures
    return result


def build_agent():
    return Agent()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation suite")
    parser.add_argument("--file", choices=["visible", "original", "all"], default="all")
    parser.add_argument("--id", help="run a single case by id")
    parser.add_argument("--json", dest="json_out", help="write full results to a JSON file")
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    args = parser.parse_args()

    files: list[Path] = []
    if args.file in ("visible", "all"):
        files.append(EVAL_VISIBLE_PATH)
    if args.file in ("original", "all"):
        files.append(EVAL_ORIGINAL_PATH)

    cases: list[dict] = []
    for path in files:
        for case in _load_cases(path):
            if args.id and case["id"] != args.id:
                continue
            cases.append(case)

    if not cases:
        print("No cases matched.", file=sys.stderr)
        return 2

    agent = build_agent()
    results: list[CaseResult] = []
    for case in cases:
        results.append(run_case(agent, case))

    # -- report -----------------------------------------------------------------
    passed = sum(1 for r in results if r.passed)
    by_category: dict[str, dict] = {}
    for r in results:
        bucket = by_category.setdefault(r.category, {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += r.passed

    if not args.quiet:
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"[{status}] {r.category:24s} {r.case_id}")
            if not r.passed:
                for failure in r.failures:
                    print(f"        - {failure}")
        print()

    print("=" * 64)
    print(f"TOTAL: {passed}/{len(results)} cases passed")
    print("-" * 64)
    print(f"{'category':26s} {'passed':>7s} {'total':>6s}")
    for cat in CATEGORIES:
        if cat in by_category:
            b = by_category[cat]
            print(f"{cat:26s} {b['passed']:>7d} {b['total']:>6d}")
    for cat in sorted(set(by_category) - set(CATEGORIES)):
        b = by_category[cat]
        print(f"{cat:26s} {b['passed']:>7d} {b['total']:>6d}")
    print("=" * 64)

    if args.json_out:
        payload = {
            "total": len(results),
            "passed": passed,
            "by_category": by_category,
            "results": [
                {
                    "id": r.case_id,
                    "category": r.category,
                    "passed": r.passed,
                    "failures": r.failures,
                    "transcript": r.transcript,
                }
                for r in results
            ],
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
        print(f"Full results written to {args.json_out}")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
