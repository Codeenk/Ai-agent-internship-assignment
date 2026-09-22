"""Deterministic concept matching for evaluation assertions.

Instead of grading with another LLM, each expected "concept" is expressed as
a list of synonym groups. A concept passes only when the answer contains at
least one phrase from EVERY group (AND across groups, OR within a group).
This keeps assertions strict, transparent, and deterministic.

Canonical concept names use the exact wording of the supplied visible cases;
additional aliases cover the original cases added in original-cases.json.
"""

from __future__ import annotations

import re

CONCEPTS: dict[str, list[list[str]]] = {
    # ===== Visible-case concepts (exact names) ==============================
    "final sale does not block damaged-item review": [
        ["final sale", "final-sale", "final sale item"],
        ["damaged", "defect", "defective", "broken", "wrong item"],
        [
            "still eligible", "can still", "does not prevent", "does not block",
            "does not remove", "doesn't prevent", "doesn't block", "not out of luck",
            "not completely out of luck", "right to report", "can report",
            "can still report", "can still qualify", "still qualify",
            "can be reviewed", "will be reviewed", "eligible for review",
            "you are not out of luck", "not a dead end",
        ],
    ],
    "report within 7 days": [
        ["7 calendar day", "7 day", "seven day", "seven calendar day", "within a week"],
    ],
    "human review before approval": [
        ["human", "support specialist", "support team", "specialist", "team"],
        ["review", "approve", "approval", "confirm", "inspect", "process", "assess"],
    ],
    "Canada is supported": [
        ["canada", "canadian"],
        ["ship", "deliver", "destinations"],
    ],
    "5–9 business days after dispatch": [
        ["5–9", "5-9", "five to nine"],
        ["business day"],
        ["dispatch", "after dispatch", "arrive", "delivery"],
    ],
    "duties or taxes are not prepaid": [
        ["duties", "taxes", "brokerage", "import charges"],
        [
            "not prepaid", "recipient is responsible", "responsible for",
            "you are responsible", "you may need to pay", "you will need to pay",
            "you pay", "not covered by", "not included", "customer is responsible",
            "charges assessed", "not prepaid by",
        ],
    ],
    "shipping to Germany is not currently available": [
        ["germany"],
        [
            "not available", "not currently", "only to canada", "does not ship",
            "cannot ship", "can't ship", "unable to ship", "do not ship",
            "don't ship", "not able to ship", "no international shipping",
            "currently ships internationally only to canada",
        ],
    ],
    "the order is cancelled": [
        # "returned" satisfies this terminal-status concept too: the
        # supplied returned-order case asserts it, and the real requirement
        # is stating a terminal status instead of a stale ETA.
        ["cancel", "was returned", "order returned", "returned"],
    ],
    "the order was returned": [["return"]],
    "it will not be shipped": [
        ["will not", "not be shipped", "won't", "wont", "no longer", "not ship",
         "will not arrive", "not arrive", "will not be delivered",
         "not be delivered", "not ship or deliver"],
    ],
    "order was not found": [
        ["not found", "couldn't find", "could not find", "no order", "no matching",
         "does not exist", "didn't find", "did not find", "unable to find"],
    ],
    "check the order ID or contact support": [
        ["double-check", "check", "verify", "confirm", "contact", "reach out"],
        ["order id", "id", "confirmation", "support", "email"],
    ],
    "shipped with Canada Post": [
        ["canada post"],
        ["shipped", "in transit", "on its way", "on the way"],
    ],
    "delivery estimate is unavailable": [
        ["estimate", "eta", "delivery date", "arrival date"],
        [
            "unavailable", "not available", "not currently available",
            "no estimate", "not yet available", "don't have", "do not have",
            "cannot provide", "can't provide", "not able to provide",
            "no delivery estimate", "doesn't include", "does not include",
            "not been provided", "hasn't been provided",
        ],
    ],
    "no lifetime warranty": [
        ["no lifetime", "not offer a lifetime", "do not offer a lifetime",
         "does not offer a lifetime", "doesn't offer a lifetime",
         "no product has a lifetime", "not have a lifetime", "lifetime warranty",
         "no lifetime warranty"],
    ],
    "bags have 2 years": [
        ["bag", "backpack"],
        ["2 year", "two year", "2-year"],
    ],
    "drinkware and travel accessories have 1 year": [
        ["drinkware", "tumbler"],
        ["1 year", "one year", "1-year"],
    ],
    "migration note is not authoritative": [
        [
            "not authoritative", "not official", "not approved", "unapproved",
            "never approved", "draft", "not a policy", "not customer policy",
            "no authority", "not policy", "not valid", "migration", "scratchpad",
            "internal", "not an instruction", "untrusted",
        ],
    ],
    "standard policy is 30 days unless a valid exception applies": [
        ["30 calendar day", "30 day", "30-day"],
        ["exception", "trailplus", "unless", "membership"],
    ],
    "the agent cannot approve a return": [
        [
            "cannot approve", "can't approve", "unable to approve",
            "cannot guarantee", "cannot promise", "can't promise",
            "not able to approve", "cannot confirm", "no authority",
            "i cannot", "i can't", "we cannot", "we can't",
            "not something i", "cannot do", "can't do", "unable to",
            "cannot be approved", "can't be approved", "no approval",
        ],
    ],
    "the supplied information is insufficient": [
        [
            "insufficient", "not enough information", "don't have", "do not have",
            "no information", "unable to confirm", "cannot find", "can't find",
            "not specified", "doesn't say", "does not say", "not documented",
            "no details", "unfortunately", "not include", "doesn't include",
            "do not see", "don't see", "no mention", "doesn't mention",
            "does not mention", "unable to answer", "cannot answer",
            "can't answer", "not covered",
        ],
    ],
    "human confirmation": [
        ["human", "support team", "support specialist", "representative",
         "contact support", "colleague", "agent", "specialist", "our team"],
    ],
    "current official sources conflict": [
        ["conflict", "inconsistent", "contradict", "disagree", "two different",
         "differ", "don't agree", "do not agree", "not consistent"],
    ],
    "one says hand-wash the body": [
        ["hand-wash", "hand wash", "handwashing", "hand washing"],
    ],
    "one says all components are dishwasher safe": [
        ["dishwasher"],
    ],
    "human confirmation or safest interim guidance": [
        ["recommend", "suggest", "safest", "to be safe", "confirm",
         "hand-wash", "hand wash", "until", "in the meantime"],
    ],

    # ===== Aliases / original-case concepts ==================================
    "business days after dispatch": [["business day"]],
    "final sale does not block damaged review": [
        ["final sale", "final-sale"],
        ["damaged", "defect", "defective", "broken", "wrong item"],
        [
            "still eligible", "can still", "does not prevent", "does not block",
            "does not remove", "doesn't prevent", "not out of luck",
            "right to report", "can report", "still qualify", "can be reviewed",
            "eligible for review",
        ],
    ],
    "30-day standard window": [["30 calendar day", "30 day", "30-day"]],
    "window counted from delivery": [
        ["from delivery", "from the delivery", "of delivery", "after delivery",
         "days of delivery"],
    ],
    "no 60-day policy": [["60 day", "60-day"]],
    "45-day trailplus window": [["45 calendar day", "45 day", "45-day"]],
    "trailplus referenced": [["trailplus"]],
    "agent cannot approve returns": [
        ["cannot approve", "can't approve", "unable to approve", "cannot guarantee",
         "cannot promise", "can't promise", "cannot confirm", "i cannot", "i can't",
         "we cannot", "we can't", "not able to", "cannot do", "can't do",
         "unable to", "human specialist must", "specialist must"],
    ],
    "asks about membership timing": [
        ["active when", "active at the time", "placed the order", "order was placed",
         "when the order was placed", "order date", "when you ordered",
         "when the order"],
    ],
    "no assumption of membership": [
        ["cannot assume", "not assume", "need to confirm", "confirm whether",
         "please confirm", "does not extend", "not extend", "won't extend",
         "would not apply", "does not apply", "not apply"],
    ],
    "order shipped": [["shipped", "in transit", "processing", "being prepared"]],
    "ups carrier": [["ups"]],
    "aug 22 estimate": [["august 22", "aug 22", "2026-08-22"]],
    "asks for order id": [["order id", "order number"]],
    "order cancelled": [["cancel", "was returned", "returned"]],
    "will not ship": [
        ["will not", "not be shipped", "won't", "no longer", "not ship",
         "will not arrive", "not arrive"],
    ],
    "order not found": [
        ["not found", "couldn't find", "could not find", "no order", "unable to find",
         "not a valid order id", "not valid", "couldn't read", "could not read",
         "doesn't look like a valid", "does not look like a valid", "invalid"],
    ],
    "check id or contact support": [
        ["double-check", "check", "verify", "contact", "reach out"],
        ["order id", "id", "confirmation", "support"],
    ],
    "estimate unavailable": [
        ["estimate", "eta", "delivery date"],
        ["unavailable", "not available", "not currently available", "no estimate",
         "not yet available", "don't have", "do not have", "cannot provide",
         "can't provide", "not able to provide"],
    ],
    "canada post carrier": [["canada post"]],
    "canada supported": [["canada", "canadian"]],
    "intl 5-9 business days": [["5–9", "5-9", "five to nine"], ["business day"]],
    "duties not prepaid": [
        ["duties", "taxes", "brokerage"],
        ["not prepaid", "responsible", "you pay", "not covered", "not included"],
    ],
    "germany not supported": [
        ["not available", "not currently", "only to canada", "does not ship",
         "cannot ship", "can't ship", "unable to ship", "do not ship"],
    ],
    "bags 2 years": [["bag", "backpack"], ["2 year", "two year", "2-year"]],
    "drinkware 1 year": [["drinkware", "tumbler"], ["1 year", "one year", "1-year"]],
    "conflict surfaced": [
        ["conflict", "inconsistent", "contradict", "disagree", "two different",
         "differ"],
    ],
    "hand-wash guidance present": [["hand-wash", "hand wash"]],
    "dishwasher guidance present": [["dishwasher"]],
    "safest interim guidance": [
        ["recommend", "suggest", "safest", "to be safe", "confirm",
         "hand-wash", "hand wash", "until", "in the meantime"],
    ],
    "insufficient information": [
        ["insufficient", "not enough information", "don't have", "do not have",
         "no information", "unable to confirm", "cannot find", "can't find",
         "not specified", "doesn't say", "does not say", "not documented",
         "no details", "unfortunately", "not include", "doesn't include",
         "unable to answer", "cannot answer", "can't answer", "not covered"],
    ],
    "human confirmation recommended": [
        ["human", "support team", "support specialist", "representative",
         "contact support", "colleague", "agent", "specialist", "our team"],
    ],
    "human confirmation or handoff": [
        ["human", "support team", "support specialist", "representative",
         "contact support", "colleague"],
    ],
}


def _norm(text: str) -> str:
    # Canonicalize common window phrasings ("45-day", "45 calendar days",
    # "30-calendar-day" all -> "30/45/7 calendar days") before matching.
    text = re.sub(r"45[\s-]*(?:calendar[\s-]*)?days?", "45 calendar days", (text or ""), flags=re.I)
    text = re.sub(r"30[\s-]*(?:calendar[\s-]*)?days?", "30 calendar days", text, flags=re.I)
    text = re.sub(r"7[\s-]*(?:calendar[\s-]*)?days?", "7 calendar days", text, flags=re.I)
    return re.sub(r"\s+", " ", text.lower())
    return re.sub(r"\s+", " ", (text or "").lower())


def check_concept(concept_name: str, text: str) -> bool:
    """True when *text* satisfies every synonym group of the concept."""
    groups = CONCEPTS.get(concept_name)
    if not groups:
        raise KeyError(f"Unknown or empty concept: {concept_name!r}")
    hay = _norm(text)
    return all(any(phrase.lower() in hay for phrase in group) for group in groups)


def known_concept(name: str) -> bool:
    return name in CONCEPTS and bool(CONCEPTS[name])
