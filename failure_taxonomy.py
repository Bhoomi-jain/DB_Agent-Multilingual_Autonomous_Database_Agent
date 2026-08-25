"""
failure_taxonomy.py — Single source of truth for bug-category slugs.

Every semantic validator raises SemanticValidationError tagged with one of
the slugs below (message prefixed "[slug]" so classification and reports
can join on it). The registry binds together:

  - the slug used in code, metrics and reports ("missing_join")
  - the WRONG_ANSWERS.md bug label it fixes ("BUG-C1")
  - which fix number(s) in the §6.29 batch implement it
  - the core_agent.FailureClass constant runs get quantified under
  - the W-entries / ledger rows that documented the observed failures

This module deliberately does NOT import core_agent (which imports this
one): runtime classes are referenced by NAME so the taxonomy stays
dependency-free and testable in isolation.

Conventions for validators:

    raise SemanticValidationError(f"[{slug}] actionable message", category=slug)

Reports/metrics therefore carry machine-grep-able tokens while humans read
the same line. run_tests.py joins its failure classification on FAILURES.
"""

FAILURES = {
    # --- the five primary categories -------------------------------------
    "missing_join": {
        "label": "BUG-C1",
        "fixes": [3],
        "failure_class": "JOIN_ERROR",
        "summary": "query's join graph has >1 disconnected component",
        "refs": ["W-006", "T-15"],
    },
    "wrong_grain": {
        "label": "BUG-H1",
        "fixes": [8],
        "failure_class": "GRAIN_ERROR",
        "summary": "aggregate computed over joined-detail grain instead of planned entity",
        "refs": ["W-002", "T-08"],
    },
    "format_error": {
        "label": "BUG-H3",
        "fixes": [5],
        "failure_class": "VERIFICATION_ERROR",
        "summary": "formatter collapse / unusable or vacuously-verified large listings",
        "refs": ["W-004", "T-12"],
    },
    "intent_error": {
        "label": "BUG-M1",
        "fixes": [7],
        "failure_class": "METRIC_MISMATCH",
        "summary": "listing question answered with a scalar aggregate (shape drift)",
        "refs": ["W-001", "W-005", "T-07", "T-13"],
    },
    "semantic_error": {
        "label": "SEMANTIC_LAYER",
        "fixes": [10],
        "failure_class": "AGGREGATION_ERROR",
        "summary": "double aggregation: stored measure multiplied inside outer aggregate",
        "refs": ["W-008", "T-18"],
    },
    # --- completing coverage of all ten fixes -----------------------------
    "overvalidation": {
        "label": "BUG-A1",
        "fixes": [2],
        "failure_class": "PLAN_TABLE_REJECTION",
        "summary": "uncorroborated plan-table claim rejects correct SQL repeatedly",
        "refs": ["W-007", "T-17", "T-19"],
    },
    "fanout_seam": {
        "label": "BUG-C2",
        "fixes": [4],
        "failure_class": "AGGREGATION_ERROR",
        "summary": "parent-column arithmetic slips the detail-side fanout immunity",
        "refs": ["W-008", "T-18"],
    },
    "scope_drift": {
        "label": "BUG-H2",
        "fixes": [6],
        "failure_class": "RANKING_ERROR",
        "summary": "LIMIT/top-N injected into unranked listing requests",
        "refs": ["W-003", "T-11"],
    },
    "verifier_noise": {
        "label": "BUG-M2",
        "fixes": [1],
        "failure_class": "ROW_ATTRIBUTION_ERROR",
        "summary": "attribution checker reads list indices as cited figures",
        "refs": ["W-003", "T-11"],
    },
    "entity_binding": {
        "label": "BUG-M3",
        "fixes": [9],
        "failure_class": "ROW_ATTRIBUTION_ERROR",
        "summary": "truncated/partial entity citations fail to bind to result rows",
        "refs": ["W-003", "T-11"],
    },
}

DEFAULT_SLUG = "semantic_error"


def slug_of(message: str):
    """Extract a registered slug from a '[slug]-prefixed' validator message.
    Returns None when the message carries no recognized tag."""
    if not message:
        return None
    m = message.lstrip()
    if not m.startswith("["):
        return None
    end = m.find("]")
    if end == -1:
        return None
    candidate = m[1:end].strip()
    return candidate if candidate in FAILURES else None


def tag_message(slug: str, message: str) -> str:
    """Prefix helper so every raise site formats identically."""
    return f"[{slug}] {message}"


def describe(slug: str) -> str:
    """One-line human description for reports and debug output."""
    entry = FAILURES.get(slug)
    if not entry:
        return f"{slug}: (unregistered)"
    return (f"{slug} ({entry['label']}, fix#{'/'.join(map(str, entry['fixes']))}, "
            f"class {entry['failure_class']}): {entry['summary']} "
            f"[{', '.join(entry['refs'])}]")


def all_slugs() -> list:
    """Stable iteration order for report tables."""
    return list(FAILURES.keys())
