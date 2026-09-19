"""Unit tests for validate_plan_matches_sql — the Phase-2 structural check.
No database needed: these exercise pure AST-vs-plan logic."""

from core_agent import validate_plan_matches_sql, SemanticValidationError, \
    corroborate_ranking_direction


def test_ranking_direction_polarity():
    """§6.24: llama3.2 emitted direction=ASC for a MOST question; lexical
    polarity must overpower the plan's claim."""
    cases = [
        ("Which country has the most customers?", "ASC", "DESC"),
        ("Top 5 spenders", "ASC", "DESC"),
        ("least popular products", "DESC", "ASC"),
        ("lowest revenue countries", "DESC", "ASC"),
        # Both polarities present or neither: keep plan's claim.
        ("most and least combined", "ASC", "ASC"),
        ("plain listing of names", "ASC", "ASC"),
    ]
    for q, claimed, expected in cases:
        got = corroborate_ranking_direction(q, claimed)
        assert got == expected, f"{q!r}: claimed {claimed}, expected {expected}, got {got}"
    print(f"polarity corroboration: {len(cases)}/{len(cases)} cases correct")


test_ranking_direction_polarity()


def expect_reject(label, plan, sql, schemas=None):
    try:
        validate_plan_matches_sql(plan, sql, "postgres", table_schemas=schemas)
        raise AssertionError(f"FALSE NEGATIVE on '{label}': plan={plan}, sql={sql!r}")
    except SemanticValidationError as e:
        print(f"rejected as expected: {label} :: {str(e)[:70]}...")


def expect_pass(label, plan, sql, schemas=None):
    validate_plan_matches_sql(plan, sql, "postgres", table_schemas=schemas)
    print(f"passed as expected: {label}")


# ---- metric checks -------------------------------------------------------
expect_pass("plan SUM satisfied",
            {"metric": "SUM"},
            "SELECT SUM(total) FROM invoice")
expect_reject("plan SUM got COUNT (the Q4 class)",
              {"metric": "SUM"},
              "SELECT c.name, COUNT(i.invoice_id) FROM customer c "
              "JOIN invoice i ON i.customer_id = c.customer_id GROUP BY c.name")
expect_pass("plan COUNT satisfied",
            {"metric": "COUNT"},
            "SELECT COUNT(*) FROM invoice")
expect_pass("mixed SUM+COUNT passes a SUM plan (dashboards do both)",
            {"metric": "SUM"},
            "SELECT SUM(total), COUNT(*) FROM invoice")
expect_reject("aggregates on a plain-listing question",
              {"metric": "NONE"},
              "SELECT SUM(total) FROM invoice")
expect_reject("missing aggregation entirely",
              {"metric": "SUM"},
              "SELECT total FROM invoice")

# ---- metric_column check -------------------------------------------------
expect_pass("SUM over planned column",
            {"metric": "SUM", "metric_column": "total"},
            "SELECT SUM(total) FROM invoice")
expect_reject("SUM over the WRONG column",
              {"metric": "SUM", "metric_column": "total"},
              "SELECT SUM(invoice_id) FROM invoice")

# Self-inconsistent plan (observed live with llama3.2): entity=Invoice but
# metric_column='amount' is a CHILD table's column. The plan contradicts
# itself, so the column is discarded and correct SQL must NOT be rejected.
_SCHEMAS = {
    "invoice": [{"name": "invoice_id", "type": "INT", "primary_key": True},
                {"name": "total", "type": "NUMERIC", "primary_key": False}],
    "invoiceline": [{"name": "line_id", "type": "INT", "primary_key": True},
                    {"name": "invoice_id", "type": "INT", "primary_key": False},
                    {"name": "quantity", "type": "INTEGER", "primary_key": False},
                    {"name": "amount", "type": "NUMERIC", "primary_key": False}],
}
# Gating needs the schema to know 'amount' isn't an invoice column:
validate_plan_matches_sql(
    {"metric": "SUM", "metric_column": "amount", "entity": "invoice"},
    "SELECT SUM(total) FROM invoice", "postgres", table_schemas=_SCHEMAS)
print("passed as expected: self-inconsistent plan column ignored")
# Column-level enforcement is deliberately a HINT after live testing:
# any numeric measure among the relevant tables satisfies the plan (planners
# say 'amount', models write the equivalent fact-side total — both compute
# the same figure at the query's grain). Documented blind spot:
# numeric-vs-different-numeric (Quantity instead of Amount) is owned by
# answer verification, not by plan policing.
# STRICT tier: planned column exists in a queried table -> exact column
# required. The mismatch below is exactly what repair_metric_column fixes
# deterministically (same-table Quantity->Amount swap) before retry.
expect_reject("strict: SUM(quantity) vs planned amount on same joined table",
              {"metric": "SUM", "metric_column": "amount", "entity": "invoice"},
              "SELECT SUM(il.quantity) FROM invoiceline il "
              "JOIN invoice i ON i.invoice_id = il.invoice_id", schemas=_SCHEMAS)

_SCHEMAS_TEXT = dict(_SCHEMAS)
_SCHEMAS_TEXT["invoice"] = _SCHEMAS["invoice"] + [
    {"name": "last_name", "type": "NVARCHAR(40)", "primary_key": False}]
expect_reject("aggregating a TEXT column never satisfies a SUM plan",
              {"metric": "SUM", "metric_column": "total"},
              "SELECT SUM(last_name) FROM invoice", schemas=_SCHEMAS_TEXT)

expect_pass("entity-side money column satisfies a child-column plan "
            "(post-fanout-repair shape)",
            {"metric": "SUM", "metric_column": "amount", "entity": "invoice"},
            "SELECT SUM(total) FROM invoice", schemas=_SCHEMAS)

# Ambiguous column name (exists in 2+ tables): discarded, never enforced.
_SCHEMAS_AMBIG = dict(_SCHEMAS)
_SCHEMAS_AMBIG["invoiceline"] = _SCHEMAS["invoiceline"] + [
    {"name": "invoice_id", "type": "INT", "primary_key": False}]
expect_pass("ambiguous plan column discarded (fail-open)",
            {"metric": "SUM", "metric_column": "invoice_id", "entity": "invoice"},
            "SELECT SUM(total) FROM invoice", schemas=_SCHEMAS_AMBIG)

# ---- ranking checks ------------------------------------------------------
expect_pass("top-5 correctly implemented",
            {"ranking": {"enabled": True, "direction": "DESC", "limit": 5}},
            "SELECT name FROM customer ORDER BY spent DESC LIMIT 5")
expect_reject("ranking without ORDER BY",
              {"ranking": {"enabled": True, "direction": "DESC", "limit": 5}},
              "SELECT name FROM customer LIMIT 5")
expect_reject("wrong direction (returns the WORST)",
              {"ranking": {"enabled": True, "direction": "DESC", "limit": 5}},
              "SELECT name FROM customer ORDER BY spent ASC LIMIT 5")
expect_reject("LIMIT exceeds requested top-N",
              {"ranking": {"enabled": True, "direction": "DESC", "limit": 5}},
              "SELECT name FROM customer ORDER BY spent DESC LIMIT 20")

# ---- entity check --------------------------------------------------------
expect_pass("planned entity present",
            {"entity": "customer"},
            "SELECT c.name FROM customer c JOIN invoice i ON i.customer_id = c.customer_id")
expect_reject("planned entity absent from query",
              {"entity": "customer"},
              "SELECT name FROM employee")

# ---- fail-open behavior --------------------------------------------------
expect_pass("empty plan is inert", {}, "SELECT anything_goes FROM wherever")
expect_pass("None plan is inert", None, "SELECT anything_goes FROM wherever")
expect_pass("partial plan validates only present keys",
            {"entity": "invoice"},  # no metric/ranking keys -> those unchecked
            "SELECT COUNT(x) FROM invoice ORDER BY y LIMIT 99")

print("\n--- validate_plan_matches_sql: all structural checks behave ---")
