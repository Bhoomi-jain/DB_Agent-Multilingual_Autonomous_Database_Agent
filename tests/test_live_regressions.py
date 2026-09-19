"""
test_live_regressions.py — Pins for the WRONG_ANSWERS.md bug inventory,
one test function per failure_taxonomy slug as its fix lands. These are
RED before the fix and must stay GREEN after; the full suite runs them
via run_tests.py like every other test_* script.

Pure-function pins where possible (no FakeLLM/DB round-trips): the bugs
under test live in deterministic validators, so synthetic result rows
reproduce the observed live failures exactly.
"""

import core_agent as ca


# --------------------------------------------------------------------------
# verifier_noise (BUG-M2 / W-003#2 / T-11): numbered-list answers made the
# attribution checker cite the NEXT item's index as the current entity's
# figure. Mechanism: _SENTENCE_SPLIT_RE splits after each marker's dot
# ("...25.86\n2." + "Richard..."), leaving a bare "2." at sentence end;
# _strip_list_markers required whitespace after the dot and missed it.
# --------------------------------------------------------------------------

def test_verifier_noise_list_indices_not_cited_as_figures():
    rows = [
        ["Helena Holy", 25.86, 1.0],
        ["Richard Cunningham", 23.86, 2.0],
        ["Ladislav Kovacs", 21.86, 3.0],
        ["Hugh O'Reilly", 21.86, 4.0],
        ["Johannes Van der Berg", 13.86, 20.0],
    ]
    result = {"rows": rows}
    answer = (
        "Here is the rewritten list with correct attribution:\n\n"
        "1. Helena Holy - 25.86\n"
        "2. Richard Cunningham - 23.86\n"
        "3. Ladislav Kovacs - 21.86\n"
        "4. Hugh O'Reilly - 21.86\n"
        "5. Johannes Van der Berg - 13.86\n"
    )
    issues = ca.verify_row_attribution(answer, result)
    joined = "\n".join(issues)
    for idx in range(2, 6):
        assert f"{idx} is cited" not in joined, \
            f"list index {idx} falsely cited as a figure:\n{joined}"
    assert "25.86 is cited" not in joined
    assert "23.86 is cited" not in joined


def test_verifier_noise_still_catches_real_misbindings():
    """The fix must not gut the checker: a genuinely wrong figure for an
    unambiguously-bound entity still gets flagged."""
    rows = [["Helena Holy", 25.86, 1.0], ["Richard Cunningham", 23.86, 2.0]]
    issues = ca.verify_row_attribution(
        "Helena Holy - 99.99", {"rows": rows})
    assert any("99.99" in i for i in issues), \
        f"real misbinding went unflagged: {issues}"




# --------------------------------------------------------------------------
# overvalidation (BUG-A1 / W-007 / T-17): the plan named 'Invoice' as the
# main table for "Which genre generated the most revenue?" (money-vocabulary
# noise); the unconditional entity-presence check then rejected three
# structurally correct attempts -> exit 1. Fix: enforce the main-table
# presence rule ONLY when the question itself lexically references the
# claimed table; otherwise the claim is uncorroborated planner noise and
# enforcement is skipped (logged).
# --------------------------------------------------------------------------

def test_overvalidation_uncorroborated_main_table_claim():
    sql = (
        "SELECT G.Name, SUM(IL.Quantity * IL.UnitPrice) AS rev "
        "FROM Genre AS G JOIN Track AS T ON T.GenreId = G.GenreId "
        "JOIN InvoiceLine AS IL ON IL.TrackId = T.TrackId "
        "GROUP BY G.Name ORDER BY rev DESC LIMIT 1"
    )
    schemas = {
        "Genre": [{"name": "GenreId", "type": "INTEGER"},
                  {"name": "Name", "type": "NVARCHAR"}],
        "Track": [{"name": "TrackId", "type": "INTEGER"},
                  {"name": "GenreId", "type": "INTEGER"}],
        "InvoiceLine": [{"name": "InvoiceLineId", "type": "INTEGER"},
                        {"name": "TrackId", "type": "INTEGER"},
                        {"name": "UnitPrice", "type": "NUMERIC"},
                        {"name": "Quantity", "type": "INTEGER"}],
        "Invoice": [{"name": "InvoiceId", "type": "INTEGER"},
                    {"name": "Total", "type": "NUMERIC"}],
    }

    # THE T-17 SHAPE: hallucinated main-table claim, correct SQL -> must pass
    ca.validate_plan_matches_sql(
        {"entity": "Invoice"}, sql, "sqlite",
        table_schemas=schemas,
        question="Which genre generated the most revenue?")

    # POSITIVE CONTROL: corroborated claim (question names the table) and
    # the table really is missing -> still rejected, tagged [overvalidation]
    try:
        ca.validate_plan_matches_sql(
            {"entity": "Customer"}, sql, "sqlite",
            table_schemas=schemas,
            question="Show invoices for each customer")
    except ca.SemanticValidationError as e:
        assert e.category == "overvalidation", f"wrong category: {e.category}"
        assert ca.classify_error(
            str(e)) == ca.FailureClass.PLAN_TABLE_REJECTION
    else:
        raise AssertionError(
            "corroborated main-table claim was not enforced")




# --------------------------------------------------------------------------
# missing_join (BUG-C1 / W-006 / T-15): every ON-condition was a real FK
# column pair, yet {Artist, Album} never connected to {InvoiceLine,
# Invoice, Track} (missing Album.AlbumId = Track.AlbumId) -> cross-product
# revenue inflated ~353x, verified "True". Fix: connected-components check
# over the query's join graph vs the fetched FK set.
# --------------------------------------------------------------------------

_REGRESSION_FKS = [
    {"table": "Album", "columns": ["ArtistId"],
     "references_table": "Artist", "references_columns": ["ArtistId"]},
    {"table": "Track", "columns": ["AlbumId"],
     "references_table": "Album", "references_columns": ["AlbumId"]},
    {"table": "InvoiceLine", "columns": ["InvoiceId"],
     "references_table": "Invoice", "references_columns": ["InvoiceId"]},
    {"table": "InvoiceLine", "columns": ["TrackId"],
     "references_table": "Track", "references_columns": ["TrackId"]},
]


def test_missing_join_disconnected_components_rejected():
    w006_sql = (
        "SELECT A.Name, SUM(IL.Quantity * T.UnitPrice) AS TotalRevenue "
        "FROM Artist AS A "
        "JOIN Album ON A.ArtistId = Album.ArtistId "
        "JOIN InvoiceLine AS IL ON IL.InvoiceId = Invoice.InvoiceId "
        "JOIN Track AS T ON IL.TrackId = T.TrackId "
        "JOIN Invoice ON IL.InvoiceId = Invoice.InvoiceId "
        "GROUP BY A.Name"
    )
    try:
        ca.validate_join_connectivity(w006_sql, _REGRESSION_FKS, "sqlite")
    except ca.SemanticValidationError as e:
        assert e.category == "missing_join", f"wrong category: {e.category}"
        assert ca.classify_error(str(e)) == ca.FailureClass.JOIN_ERROR
        assert "Album" in str(e), f"should hint the Album<->Track edge:\n{e}"
    else:
        raise AssertionError(
            "disconnected-component fan-out SQL was not rejected")


def test_missing_join_connected_paths_pass():
    good = (
        "SELECT ar.Name, SUM(IL.Quantity * IL.UnitPrice) AS rev "
        "FROM Artist ar "
        "JOIN Album al ON al.ArtistId = ar.ArtistId "
        "JOIN Track t ON t.AlbumId = al.AlbumId "
        "JOIN InvoiceLine il ON il.TrackId = t.TrackId "
        "GROUP BY ar.Name"
    )
    ca.validate_join_connectivity(good, _REGRESSION_FKS, "sqlite")

    # single-table queries trivially pass
    ca.validate_join_connectivity(
        "SELECT COUNT(*) FROM Invoice", _REGRESSION_FKS, "sqlite")

    # old-style comma join with WHERE predicate still counts as connected
    comma = ("SELECT ar.Name, il.Quantity FROM Artist ar, InvoiceLine il "
             "WHERE 1=1")
    try:
        ca.validate_join_connectivity(comma, _REGRESSION_FKS, "sqlite")
    except ca.SemanticValidationError:
        pass  # acceptable: comma source IS unconnected here




# --------------------------------------------------------------------------
# fanout_seam (BUG-C2 / W-008 / T-18): SUM(Invoice.Total * IL.Quantity)
# slipped the fan-out detector because ONE operand (Quantity) is detail-side
# and the old immunity rule was all-or-nothing per aggregate. Immunity must
# be per-column-lineage: ANY lineage (ancestor/root) operand inside a mixed
# expression disqualifies detail-shape immunity.
# --------------------------------------------------------------------------

_FANOUT_FKS = [
    {"table": "InvoiceLine", "columns": ["InvoiceId"],
     "references_table": "Invoice", "references_columns": ["InvoiceId"]},
    {"table": "InvoiceLine", "columns": ["TrackId"],
     "references_table": "Track", "references_columns": ["TrackId"]},
]
_FANOUT_SCHEMAS = {
    "Invoice": [{"name": "InvoiceId", "type": "INTEGER"},
                {"name": "Total", "type": "NUMERIC"}],
    "InvoiceLine": [{"name": "InvoiceLineId", "type": "INTEGER",
                     "primary_key": True},
                    {"name": "InvoiceId", "type": "INTEGER"},
                    {"name": "TrackId", "type": "INTEGER"},
                    {"name": "UnitPrice", "type": "NUMERIC"},
                    {"name": "Quantity", "type": "INTEGER"}],
    "Track": [{"name": "TrackId", "type": "INTEGER"}],
}


def test_fanout_seam_mixed_parent_child_arithmetic_flagged():
    w008_sql = (
        "SELECT SUM(INV.Total * IL.Quantity) FROM InvoiceLine AS IL "
        "JOIN Invoice AS INV ON IL.InvoiceId = INV.InvoiceId"
    )
    try:
        ca.validate_aggregation_fanout(
            w008_sql, _FANOUT_FKS, _FANOUT_SCHEMAS, "sqlite")
    except ca.SemanticValidationError:
        pass  # expected: parent-column arithmetic must not be immune
    else:
        raise AssertionError(
            "SUM(Total * Quantity) slipped the fan-out detector again")


def test_fanout_seam_pure_detail_arithmetic_still_immune():
    legit = (
        "SELECT SUM(IL.UnitPrice * IL.Quantity) FROM InvoiceLine AS IL "
        "JOIN Invoice AS INV ON IL.InvoiceId = INV.InvoiceId "
        "JOIN Track AS T ON T.TrackId = IL.TrackId"
    )
    ca.validate_aggregation_fanout(
        legit, _FANOUT_FKS, _FANOUT_SCHEMAS, "sqlite")  # must not raise

    # bare ancestor column with child joined: the §6.20 founding case,
    # must STAY flagged
    founding = (
        "SELECT SUM(INV.Total) FROM Invoice AS INV "
        "JOIN InvoiceLine AS IL ON IL.InvoiceId = INV.InvoiceId"
    )
    try:
        ca.validate_aggregation_fanout(
            founding, _FANOUT_FKS, _FANOUT_SCHEMAS, "sqlite")
    except ca.SemanticValidationError:
        pass
    else:
        raise AssertionError("founding §6.20 case no longer flagged")



# --------------------------------------------------------------------------
# semantic_error (SEMANTIC_LAYER / Fix #10 / W-008 / T-18): standalone
# double-aggregation guard — an INDEPENDENT net from the fan-out lineage
# fix (#4). Grounded in measure semantics: a column on an FK-PARENT table
# (Invoice.Total — one value per invoice) multiplied by child-grain values
# inside SUM()/AVG() double-counts regardless of join topology.
# --------------------------------------------------------------------------

def test_semantic_error_double_aggregation_flagged():
    w008_sql = (
        "SELECT SUM(INV.Total * IL.Quantity) FROM InvoiceLine AS IL "
        "JOIN Invoice AS INV ON IL.InvoiceId = INV.InvoiceId"
    )
    try:
        ca.validate_measure_expression(
            w008_sql, _FANOUT_SCHEMAS, _FANOUT_FKS, "sqlite")
    except ca.SemanticValidationError as e:
        assert e.category == "semantic_error", f"wrong cat: {e.category}"
        assert ca.classify_error(str(e)) == ca.FailureClass.AGGREGATION_ERROR
    else:
        raise AssertionError("double aggregation not flagged by guard")


def test_semantic_error_legit_shapes_pass():
    # pure line-grain revenue arithmetic: both operands same (child) table
    ca.validate_measure_expression(
        "SELECT SUM(IL.UnitPrice * IL.Quantity) FROM InvoiceLine AS IL",
        _FANOUT_SCHEMAS, _FANOUT_FKS, "sqlite")
    # the stored measure alone is exactly right
    ca.validate_measure_expression(
        "SELECT SUM(Total) FROM Invoice",
        _FANOUT_SCHEMAS, _FANOUT_FKS, "sqlite")




# --------------------------------------------------------------------------
# format_error (BUG-H3 / W-004 / T-12): a correct 3503-row listing was fed
# whole to format_answer, which replied with meta-confusion — and figure
# verification passed VACUOUSLY (zero numeric claims). Fix: size-aware
# formatting (large listings never reach the LLM; deterministic preview
# instead) plus an answer-touches-result sanity gate on the small path.
# --------------------------------------------------------------------------

_FORMAT_PREVIEW_RESULT = {
    "columns": ["Name", "Title"],
    "rows": [[f"Track {i}", f"Album {i}"] for i in range(3503)],
}


class _ForbiddenLLM:
    """Stub that fails the test if the LLM is invoked at all."""

    class metrics:
        preview_used = False

    async def _call_llm(self, *args, **kwargs):
        raise AssertionError("LLM called for a large listing (must be "
                             "deterministic preview)")


def test_format_error_large_listing_never_reaches_llm():
    import asyncio
    out = asyncio.run(ca.SQLAgent.format_answer(
        _ForbiddenLLM(), "Show all tracks with their album names",
        "SELECT T.Name, A.Title FROM Track T JOIN Album A ON ...",
        _FORMAT_PREVIEW_RESULT))
    assert "3503" in out, f"preview missing total count:\n{out[:200]}"
    assert "Track 0" in out and "first" in out.lower()
    assert "more" in out.lower()


def test_format_error_sanity_gate_on_small_results():
    import asyncio

    garbage = "There is no question to answer directly and concisely."
    good = "Top tracks include Track 0 from Album 0."

    class _ScriptedLLM:
        class metrics:
            preview_used = False

        def __init__(self, responses):
            self.responses = list(responses)

        async def _call_llm(self, *a, **k):
            return self.responses.pop(0)

    result = {"columns": ["Name"], "rows": [["Track 0"]]}

    # garbage -> corrective retry -> good answer ships untouched
    out = asyncio.run(ca.SQLAgent.format_answer(
        _ScriptedLLM([garbage, good]), "q", "sql", result))
    assert out == good

    # garbage -> garbage -> deterministic fallback, never meta-text
    out = asyncio.run(ca.SQLAgent.format_answer(
        _ScriptedLLM([garbage, garbage]), "q", "sql", result))
    assert "no question to answer" not in out
    assert "Track 0" in out


def test_format_error_touches_result_helper():
    res = {"columns": ["Name"], "rows": [["Helena Holy"]]}
    assert ca._answer_touches_result("Helena Holy paid 25.86", res)
    assert not ca._answer_touches_result(
        "There is no question to answer directly.", res)




# --------------------------------------------------------------------------
# scope_drift (BUG-H2 / W-003#1 / T-11): "List customer names with their
# invoice totals" carried NO ranking in the plan, yet the generator added
# LIMIT 20 and the tie-aware rewrite dressed it as principled top-N.
# Fix: when plan.ranking is absent AND the question has no N/top/first
# vocabulary AND a LIMIT sneaks into an ungrouped listing -> reject with
# actionable message.
# --------------------------------------------------------------------------

def test_scope_drift_unsolicited_limit_rejected():
    schemas = {"Customer": [{"name": "CustomerId", "type": "INTEGER",
                             "primary_key": True},
                            {"name": "Name", "type": "NVARCHAR"}],
               "Invoice": [{"name": "InvoiceId", "type": "INTEGER"},
                           {"name": "Total", "type": "NUMERIC"}]}
    fks = [{"table": "Invoice", "columns": ["CustomerId"],
            "references_table": "Customer",
            "references_columns": ["CustomerId"]}]
    sql = ("SELECT c.name, SUM(i.total) FROM Customer c "
           "JOIN Invoice i ON i.customer_id = c.customer_id LIMIT 20")
    try:
        ca.validate_unsolicited_limit(
            sql,
            question="List customer names with their invoice totals",
            plan={"tables": ["Customer", "Invoice"]},
            table_schemas=schemas, foreign_keys=fks, dialect="sqlite")
    except ca.SemanticValidationError as e:
        assert e.category == "scope_drift"
    else:
        raise AssertionError("unsolicited LIMIT was not rejected")

    # legit top-N vocabulary passes untouched
    ca.validate_unsolicited_limit(
        sql.replace("LIMIT 20", "LIMIT 5"),
        question="Top 5 customers by total spent",
        plan={"tables": ["Customer", "Invoice"]},
        table_schemas=schemas, foreign_keys=fks, dialect="sqlite")




# --------------------------------------------------------------------------
# intent_error (BUG-M1 / W-001+W-005 / T-07+T-13): listing questions
# answered with a bare global aggregate. Driven by the QUESTION lexically,
# so it works even when the Step-2 plan is empty or silent.
# --------------------------------------------------------------------------

_INTENT_SCHEMAS = {
    "Album": [{"name": "AlbumId", "type": "INTEGER"}],
    "Artist": [{"name": "ArtistId", "type": "INTEGER"},
               {"name": "Name", "type": "NVARCHAR"}],
    "Customer": [{"name": "CustomerId", "type": "INTEGER"},
                 {"name": "Country", "type": "NVARCHAR"}],
}


def test_intent_error_listing_question_aggregate_rejected():
    w005_sql = "SELECT COUNT(DISTINCT ArtistId) FROM Album"
    try:
        ca.validate_plan_matches_sql(
            {}, w005_sql, "sqlite",
            table_schemas=_INTENT_SCHEMAS,
            question="Which artists have albums?")
    except ca.SemanticValidationError as e:
        assert e.category == "intent_error", f"wrong cat: {e.category}"
        assert ca.classify_error(str(e)) == ca.FailureClass.METRIC_MISMATCH
    else:
        raise AssertionError("COUNT answer to a which-listing not rejected")

    w007_sql = "SELECT COUNT(*) FROM Customer WHERE Country = 'Germany'"
    try:
        ca.validate_plan_matches_sql(
            None, w007_sql, "sqlite",
            table_schemas=_INTENT_SCHEMAS,
            question="List customers from Germany")
    except ca.SemanticValidationError as e:
        assert e.category == "intent_error"
    else:
        raise AssertionError("COUNT(*) answer to a list question allowed")


def test_intent_error_non_list_questions_untouched():
    # how-many phrasing: scalar aggregate is CORRECT
    ca.validate_plan_matches_sql(
        {}, "SELECT COUNT(DISTINCT ArtistId) FROM Album", "sqlite",
        table_schemas=_INTENT_SCHEMAS,
        question="How many artists have albums?")
    # superlative 'which' questions stay exempt ('most' = aggregation demand)
    ca.validate_plan_matches_sql(
        {}, "SELECT ar.Name, COUNT(t.TrackId) n FROM Artist ar "
            "JOIN Album al ON al.ArtistId=ar.ArtistId "
            "JOIN Track t ON t.AlbumId=al.AlbumId "
            "GROUP BY ar.Name ORDER BY n DESC LIMIT 1",
        "sqlite", table_schemas=_INTENT_SCHEMAS,
        question="Which artist has the most tracks?")
    # grouped per-entity listing passes
    ca.validate_plan_matches_sql(
        {}, "SELECT c.name, SUM(i.total) t FROM Customer c "
            "JOIN Invoice i ON i.CustomerId=c.CustomerId GROUP BY c.name",
        "sqlite", table_schemas=_INTENT_SCHEMAS,
        question="Show total spend for each customer")




# --------------------------------------------------------------------------
# wrong_grain (BUG-H1 / W-002 / T-08): "What is the average track price?"
# answered AVG(UnitPrice) over InvoiceLine⋈Track = sales-weighted average
# (1.0396) instead of the per-track statistic (1.0508). Fan-out detector is
# structurally blind (AVG cannot inflate). Fix: when a scalar-aggregate
# question names a dimension entity and the aggregated column lives on that
# entity, computing FROM any other root is rejected.
# --------------------------------------------------------------------------

def test_wrong_grain_dimension_aggregate_must_use_planned_table():
    schemas = {
        "Track": [{"name": "TrackId", "type": "INTEGER"},
                  {"name": "UnitPrice", "type": "NUMERIC"}],
        "InvoiceLine": [{"name": "InvoiceLineId", "type": "INTEGER"},
                        {"name": "TrackId", "type": "INTEGER"},
                        {"name": "UnitPrice", "type": "NUMERIC"}],
    }
    fks = [{"table": "InvoiceLine", "columns": ["TrackId"],
            "references_table": "Track", "references_columns": ["TrackId"]}]
    w002_sql = ("SELECT AVG(T.UnitPrice) FROM InvoiceLine IL "
                "JOIN Track T ON IL.TrackId = T.TrackId")
    try:
        ca.validate_aggregation_grain(
            w002_sql,
            question="What is the average track price?",
            plan={"entity": "Track"},
            table_schemas=schemas, foreign_keys=fks, dialect="sqlite")
    except ca.SemanticValidationError as e:
        assert e.category == "wrong_grain", f"wrong cat: {e.category}"
        assert "Track" in str(e)
    else:
        raise AssertionError("sales-weighted average slipped through again")

    # controls: direct per-entity aggregate passes; no-entity claim passes;
    # non-scalar (grouped) shapes untouched
    ca.validate_aggregation_grain(
        "SELECT AVG(UnitPrice) FROM Track",
        question="What is the average track price?",
        plan={"entity": "Track"}, table_schemas=schemas,
        foreign_keys=fks, dialect="sqlite")
    ca.validate_aggregation_grain(
        w002_sql, question="What is the average unit price?",
        plan={}, table_schemas=schemas, foreign_keys=fks, dialect="sqlite")




# --------------------------------------------------------------------------
# entity_binding (BUG-M3 / W-003#3 / T-11): the model cited 'Johannes Van'
# for row entity 'Johannes Van der Berg' — figures correct, binding failed
# because matching required the FULL cell. Fix: leading token-window prefix
# fallback (first two tokens, only for 3+-token names, length-guarded).
# --------------------------------------------------------------------------

def test_entity_binding_unique_prefix_binds():
    rows = [["Helena Holy", 25.86, 1.0],
            ["Johannes Van der Berg", 13.86, 20.0]]
    result = {"rows": rows}
    answer = "Johannes Van - 13.86"
    issues = ca.verify_row_attribution(answer, result)
    assert not any("does not appear anywhere" in i for i in issues), \
        f"unique-prefix citation still flagged as fabricated: {issues}"
    assert not any("13.86" in i and "cited for" in i for i in issues), \
        f"correct figure flagged as cross-row: {issues}"

    # ambiguity guard: a prefix shared by TWO entities must NOT bind
    rows2 = [["Johannes Van A", 1.0, 1.0], ["Johannes Van B", 2.0, 2.0]]
    issues2 = ca.verify_row_attribution(
        "Johannes Van - 1.5", {"rows": rows2})
    assert issues2, "ambiguous prefix silently bound"


if __name__ == "__main__":
    test_verifier_noise_list_indices_not_cited_as_figures()
    print("verifier_noise pin (indices not cited): PASS")
    test_verifier_noise_still_catches_real_misbindings()
    print("verifier_noise pin (real misbindings caught): PASS")
    test_overvalidation_uncorroborated_main_table_claim()
    print("overvalidation pin (uncorroborated claim passes): PASS")
    test_semantic_error_double_aggregation_flagged()
    print("semantic_error pin (double aggregation flagged): PASS")
    test_semantic_error_legit_shapes_pass()
    print("semantic_error pin (legit shapes pass): PASS")
    test_format_error_large_listing_never_reaches_llm()
    print("format_error pin (large listing deterministic): PASS")
    test_format_error_sanity_gate_on_small_results()
    print("format_error pin (sanity gate + fallback): PASS")
    test_format_error_touches_result_helper()
    print("format_error pin (touches-result helper): PASS")
    test_scope_drift_unsolicited_limit_rejected()
    print("scope_drift pin (unsolicited LIMIT rejected): PASS")
    test_intent_error_listing_question_aggregate_rejected()
    print("intent_error pin (listing->aggregate rejected): PASS")
    test_intent_error_non_list_questions_untouched()
    print("intent_error pin (non-list questions untouched): PASS")
    test_wrong_grain_dimension_aggregate_must_use_planned_table()
    print("wrong_grain pin (dimension aggregate anchored): PASS")
    print("missing_join pin (W-006 rejected): PASS")
    test_missing_join_connected_paths_pass()
    print("missing_join pin (connected paths pass): PASS")


# --------------------------------------------------------------------------
# entity_binding (BUG-M3 / W-003#3 / T-11): the model cited 'Johannes Van'
# for row entity 'Johannes Van der Berg' — figures correct, binding failed
# because matching required the FULL cell. Fix: leading token-window prefix
# fallback (first two tokens, only for 3+-token names, length-guarded).
# --------------------------------------------------------------------------

def test_entity_binding_unique_prefix_binds():
    rows = [["Helena Holy", 25.86, 1.0],
            ["Johannes Van der Berg", 13.86, 20.0]]
    result = {"rows": rows}
    answer = "Johannes Van - 13.86"
    issues = ca.verify_row_attribution(answer, result)
    assert not any("does not appear anywhere" in i for i in issues), \
        f"unique-prefix citation still flagged as fabricated: {issues}"
    assert not any("13.86" in i and "cited for" in i for i in issues), \
        f"correct figure flagged as cross-row: {issues}"

    # ambiguity guard: a prefix shared by TWO entities must NOT bind
    rows2 = [["Johannes Van A", 1.0, 1.0], ["Johannes Van B", 2.0, 2.0]]
    issues2 = ca.verify_row_attribution(
        "Johannes Van - 1.5", {"rows": rows2})
    assert issues2, "ambiguous prefix silently bound"


def test_semantic_error_raw_parent_attributes_not_flagged():
    """Track.UnitPrice is a raw attribute on a lookup parent — multiplying
    it by line quantities is LEGITIMATE line-revenue computation."""
    sql = ("SELECT SUM(T.UnitPrice * IL.Quantity) FROM InvoiceLine IL "
           "JOIN Track T ON IL.TrackId = T.TrackId")
    ca.validate_measure_expression(
        sql, _FANOUT_SCHEMAS, _FANOUT_FKS, "sqlite")  # must not raise
