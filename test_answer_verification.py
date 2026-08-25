import asyncio
from core_agent import FailureClass, SQLAgent
from db_targets import PG_URL


class FakeMsg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    async def ainvoke(self, prompt):
        return FakeMsg(self.responses.pop(0))


class ShapeLLM:
    """Routes by PROMPT SHAPE (see test_sqlite_repro): generate_sql vs
    format_answer, so a corrective-retry path can't desync a positional
    script. Each format_answer call pops the next scripted answer."""
    def __init__(self, sql_response, format_responses):
        self.sql_response = sql_response
        self.format_responses = list(format_responses)

    async def ainvoke(self, prompt):
        if "Answer the question directly" in prompt:
            return FakeMsg(self.format_responses.pop(0))
        return FakeMsg(self.sql_response)


GOOD_SQL = """```sql
SELECT name, country FROM customers WHERE country = 'Canada'
```"""


async def main():
    llm = FakeLLM([
        GOOD_SQL,
        # format_answer attempt 1: hallucinates a fabricated "correction" note
        # and a wrong figure, exactly like the real observed failure.
        "Alice is from Canada (note: this seems like it could be a typo, "
        "possibly should be 15 customers, not 1).",
        # format_answer attempt 2 (after verification forces a retry):
        # correct, faithful answer.
        "Alice is the customer from Canada.",
    ])
    agent = SQLAgent(
        db_url=PG_URL,
        llm=llm, dialect="PostgreSQL", max_retries=1, use_cache=False,
    )
    answer, sql, metrics = await agent.run("Which customers are from Canada?")

    print("ANSWER:", answer)
    print(metrics.summary())

    assert "15" not in answer, "hallucinated number should not survive to final answer"
    assert metrics.verification_retries == 1
    assert metrics.answer_verified is True

    print("\n--- answer verification correctly caught the hallucinated figure "
          "and forced a corrective retry ---")

    await scenario_row_attribution_tie_pick()
    await scenario_row_attribution_cross_row_figure()
    await scenario_row_attribution_clean_listing_passes()


# Row-aware attribution verification (verify_row_attribution). Fixture
# spends: Alice 109.91, Bob 99.98, Carol 9.99 — three distinct rows, so a
# figure cited for the wrong customer is detectable as cross-row.
SPEND_SQL = """```sql
SELECT cu.name, SUM(oi.quantity * oi.unit_price) AS spend
FROM customers cu
JOIN orders o ON cu.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY cu.name
ORDER BY spend DESC
```"""

# Three rows each ending in rank 1 — the shape rewrite_top_n_with_ties
# produces when several rows tie for the top spot. No ORDER BY/LIMIT so
# the rewrite itself stays out of the way; the literal column just makes
# the result look like its output.
TIED_SQL = """```sql
SELECT name, country, 1 AS __rnk FROM customers
```"""


async def scenario_row_attribution_tie_pick():
    """58-ties live failure in miniature: query returns several top-ranked
    rows, formatter names ONE arbitrary winner. Figures all trace (flat
    verification passes!) but attribution must flag it and force a retry."""
    llm = ShapeLLM(
        TIED_SQL,
        ["Bob from the USA is the top customer.",                       # bad: picks one of 3 tied
         "All three customers are tied: Alice, Bob, and Carol."],       # corrected
    )
    agent = SQLAgent(db_url=PG_URL, llm=llm, dialect="PostgreSQL",
                     max_retries=1, use_cache=False)
    answer, sql, metrics = await agent.run("Who is the top customer?")

    print("\nANSWER:", answer)
    print(metrics.summary())
    assert FailureClass.ROW_ATTRIBUTION_ERROR in metrics.failure_classes, (
        f"tie-pick should tag ROW_ATTRIBUTION_ERROR, got {metrics.failure_classes}"
    )
    assert metrics.verification_retries == 1
    assert metrics.answer_verified is True
    assert "⚠️" not in answer, "corrected answer must not carry the warning banner"
    print("--- tie arbitrary-pick caught, corrective retry reports the tie ---")


async def scenario_row_attribution_cross_row_figure():
    """The core row-binding gap: every number exists SOMEWHERE (flat check
    passes) but not in the cited entity's row. Must be flagged + retried."""
    llm = ShapeLLM(
        SPEND_SQL,
        ["Carol spent 99.98 in total.",                                 # Bob's figure on Carol's row
         "Carol spent 9.99 in total."],                                 # her real figure
    )
    agent = SQLAgent(db_url=PG_URL, llm=llm, dialect="PostgreSQL",
                     max_retries=1, use_cache=False)
    answer, sql, metrics = await agent.run("What did each customer spend in total?")

    print("\nANSWER:", answer)
    assert FailureClass.ROW_ATTRIBUTION_ERROR in metrics.failure_classes
    assert metrics.verification_retries == 1
    assert metrics.answer_verified is True
    assert "9.99" in answer and "99.98" not in answer
    print("--- cross-row figure caught and corrected to the entity's own value ---")


async def scenario_row_attribution_clean_listing_passes():
    """Anti-false-positive guard (§6.22 lesson): a legitimate listing citing
    each entity's OWN figures must pass with zero retries."""
    llm = ShapeLLM(
        SPEND_SQL,
        ["Alice spent 109.91, Bob spent 99.98, and Carol spent 9.99."],
    )
    agent = SQLAgent(db_url=PG_URL, llm=llm, dialect="PostgreSQL",
                     max_retries=1, use_cache=False)
    answer, sql, metrics = await agent.run("What did each customer spend in total?")

    print("\nANSWER:", answer)
    assert metrics.answer_verified is True
    assert metrics.verification_retries == 0, (
        f"clean listing must not trigger verification, got {metrics.verification_retries}"
    )
    assert FailureClass.ROW_ATTRIBUTION_ERROR not in metrics.failure_classes, (
        f"false positive: {metrics.failure_classes}"
    )
    print("--- clean per-entity listing passes without false positives ---")


asyncio.run(main())