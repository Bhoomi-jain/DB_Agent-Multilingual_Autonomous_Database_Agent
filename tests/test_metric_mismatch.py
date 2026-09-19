import asyncio
from core_agent import FailureClass, SQLAgent
from db_targets import SQLITE_URL


class FakeMsg:
    def __init__(self, content):
        self.content = content


class RoutingLLM:
    """Responds by PROMPT SHAPE, not call order (see test_sqlite_repro):
    pick_relevant_tables / generate_sql / format_answer are distinguishable
    by their prompt text, so every retry path stays valid without a brittle
    pop-in-order script."""
    def __init__(self, bad_sql, good_sql, plan_json=None, answer="Done."):
        self.bad_sql = bad_sql
        self.good_sql = good_sql
        self.plan_json = plan_json
        self.answer = answer
        self.generate_calls = 0

    async def ainvoke(self, prompt):
        if "Answer the question directly" in prompt:
            return FakeMsg(self.answer)                       # format_answer
        if self.plan_json is not None and '"metric"' in prompt \
                and "Write a single read-only" not in prompt:
            return FakeMsg(self.plan_json)                    # pick_relevant_tables
        self.generate_calls += 1
        # First generation tries the wrong-measure ranking; every later
        # generation (i.e. after the actionable rejection) behaves.
        return FakeMsg(self.bad_sql if self.generate_calls == 1
                       else self.good_sql)                    # generate_sql


BAD_RANKING_SQL = """```sql
SELECT c.name, SUM(oi.quantity) AS TotalQty
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY c.name
ORDER BY TotalQty DESC
LIMIT 1
```"""

GOOD_PRICE_SQL = """```sql
SELECT c.name, oi.unit_price AS PricePaid
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
WHERE oi.unit_price = (SELECT MAX(unit_price) FROM order_items)
ORDER BY oi.unit_price DESC
LIMIT 1
```"""

# The observed live failure shape: "most expensive X" answered by whoever
# bought the MOST ITEMS. Structurally valid SQL, real FK joins, verified
# numbers — just the wrong measure entirely.
QUESTION = "Which customer bought the most expensive item?"


async def scenario_a_ranking_target_rejected():
    llm = RoutingLLM(BAD_RANKING_SQL, GOOD_PRICE_SQL,
                     answer="Bob bought the most expensive item.")
    agent = SQLAgent(db_url=SQLITE_URL, llm=llm, dialect="SQLite",
                     max_retries=1, use_cache=False)
    answer, sql, metrics = await agent.run(QUESTION)

    print("SQL:", sql)
    print(metrics.summary())
    assert "unit_price" in sql.lower(), f"retry should rank by price, got: {sql!r}"
    assert metrics.failure_classes.get(FailureClass.METRIC_MISMATCH, 0) >= 1, (
        f"expected METRIC_MISMATCH to be tagged, got: {metrics.failure_classes}"
    )
    assert metrics.retries == 1, (
        f"the rejection must feed exactly one LLM retry, got {metrics.retries}"
    )
    assert metrics.answer_verified is True
    print("--- A: wrong-measure ranking rejected with METRIC_MISMATCH, "
          "actionable retry fixed it ---\n")


PLAN_WITH_BOGUS_METRIC = (
    '{"tables": ["customers", "orders", "order_items", "audit_log"], '
    '"metric": "MAX", "metric_column": "amount", "entity": null, '
    '"ranking": {"enabled": true, "direction": "DESC", "limit": 1}, '
    '"grouping": null}'
)


async def scenario_b_plan_metric_discarded_inferred_fallback():
    """The exact production gap: >3 tables forces the plan step, the plan
    hallucinates metric=MAX/'amount', the corroboration gate discards it —
    and validation must STILL reject the unrelated ranking via the inferred
    price dimension instead of passing silently like it used to.

    The fixture has exactly 3 base tables and the plan step is skipped at
    <=3, so a throwaway 4th table pushes past the skip (created here,
    dropped in finally — §4.2 rule: self-contained tests sweep their own
    extras; seed_testdb also sweeps leftovers)."""
    import sqlite3
    conn = sqlite3.connect(SQLITE_URL.split("///", 1)[1])
    try:
        conn.execute("CREATE TABLE audit_log "
                     "(id INTEGER PRIMARY KEY, note TEXT)")
        conn.commit()
    finally:
        conn.close()

    llm = RoutingLLM(BAD_RANKING_SQL, GOOD_PRICE_SQL,
                     plan_json=PLAN_WITH_BOGUS_METRIC,
                     answer="Bob bought the most expensive item.")
    agent = SQLAgent(db_url=SQLITE_URL, llm=llm, dialect="SQLite",
                     max_retries=1, use_cache=False)
    try:
        answer, sql, metrics = await agent.run(QUESTION)
    finally:
        conn = sqlite3.connect(SQLITE_URL.split("///", 1)[1])
        try:
            conn.execute("DROP TABLE IF EXISTS audit_log")
            conn.commit()
        finally:
            conn.close()

    print("SQL:", sql)
    print(metrics.summary())
    # Proof the plan step RAN: 4 LLM calls (plan + 2 generates + format),
    # not 3 — otherwise this scenario proves nothing about the fallback.
    assert metrics.llm_calls == 4, (
        f"plan step should have executed (>3 tables), got {metrics.llm_calls} LLM calls"
    )
    assert metrics.failure_classes.get(FailureClass.METRIC_MISMATCH, 0) >= 1
    assert "unit_price" in sql.lower()
    print("--- B: discarded plan metric replaced by inferred dimension, "
          "bad ranking still caught ---\n")


FAIL_OPEN_SQL = """```sql
SELECT name FROM customers LIMIT 5
```"""


async def scenario_c_fail_open_without_recognized_dimension():
    """Anti-regression guard (§6.24a rule): a superlative question with NO
    recognizable measure vocabulary must NOT be rejected — word-only
    signals killed three correct attempts once before."""
    llm = RoutingLLM(FAIL_OPEN_SQL, FAIL_OPEN_SQL,
                     answer="Here are some customers.")
    agent = SQLAgent(db_url=SQLITE_URL, llm=llm, dialect="SQLite",
                     max_retries=1, use_cache=False)
    answer, sql, metrics = await agent.run("Which customer is the best?")
    print(metrics.summary())
    assert metrics.failure_classes.get(FailureClass.METRIC_MISMATCH, 0) == 0, (
        f"fail-open violated: {metrics.failure_classes}"
    )
    assert metrics.retries == 0
    print("--- C: unrecognizable dimension fails open, zero false "
          "rejections ---\n")


# The §6.28 pin that FAILED TO FIRE before the agg-shape check: the alias
# 'total_price' segment-hits the price family, so family-level validation
# passed while the query actually ranked customers by SUMMED prices —
# totaling prices is not locating the most expensive single one.
SUM_DRIFT_SQL = """```sql
SELECT c.name, SUM(oi.unit_price) AS total_price
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY c.name
ORDER BY total_price DESC
LIMIT 1
```"""


async def scenario_d_sum_drift_rejected():
    """Hard metric enforcement: "most expensive" demands MAX over price.
    SUM(unit_price) per customer family-hits via its alias and slipped
    through every earlier layer; only the aggregate shape betrays it."""
    llm = RoutingLLM(SUM_DRIFT_SQL, GOOD_PRICE_SQL,
                     answer="Bob bought the most expensive item.")
    agent = SQLAgent(db_url=SQLITE_URL, llm=llm, dialect="SQLite",
                     max_retries=1, use_cache=False)
    answer, sql, metrics = await agent.run(QUESTION)

    print("SQL:", sql)
    print(metrics.summary())
    assert metrics.failure_classes.get(FailureClass.METRIC_MISMATCH, 0) >= 1, (
        f"SUM-drift ranking must be rejected as METRIC_MISMATCH, "
        f"got: {metrics.failure_classes}"
    )
    assert metrics.retries == 1, (
        f"exactly one corrective retry expected, got {metrics.retries}"
    )
    assert "unit_price" in sql.lower() and "sum(" not in sql.lower(), (
        f"retry must drop the SUM drift, got: {sql!r}"
    )
    assert metrics.answer_verified is True
    print("--- D: SUM(price) drift rejected by aggregate-shape check, "
          "retry ranks by the price column itself ---\n")


PLAN_WITH_TRUSTED_MAX = (
    '{"tables": ["customers", "orders", "order_items", "audit_log"], '
    '"metric": "MAX", "metric_column": "unit_price", "entity": null, '
    '"ranking": {"enabled": true, "direction": "DESC", "limit": 1}, '
    '"grouping": null}'
)


async def scenario_e_plan_max_survives_gate_and_enforced():
    """§6.28 Fix B pin: pre-fix, a CORRECT plan claiming metric=MAX for a
    "most expensive" question was discarded at the corroboration gate —
    money/count vocabulary doesn't cover price superlatives. Post-fix the
    question's own polarity corroborates the claim, so plan['metric']
    survives (unit assertion) and validate_plan_matches_sql hard-enforces
    it downstream (integration tail)."""
    llm = RoutingLLM(BAD_RANKING_SQL, GOOD_PRICE_SQL,
                     plan_json=PLAN_WITH_TRUSTED_MAX,
                     answer="Bob bought the most expensive item.")
    agent = SQLAgent(db_url=SQLITE_URL, llm=llm, dialect="SQLite",
                     max_retries=1, use_cache=False)

    # Unit half: call the Step-2 planner directly with >3 tables so the
    # skip doesn't fire, then inspect what the gate decided to trust.
    tables, plan = await agent.pick_relevant_tables(
        QUESTION, ["customers", "orders", "order_items", "audit_log"])
    print("plan:", plan)
    assert plan.get("metric") == "MAX", (
        f"polarity-corroborated metric must survive the gate, got: {plan}"
    )
    assert "metric_inferred" not in plan, (
        f"metric should be TRUSTED, not downgraded to inferred: {plan}"
    )

    # Integration half (fresh agent — metrics must reflect ONLY this run):
    # same trusted plan inside a full run (>3 real tables), where the
    # surviving metric gets enforced against SQL.
    import sqlite3
    conn = sqlite3.connect(SQLITE_URL.split("///", 1)[1])
    try:
        conn.execute("CREATE TABLE audit_log "
                     "(id INTEGER PRIMARY KEY, note TEXT)")
        conn.commit()
    finally:
        conn.close()
    llm2 = RoutingLLM(BAD_RANKING_SQL, GOOD_PRICE_SQL,
                      plan_json=PLAN_WITH_TRUSTED_MAX,
                      answer="Bob bought the most expensive item.")
    agent = SQLAgent(db_url=SQLITE_URL, llm=llm2, dialect="SQLite",
                     max_retries=1, use_cache=False)
    try:
        answer, sql, metrics = await agent.run(QUESTION)
    finally:
        conn = sqlite3.connect(SQLITE_URL.split("///", 1)[1])
        try:
            conn.execute("DROP TABLE IF EXISTS audit_log")
            conn.commit()
        finally:
            conn.close()

    print("SQL:", sql)
    print(metrics.summary())
    assert metrics.llm_calls == 4, (
        f"plan step should have executed (>3 tables), got {metrics.llm_calls} LLM calls"
    )
    assert metrics.failure_classes.get(FailureClass.METRIC_MISMATCH, 0) >= 1
    assert "unit_price" in sql.lower()
    assert metrics.answer_verified is True
    print("--- E: plan's correct MAX survives corroboration and is "
          "hard-enforced on SQL ---\n")


CHEAPEST_MIN_SQL = """```sql
SELECT c.name, MIN(oi.unit_price) AS MinPrice
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY c.name
ORDER BY MinPrice ASC
LIMIT 1
```"""

AVG_PRICE_SQL = """```sql
SELECT AVG(oi.unit_price) AS AvgPrice FROM order_items
```"""


async def scenario_f_anti_false_positive_guards():
    """§6.24a discipline pinned: polarity enforcement must never reject a
    conforming or out-of-scope query. (1) cheapest+MIN passes; (2) an
    average-price question (no ranking language) is untouched; (3) the
    canonical bare-column form for most-expensive passes zero-retry."""
    llm = RoutingLLM(CHEAPEST_MIN_SQL, CHEAPEST_MIN_SQL,
                     answer="The cheapest item went for a low price.")
    agent = SQLAgent(db_url=SQLITE_URL, llm=llm, dialect="SQLite",
                     max_retries=1, use_cache=False)
    _, _, m1 = await agent.run("Which customer paid the cheapest price?")
    print(m1.summary())
    assert FailureClass.METRIC_MISMATCH not in m1.failure_classes
    assert m1.retries == 0

    llm = RoutingLLM(AVG_PRICE_SQL, AVG_PRICE_SQL,
                     answer="Here is the average price.")
    agent = SQLAgent(db_url=SQLITE_URL, llm=llm, dialect="SQLite",
                     max_retries=1, use_cache=False)
    _, _, m2 = await agent.run(
        "What is the average unit price across all order items?")
    print(m2.summary())
    assert FailureClass.METRIC_MISMATCH not in m2.failure_classes
    assert m2.retries == 0

    llm = RoutingLLM(GOOD_PRICE_SQL, GOOD_PRICE_SQL,
                     answer="Bob bought the most expensive item.")
    agent = SQLAgent(db_url=SQLITE_URL, llm=llm, dialect="SQLite",
                     max_retries=1, use_cache=False)
    _, _, m3 = await agent.run(QUESTION)
    print(m3.summary())
    assert m3.retries == 0, (
        f"canonical bare-column form must pass untouched, "
        f"got {m3.retries} retries: {m3.failure_classes}"
    )
    assert m3.answer_verified is True
    print("--- F: MIN-form, non-ranking AVG question, and bare-column "
          "form all fail open ---\n")


async def main():
    await scenario_a_ranking_target_rejected()
    await scenario_b_plan_metric_discarded_inferred_fallback()
    await scenario_c_fail_open_without_recognized_dimension()
    await scenario_d_sum_drift_rejected()
    await scenario_e_plan_max_survives_gate_and_enforced()
    await scenario_f_anti_false_positive_guards()


asyncio.run(main())
