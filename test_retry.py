import asyncio
from core_agent import SQLAgent
from db_targets import PG_URL


class FakeMsg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def ainvoke(self, prompt):
        self.calls.append(prompt)
        return FakeMsg(self.responses.pop(0))


BAD_SQL = """```sql
SELECT c.name, SUM(oi.quantity * oi.unit_price) AS revenue
FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN order_items oi ON oi.order_id = o.order_id
WHERE o.status = 'completed'
GROUP BY c.name, c.no_such_column
```"""

GOOD_SQL = """```sql
SELECT c.name, SUM(oi.quantity * oi.unit_price) AS revenue
FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN order_items oi ON oi.order_id = o.order_id
WHERE o.status = 'completed'
GROUP BY c.name
ORDER BY revenue DESC
```"""

responses = [
    # Note: pick_relevant_tables is SKIPPED (no LLM call) because this test
    # schema has only 3 tables <= the no-op threshold — that's the
    # schema-filtering cost optimization working as intended, not a bug.
    BAD_SQL,                                     # 1. generate_sql attempt 1 (bad column -> DB error)
    GOOD_SQL,                                     # 2. generate_sql attempt 2 (corrected)
    "Alice generated the most revenue among completed orders.",  # 3. format_answer
]


async def main():
    llm = FakeLLM(responses)
    agent = SQLAgent(
        db_url=PG_URL,
        llm=llm, dialect="PostgreSQL", max_retries=2, use_cache=True,
    )
    answer, sql, metrics = await agent.run("Which customer generated the most revenue?")
    print("ANSWER:", answer)
    print("SQL (final):", " ".join(sql.split()))
    print("LLM calls made:", len(llm.calls), "(expected 3 — table-pick skipped for a 3-table schema)")
    print(metrics.summary())
    assert metrics.retries == 1, f"expected 1 retry, got {metrics.retries}"
    assert metrics.llm_calls == 3
    # Fine-grained caching: 1 miss for get_tables + 3 misses for each
    # table's describe_table + 1 miss for get_foreign_keys = 5 total.
    assert metrics.cache_misses == 5 and metrics.cache_hits == 0
    assert metrics.tool_calls > 0
    assert "Alice" in answer
    print("\n--- retry-loop + real DB execution verified correct ---")


asyncio.run(main())
