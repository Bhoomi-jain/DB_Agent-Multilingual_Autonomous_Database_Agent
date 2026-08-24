import asyncio
from sqlalchemy import create_engine, text
from core_agent import SQLAgent

from db_targets import PG_URL as DB_URL

EXTRA_TABLES = [f"artist_extra_{i}" for i in range(1, 8)]  # 7 unrelated tables


def setup_extra_tables():
    """Add 7 unrelated tables so this DB has 10 tables total (customers,
    orders, order_items + 7), matching Chinook's scale — reproduces the
    over-fetching scenario from the bug report."""
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        for t in EXTRA_TABLES:
            conn.execute(text(f"CREATE TABLE IF NOT EXISTS {t} (id SERIAL PRIMARY KEY, x TEXT)"))
        conn.commit()
    engine.dispose()


def teardown_extra_tables():
    """Self-contained: don't leave extra tables behind for other tests."""
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {', '.join(EXTRA_TABLES)} CASCADE"))
        conn.commit()
    engine.dispose()


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


GOOD_SQL = """```sql
SELECT c.name, SUM(oi.quantity * oi.unit_price) AS revenue
FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN order_items oi ON oi.order_id = o.order_id
WHERE o.status = 'completed'
GROUP BY c.name
ORDER BY revenue DESC
```"""

responses = [
    # 1. pick_relevant_tables + query plan (10 tables > 3, so this LLM call
    #    happens) — plan-object format since the merged plan call.
    '{"tables": ["customers", "orders", "order_items"]}',
    GOOD_SQL,                                    # 2. generate_sql
    "Alice generated the most revenue.",         # 3. format_answer
]


async def main():
    setup_extra_tables()
    try:
        llm = FakeLLM(responses)
        agent = SQLAgent(
            db_url=DB_URL,
            llm=llm, dialect="PostgreSQL", max_retries=1, use_cache=False,
        )
        answer, sql, metrics = await agent.run("Which customer generated the most revenue?")

        print("ANSWER:", answer)
        print(metrics.summary())

        assert metrics.tool_calls == 6, (
            f"expected 6 tool calls (schema filtering applied to FETCHING, not just "
            f"the prompt), got {metrics.tool_calls} — over-fetching bug is back!"
        )
        print(f"\nTool calls: {metrics.tool_calls} (would have been 13 on a 10-table DB "
              f"before the fix — describe_table is now only called for relevant tables)")
        print("--- over-fetching fix verified on a 10-table schema ---")
    finally:
        teardown_extra_tables()


asyncio.run(main())
