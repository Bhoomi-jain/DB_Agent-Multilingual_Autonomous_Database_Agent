import asyncio
from core_agent import SQLAgent


class FakeMsg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    async def ainvoke(self, prompt):
        return FakeMsg(self.responses.pop(0))


# Reproduces the exact real failure shape: aliases used in SELECT/GROUP BY
# but never declared in FROM/JOIN. If repair works, this succeeds on the
# FIRST attempt — no retry needed, unlike the real run which failed 3x.
UNDECLARED_ALIAS_SQL = """```sql
SELECT C.name AS CustomerName, COUNT(OI.item_id) AS ItemCount
FROM order_items
JOIN orders ON order_items.order_id = orders.order_id
JOIN customers ON orders.customer_id = customers.customer_id
GROUP BY C.name
ORDER BY ItemCount DESC
```"""


async def main():
    llm = FakeLLM([
        UNDECLARED_ALIAS_SQL,
        "Alice has ordered the most items.",  # format_answer
    ])
    agent = SQLAgent(
        db_url="postgresql+psycopg2://postgres:postgres@localhost/testdb",
        llm=llm, dialect="PostgreSQL", max_retries=2, use_cache=False,
    )
    answer, sql, metrics = await agent.run("Which customer has ordered the most items?")

    print("ANSWER:", answer)
    print("REPAIRED SQL:", " ".join(sql.split()))
    print(metrics.summary())

    assert metrics.retries == 0, (
        f"expected 0 retries — repair should fix this on the FIRST attempt, "
        f"got {metrics.retries} retries (same as the unrepaired real failure)"
    )
    assert "AS C" in sql or "customers AS C" in sql

    print("\n--- undeclared-alias repair succeeded on first attempt, no retry "
          "needed (the real bug retried 3x identically without this fix) ---")


asyncio.run(main())