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


# Reproduces the exact real failure: references order_items (a real table)
# without ever joining it — a column-count mismatch, not an undefined alias.
MISSING_JOIN_SQL = """```sql
SELECT c.name, COUNT(order_items.item_id) AS item_count
FROM customers c
JOIN orders o ON o.customer_id = c.customer_id
GROUP BY c.name
ORDER BY item_count DESC
```"""


async def main():
    llm = FakeLLM([
        MISSING_JOIN_SQL,
        "Alice has ordered the most items.",  # format_answer
    ])
    agent = SQLAgent(
        db_url="postgresql+psycopg2://postgres:postgres@localhost/testdb",
        llm=llm, dialect="PostgreSQL", max_retries=2, use_cache=False,
    )
    answer, sql, metrics = await agent.run("Which customer has the most order items?")

    print("ANSWER:", answer)
    print("REPAIRED SQL:", " ".join(sql.split()))
    print(metrics.summary())

    assert "JOIN order_items" in sql or "order_items" in sql.split("FROM")[1]
    assert metrics.retries == 0, (
        f"expected 0 retries — missing-join repair should fix this WITHIN "
        f"the same attempt, got {metrics.retries}"
    )

    print("\n--- missing-join repair correctly added the JOIN for a "
          "referenced-but-never-joined real table, within the same attempt ---")


asyncio.run(main())