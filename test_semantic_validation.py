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


# First attempt: a hallucinated join — customers.customer_id has NO real FK
# relationship to order_items.item_id. Syntactically and read-only valid,
# but semantically nonsense — this is the exact shape of the real bug.
BAD_JOIN_SQL = """```sql
SELECT c.name, oi.product_name
FROM customers c
JOIN order_items oi ON c.customer_id = oi.item_id
```"""

# Second attempt (after semantic rejection feeds back the error): the
# correct 3-table join path.
GOOD_JOIN_SQL = """```sql
SELECT c.name, oi.product_name
FROM customers c
JOIN orders o ON o.customer_id = c.customer_id
JOIN order_items oi ON oi.order_id = o.order_id
```"""


async def main():
    llm = FakeLLM([
        BAD_JOIN_SQL,                              # generate_sql attempt 1: hallucinated join
        GOOD_JOIN_SQL,                              # generate_sql attempt 2: corrected after semantic rejection
        "Each customer's ordered products are listed.",  # format_answer
    ])
    agent = SQLAgent(
        db_url="postgresql+psycopg2://postgres:postgres@localhost/testdb",
        llm=llm, dialect="PostgreSQL", max_retries=2, use_cache=False,
    )
    answer, sql, metrics = await agent.run("Which products did each customer order?")

    print("ANSWER:", answer)
    print("FINAL SQL:", " ".join(sql.split()))
    print(metrics.summary())

    assert "orders" in sql, "should have recovered to the correct 3-table join"
    assert metrics.semantic_rejections == 1, f"expected 1 semantic rejection, got {metrics.semantic_rejections}"
    assert metrics.retries == 1

    print("\n--- semantic validation correctly rejected the hallucinated join and "
          "the retry loop recovered with the real join path ---")


asyncio.run(main())