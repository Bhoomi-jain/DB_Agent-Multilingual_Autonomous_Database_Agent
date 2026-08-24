import asyncio

import sqlglot
from sqlglot import exp

from core_agent import SQLAgent
from db_targets import PG_URL


class FakeMsg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    async def ainvoke(self, prompt):
        return FakeMsg(self.responses.pop(0))


# Reproduces the exact real failure: a direct join between two tables that
# aren't actually related, skipping the real bridge table 'orders'.
BAD_JOIN_SQL = """```sql
SELECT customers.name, COUNT(*) AS item_count
FROM customers
JOIN order_items ON customers.customer_id = order_items.order_id
GROUP BY customers.name
ORDER BY item_count DESC
LIMIT 5
```"""


async def main():
    llm = FakeLLM([
        BAD_JOIN_SQL,
        "Alice has the most items.",  # format_answer
    ])
    agent = SQLAgent(
        db_url=PG_URL,
        llm=llm, dialect="PostgreSQL", max_retries=2, use_cache=False,
    )
    answer, sql, metrics = await agent.run("Which customer has ordered the most items?")

    print("ANSWER:", answer)
    print("REPAIRED SQL:", " ".join(sql.split()))
    print(metrics.summary())

    # AST-based, not substring-based: `assert "orders" in sql` was trivially
    # true even WITHOUT repair — "order_items" contains the substring
    # "orders". The real contract is that a JOIN clause for the orders table
    # exists in the final query tree.
    ast = sqlglot.parse_one(sql, read="postgres")
    joined_tables = [j.this.name for j in ast.find_all(exp.Join) if isinstance(j.this, exp.Table)]
    assert "orders" in joined_tables, (
        f"the bridge table 'orders' should have been inserted as an actual "
        f"JOIN by join-path repair; joins found: {joined_tables}"
    )
    assert metrics.semantic_rejections == 1, "semantic validation should have caught the bad join"
    assert metrics.retries == 0, (
        f"expected 0 retries — join-path repair should fix this WITHIN the "
        f"same attempt, not require the model to try again. Got {metrics.retries}."
    )

    print("\n--- join-path repair fixed the hallucinated join within the same "
          "attempt: 1 semantic rejection, 0 retries needed ---")


asyncio.run(main())