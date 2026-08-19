import asyncio
from core_agent import SQLAgent


class FakeMsg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return FakeMsg(self.responses.pop(0))


# Attempt 1: THE exact bug described — "average per customer" but no
# GROUP BY, collapsing every order into one overall average.
BAD_SQL = """```sql
SELECT AVG(order_id) AS avg_val FROM orders
```"""

# Attempt 2: correctly grouped, after the specific error is fed back.
GOOD_SQL = """```sql
SELECT customer_id, AVG(order_id) AS avg_val FROM orders GROUP BY customer_id
```"""


async def main():
    llm = FakeLLM([
        BAD_SQL,
        GOOD_SQL,
        "Here is the average order ID per customer.",  # format_answer
    ])
    agent = SQLAgent(
        db_url="postgresql+psycopg2://postgres:postgres@localhost/testdb",
        llm=llm, dialect="PostgreSQL", max_retries=1, use_cache=False,
    )
    answer, sql, metrics = await agent.run("What is the average order ID per customer?")

    print("ANSWER:", answer)
    print("FINAL SQL:", " ".join(sql.split()))
    print(metrics.summary())

    assert "GROUP BY" in sql.upper(), "final query should be grouped"
    assert metrics.retries == 1, f"expected 1 retry (ungrouped -> grouped), got {metrics.retries}"

    # Confirm the retry prompt actually named the specific missing grouping term
    retry_prompt = llm.prompts[1]
    assert "customer" in retry_prompt.lower()
    assert "GROUP BY" in retry_prompt

    print("\n--- grouping-intent validation caught the ungrouped aggregate and "
          "forced a corrective retry with a specific, actionable error ---")


asyncio.run(main())
