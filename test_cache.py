import asyncio
from core_agent import SQLAgent
from db_targets import pick_backend


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

# The caching assertions below (5 misses on the first run; 5 hits + a single
# tool call on the second) exercise SchemaCache, not any particular backend —
# the schema shape is identical on any engine with the three baseline tables.
# pick_backend() (in db_targets.py) prefers MySQL when its server is up and
# falls back to the Postgres testdb the rest of the suite uses otherwise,
# rather than failing this test on a missing server. GOOD_SQL is portable
# across both.


async def run_once(db_url, dialect):
    llm = FakeLLM([GOOD_SQL, "Alice generated the most revenue."])
    agent = SQLAgent(db_url=db_url, llm=llm, dialect=dialect, max_retries=1, use_cache=True)
    answer, sql, metrics = await agent.run("Which customer generated the most revenue?")
    return answer, metrics


async def main():
    db_url, dialect, label = pick_backend()

    print(f"=== {label}: first run (expect cache MISSES) ===")
    answer1, m1 = await run_once(db_url, dialect)
    print("answer:", answer1)
    print(m1.summary())
    # Fine-grained caching: get_tables + 3x describe_table + get_foreign_keys = 5 misses
    assert m1.cache_misses == 5 and m1.cache_hits == 0
    assert m1.tool_calls >= 5  # list_tables + 3x describe_table + list_foreign_keys + run_query

    print(f"\n=== {label}: second run (expect cache HIT, fewer tool calls) ===")
    answer2, m2 = await run_once(db_url, dialect)
    print("answer:", answer2)
    print(m2.summary())
    # 5 hits now: get_tables + 3x describe_table + get_foreign_keys, all from cache
    assert m2.cache_hits == 5 and m2.cache_misses == 0
    assert m2.tool_calls == 1, f"expected only run_query (1 tool call), got {m2.tool_calls}"
    assert "Alice" in answer1 and "Alice" in answer2

    print(f"\nTool calls saved by caching: {m1.tool_calls - m2.tool_calls}")
    print("--- caching verified: second run skipped schema discovery entirely ---")


asyncio.run(main())
