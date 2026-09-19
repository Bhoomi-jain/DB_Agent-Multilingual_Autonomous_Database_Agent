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


ALWAYS_BROKEN = "```sql\nSELECT this_column_never_exists FROM customers\n```"


async def main():
    # 3 identical broken responses: pick_tables skipped (3 tables), so this
    # covers attempt 0, 1, 2 (max_retries=2 -> 3 total attempts)
    llm = FakeLLM([ALWAYS_BROKEN, ALWAYS_BROKEN, ALWAYS_BROKEN])
    agent = SQLAgent(
        db_url=PG_URL,
        llm=llm, dialect="PostgreSQL", max_retries=2, use_cache=False,
    )
    try:
        await agent.run("This will never succeed")
        print("FAIL: expected a RuntimeError to be raised")
    except RuntimeError as e:
        msg = str(e)
        print("Correctly raised after exhausting retries:", msg[:100])
        # Distinguish genuine retry-loop exhaustion from a schema-discovery
        # crash: run() connects and lists tables BEFORE entering the retry
        # loop, and both failures surface here as a plain RuntimeError —
        # indistinguishable from this vantage point unless we check the
        # message. Only the retry loop produces "Failed after N attempt(s)".
        assert "Failed after 3 attempt(s)" in msg, (
            f"expected the retry loop's exhaustion message, got something else "
            f"(likely a schema-discovery failure wearing the same exception "
            f"type, which would make the retries == 2 assertion meaningless): "
            f"{msg[:200]}"
        )
        assert agent.metrics.retries == 2, f"expected 2 retries, got {agent.metrics.retries}"
        assert len(llm.calls) == 3, f"expected exactly 3 generate_sql calls, got {len(llm.calls)}"
        print("Retry count and call count correct — no infinite loop, no silent failure")


asyncio.run(main())
