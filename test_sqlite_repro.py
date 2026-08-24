import asyncio
from core_agent import SQLAgent
from db_targets import SQLITE_URL


class FakeMsg:
    def __init__(self, content):
        self.content = content


class RoutingLLM:
    """Responds by PROMPT SHAPE, not call order.

    A pop-in-order script can't cover both the success path (2 LLM calls:
    generate, format) and any-retry path (3+: generate, generate, format)
    — whichever entry was meant for the missed call poisons the rest. That
    is exactly how this test's original 2-response script fed its ANSWER
    STRING to the SQL parser after a failed first attempt ("Could not
    parse SQL: There is 1 customer from Canada."). Routing on what each
    step's prompt looks like makes every path produce valid responses.
    """
    def __init__(self):
        self.calls = []

    async def ainvoke(self, prompt):
        self.calls.append(prompt)
        if "Answer the question directly" in prompt:
            return FakeMsg("There is 1 customer from Canada.")   # format_answer
        return FakeMsg(GOOD_SQL)                                 # generate_sql (any attempt)


GOOD_SQL = "```sql\nSELECT COUNT(*) AS n FROM customers WHERE country = 'Canada'\n```"


async def main():
    llm = RoutingLLM()
    agent = SQLAgent(
        db_url=SQLITE_URL,
        llm=llm, dialect="SQLite", max_retries=1,
        # Cache off: this was the only test writing .schema_cache.json for
        # the sqlite URL, coupling it to whatever ran before/after. Nothing
        # here measures caching (test_cache.py does that), and isolation
        # beats a few skipped tool calls.
        use_cache=False,
    )
    answer, sql, metrics = await agent.run("How many customers are from Canada?")
    print("ANSWER:", answer)
    print(metrics.summary())

    # Real assertions — this file previously had NONE, so it "passed" as
    # long as nothing raised, proving nothing about the pipeline.
    assert sql is not None and "COUNT" in sql.upper(), (
        f"expected a COUNT query to be generated, got: {sql!r}"
    )
    assert "1" in answer, (
        f"baseline fixture has exactly one Canadian customer (Alice); "
        f"answer was: {answer!r}"
    )
    assert metrics.answer_verified is True
    assert metrics.retries == 0, (
        f"first attempt should succeed against the seeded fixture; "
        f"used {metrics.retries} retries"
    )

    print("\n--- sqlite end-to-end verified: correct count, verified answer, no retries ---")


asyncio.run(main())
