import asyncio
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


GOOD_SQL = """```sql
SELECT name, country FROM customers WHERE country = 'Canada'
```"""


async def main():
    llm = FakeLLM([
        GOOD_SQL,
        # format_answer attempt 1: hallucinates a fabricated "correction" note
        # and a wrong figure, exactly like the real observed failure.
        "Alice is from Canada (note: this seems like it could be a typo, "
        "possibly should be 15 customers, not 1).",
        # format_answer attempt 2 (after verification forces a retry):
        # correct, faithful answer.
        "Alice is the customer from Canada.",
    ])
    agent = SQLAgent(
        db_url=PG_URL,
        llm=llm, dialect="PostgreSQL", max_retries=1, use_cache=False,
    )
    answer, sql, metrics = await agent.run("Which customers are from Canada?")

    print("ANSWER:", answer)
    print(metrics.summary())

    assert "15" not in answer, "hallucinated number should not survive to final answer"
    assert metrics.verification_retries == 1
    assert metrics.answer_verified is True

    print("\n--- answer verification correctly caught the hallucinated figure "
          "and forced a corrective retry ---")


asyncio.run(main())