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


async def main():
    llm = FakeLLM([
        "```sql\nSELECT COUNT(*) AS n FROM customers WHERE country = 'Canada'\n```",
        "There is 1 customer from Canada.",
    ])
    agent = SQLAgent(
        db_url="sqlite:///test_sqlite.db",
        llm=llm, dialect="SQL", max_retries=1, use_cache=True,
    )
    answer, sql, metrics = await agent.run("How many customers are from Canada?")
    print("ANSWER:", answer)
    print(metrics.summary())


asyncio.run(main())
