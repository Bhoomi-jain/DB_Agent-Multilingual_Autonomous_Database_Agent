import asyncio
from core_agent import SQLAgent
from sqlalchemy import create_engine, text

DB_URL = "postgresql+psycopg2://postgres:postgres@localhost/testdb"


def setup_isolated_table():
    """A table with NO foreign key to anything — used to test the case
    where repair_join_path genuinely cannot help (no FK path exists at
    all), so the normal LLM-retry fallback must be exercised instead."""
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS site_settings CASCADE"))
        conn.execute(text("CREATE TABLE site_settings (setting_id SERIAL PRIMARY KEY, setting_value TEXT)"))
        conn.execute(text("INSERT INTO site_settings (setting_value) VALUES ('dark_mode')"))
        conn.commit()
    engine.dispose()


def teardown_isolated_table():
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS site_settings CASCADE"))
        conn.commit()
    engine.dispose()


class FakeMsg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    async def ainvoke(self, prompt):
        return FakeMsg(self.responses.pop(0))


# Attempt 1: hallucinated join to a table that has NO FK relationship to
# anything at all — no path exists for repair_join_path to find, so this
# MUST fall back to a normal LLM retry.
BAD_JOIN_SQL = """```sql
SELECT c.name, s.setting_value
FROM customers c
JOIN site_settings s ON c.customer_id = s.setting_id
```"""

# Attempt 2: the model gives up on the impossible join and answers using
# only the real, connected table.
GOOD_SQL = """```sql
SELECT name FROM customers
```"""


async def main():
    setup_isolated_table()
    try:
        llm = FakeLLM([
            '["customers", "site_settings"]',  # pick_relevant_tables (now >3 tables with site_settings added)
            BAD_JOIN_SQL,
            GOOD_SQL,
            "Here are the customer names.",  # format_answer
        ])
        agent = SQLAgent(db_url=DB_URL, llm=llm, dialect="PostgreSQL", max_retries=2, use_cache=False)
        answer, sql, metrics = await agent.run("Which customers use dark mode?")

        print("ANSWER:", answer)
        print("FINAL SQL:", " ".join(sql.split()))
        print(metrics.summary())

        assert "site_settings" not in sql
        assert metrics.semantic_rejections == 1, f"expected 1 semantic rejection, got {metrics.semantic_rejections}"
        assert metrics.retries == 1, (
            f"expected 1 LLM retry — no FK path exists to site_settings, so "
            f"deterministic repair genuinely cannot help here, got {metrics.retries}"
        )

        print("\n--- confirmed: when no FK path exists at all, repair correctly "
              "declines to guess and falls back to a normal LLM retry ---")
    finally:
        teardown_isolated_table()


asyncio.run(main())