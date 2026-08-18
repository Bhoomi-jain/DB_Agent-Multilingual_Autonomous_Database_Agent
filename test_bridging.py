import asyncio
from core_agent import SQLAgent
from sqlalchemy import create_engine, text

DB_URL = "postgresql+psycopg2://postgres:postgres@localhost/testdb"


def setup():
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS shipments CASCADE"))
        conn.execute(text("""
            CREATE TABLE shipments (
                shipment_id SERIAL PRIMARY KEY,
                order_id INTEGER NOT NULL REFERENCES orders(order_id),
                carrier TEXT NOT NULL
            )
        """))
        conn.execute(text("INSERT INTO shipments (order_id, carrier) VALUES (1,'UPS'), (2,'FedEx'), (3,'UPS')"))
        conn.commit()
    engine.dispose()


def teardown():
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS shipments CASCADE"))
        conn.commit()
    engine.dispose()


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


GOOD_SQL = """```sql
SELECT DISTINCT c.name FROM customers c
JOIN orders o ON o.customer_id = c.customer_id
JOIN shipments s ON s.order_id = o.order_id
WHERE s.carrier = 'UPS'
```"""


async def main():
    setup()
    try:
        llm = FakeLLM([
            '["customers", "shipments"]',  # pick_relevant_tables: skips the bridge table 'orders'
            GOOD_SQL,                        # generate_sql
            "Alice and Chen had orders shipped via UPS.",  # format_answer
        ])
        agent = SQLAgent(db_url=DB_URL, llm=llm, dialect="PostgreSQL", max_retries=1, use_cache=False)
        answer, sql, metrics = await agent.run("Which customers had orders shipped via UPS?")

        print("ANSWER:", answer)
        print("SQL:", " ".join(sql.split()))
        print(metrics.summary())

        generate_sql_prompt = llm.prompts[1]
        assert "orders(" in generate_sql_prompt, (
            "'orders' bridge table schema was NOT included in the prompt — "
            "bridging failed, this is exactly the bug that caused the wrong "
            "Artist/Invoice join"
        )
        assert "FK: orders." in generate_sql_prompt or "FK: shipments." in generate_sql_prompt
        print("\nConfirmed: 'orders' bridge table schema WAS included in the "
              "generate_sql prompt, despite pick_relevant_tables never naming it.")
        print("--- FK-graph bridging verified end-to-end against live Postgres ---")
    finally:
        teardown()


asyncio.run(main())