import asyncio
from core_agent import SQLAgent
from sqlalchemy import create_engine, text

DB_URL = "postgresql+psycopg2://postgres:postgres@localhost/testdb"


def setup_tie_scenario():
    """Add a product with sales that TIE with an existing product's total,
    right at a rank boundary, to test genuine tie-handling."""
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS product_sales CASCADE"))
        conn.execute(text("CREATE TABLE product_sales (product TEXT, revenue INTEGER)"))
        conn.execute(text("""
            INSERT INTO product_sales (product, revenue) VALUES
            ('Widget', 500), ('Gadget', 400), ('Gizmo', 300),
            ('Doohickey', 200), ('Thingamajig', 100), ('Contraption', 100)
        """))  # Thingamajig and Contraption tie for last place at 100
        conn.commit()
    engine.dispose()


def teardown_tie_scenario():
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS product_sales CASCADE"))
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
        return FakeMsg(self.responses.pop(0))


TOP5_SQL = """```sql
SELECT product, revenue FROM product_sales ORDER BY revenue DESC LIMIT 5
```"""


async def main():
    setup_tie_scenario()
    try:
        llm = FakeLLM([
            '["product_sales"]',  # pick_relevant_tables (4 tables total now, >3 threshold)
            TOP5_SQL,
            "Here are the top products by revenue.",
        ])
        agent = SQLAgent(db_url=DB_URL, llm=llm, dialect="PostgreSQL", max_retries=1, use_cache=False)
        answer, sql, metrics = await agent.run("What are the top 5 products by revenue?")

        print("ANSWER:", answer)
        print("FINAL SQL:", " ".join(sql.split()))
        print(metrics.summary())

        assert "RANK()" in sql or "rnk" in sql.lower(), "should have been rewritten to be tie-aware"

        # Execute the final SQL directly to confirm it actually returns
        # BOTH tied rows (6 total), not an arbitrary 5.
        from sqlalchemy import create_engine as _ce, text as _t
        eng = _ce(DB_URL)
        with eng.connect() as conn:
            rows = conn.execute(_t(sql)).fetchall()
        eng.dispose()
        print("ROWS RETURNED:", rows)
        assert len(rows) == 6, f"expected 6 rows (both tied products included), got {len(rows)}"
        products = {r[0] for r in rows}
        assert {"Thingamajig", "Contraption"}.issubset(products), (
            "both tied products should be present, not just one"
        )

        print("\n--- confirmed: actual row count and content include both tied rows ---")
    finally:
        teardown_tie_scenario()


asyncio.run(main())