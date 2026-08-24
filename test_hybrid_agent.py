import asyncio
from sqlalchemy import create_engine, text

from hybrid_agent import HybridAgent

from db_targets import PG_URL as DB_URL


def setup():
    """Self-contained, like test_large_schema.py / test_bridging.py: adds
    a products table + order_items.product_id FK, and MUST tear both back
    down afterward so other tests' assumptions about the base schema
    (customers/orders/order_items only) still hold."""
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS products CASCADE"))
        conn.execute(text("""
            CREATE TABLE products (
                product_id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL
            )
        """))
        conn.execute(text("""
            INSERT INTO products (name, description) VALUES
            ('Bamboo Cutting Board', 'Made from sustainably harvested bamboo, this eco-friendly cutting board is durable, lightweight, and biodegradable at end of life.'),
            ('Steel Water Bottle', 'A rugged stainless steel water bottle built to last for years, dent-resistant and fully recyclable.'),
            ('Plastic Toy Truck', 'Bright red toy truck made from cheap injection-molded plastic, not recyclable, designed for short-term play.'),
            ('Organic Cotton Tote', 'Reusable tote bag woven from organic cotton, machine washable, reduces single-use plastic bag waste.'),
            ('Gaming Mouse', 'High-precision gaming mouse with RGB lighting, 16000 DPI sensor, and programmable buttons for competitive play.'),
            ('Bluetooth Speaker', 'Portable Bluetooth speaker with deep bass, 12-hour battery life, and a rugged waterproof shell.')
        """))
        conn.execute(text("ALTER TABLE order_items ADD COLUMN IF NOT EXISTS product_id INTEGER REFERENCES products(product_id)"))
        conn.execute(text("DELETE FROM order_items"))
        conn.execute(text("""
            INSERT INTO order_items (order_id, product_name, quantity, unit_price, product_id) VALUES
            (1, 'Bamboo Cutting Board', 5, 24.99, 1),
            (1, 'Steel Water Bottle', 3, 19.99, 2),
            (2, 'Bamboo Cutting Board', 2, 24.99, 1),
            (2, 'Plastic Toy Truck', 10, 4.99, 3),
            (3, 'Organic Cotton Tote', 4, 12.99, 4),
            (3, 'Gaming Mouse', 1, 59.99, 5),
            (4, 'Plastic Toy Truck', 2, 4.99, 3),
            (5, 'Bluetooth Speaker', 2, 39.99, 6)
        """))
        conn.commit()
    engine.dispose()


def teardown():
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE order_items DROP COLUMN IF EXISTS product_id"))
        conn.execute(text("DROP TABLE IF EXISTS products CASCADE"))
        # Restore order_items to the baseline other tests expect
        conn.execute(text("DELETE FROM order_items"))
        conn.execute(text("""
            INSERT INTO order_items (order_id, product_name, quantity, unit_price) VALUES
            (1, 'Widget', 3, 9.99), (1, 'Gadget', 1, 29.99), (2, 'Widget', 5, 9.99),
            (3, 'Gizmo', 2, 49.99), (4, 'Widget', 1, 9.99)
        """))
        conn.commit()
    engine.dispose()


class FakeMsg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Scripted responses consumed in call order. Each test builds its own
    instance with exactly the responses that scenario needs."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def ainvoke(self, prompt):
        self.calls.append(prompt)
        return FakeMsg(self.responses.pop(0))


def make_agent(llm):
    return HybridAgent(
        DB_URL, llm, "PostgreSQL",
        vector_table="products", vector_text_column="description", vector_id_column="product_id",
        embedding_provider="tfidf", top_k=3, distance_threshold=1.0,
        max_retries=1, use_cache=False,
    )


async def test_sql_route():
    print("=== Route: SQL (pure structured question) ===")
    llm = FakeLLM([
        "SQL",  # classify
        '{"tables": ["products"]}',  # pick_relevant_tables (4 tables now, >3 threshold)
        "```sql\nSELECT COUNT(*) AS n FROM products\n```",  # generate_sql
        "There are 6 products.",  # format_answer
    ])
    agent = make_agent(llm)
    answer, sql, metrics, route, matches = await agent.run("How many products are there?")
    print("route:", route)
    print("answer:", answer)
    print("sql:", sql)
    assert route == "sql"
    assert matches == []
    assert sql is not None
    print("PASS\n")


async def test_semantic_route():
    print("=== Route: SEMANTIC (pure meaning-based lookup, no SQL) ===")
    llm = FakeLLM([
        "SEMANTIC",  # classify — nothing else needed, no SQL call at all
    ])
    agent = make_agent(llm)
    answer, sql, metrics, route, matches = await agent.run(
        "Which products are described as cheap and disposable?"
    )
    print("route:", route)
    print("answer:", answer)
    print("matches:", [(m["name"], round(m["distance"], 3)) for m in matches])
    assert route == "semantic"
    assert sql is None
    assert any(m["name"] == "Plastic Toy Truck" for m in matches), "should retrieve the plastic toy as the top match"
    assert matches[0]["name"] == "Plastic Toy Truck", "plastic toy should be the CLOSEST match"
    print("PASS\n")


async def test_hybrid_route():
    print("=== Route: HYBRID (semantic filter -> SQL aggregate) ===")
    llm = FakeLLM([
        "HYBRID",  # classify
        '{"tables": ["order_items"]}',  # pick_relevant_tables
        # generate_sql: model correctly uses the product_id constraint
        # injected into the augmented question
        "```sql\n"
        "SELECT SUM(oi.quantity * oi.unit_price) AS revenue\n"
        "FROM order_items oi\n"
        "WHERE oi.product_id IN (3)\n"
        "```",
        "The total revenue from cheap, disposable products is $59.88.",  # format_answer
    ])
    agent = make_agent(llm)
    answer, sql, metrics, route, matches = await agent.run(
        "What is the total revenue from products described as cheap and disposable?"
    )
    print("route:", route)
    print("answer:", answer)
    print("sql:", sql)
    print("matches used:", [m["name"] for m in matches])
    assert route == "hybrid"
    assert matches[0]["name"] == "Plastic Toy Truck"
    assert sql is not None and "product_id" in sql

    # The real check: does this match the ACTUAL revenue computed directly
    # against the live DB for exactly this product?
    from sqlalchemy import create_engine, text
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        expected = conn.execute(text(
            "SELECT SUM(quantity * unit_price) FROM order_items WHERE product_id = 3"
        )).scalar()
    engine.dispose()
    print(f"expected revenue from live DB: {expected}")
    assert str(expected) == "59.88", f"expected 59.88, DB says {expected}"
    assert "59.88" in answer
    print("PASS: hybrid route's answer matches the real, independently-verified DB value\n")


async def main():
    setup()
    try:
        await test_sql_route()
        await test_semantic_route()
        await test_hybrid_route()
        print("--- all three routing paths verified end-to-end against live Postgres ---")
    finally:
        teardown()


asyncio.run(main())