import asyncio

import sqlglot
from sqlalchemy import create_engine, text

from core_agent import SQLAgent, validate_aggregation_fanout, SemanticValidationError
from db_targets import PG_URL as DB_URL


def setup():
    """Self-contained two-table fact/line-item fixture: the exact shape that
    produced the live Chinook failure (SUM over a fact table joined to its
    line items ~9x-inflated revenue)."""
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS line_demo CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS invoice_demo CASCADE"))
        conn.execute(text("""
            CREATE TABLE invoice_demo (
                invoice_id SERIAL PRIMARY KEY,
                total NUMERIC(10,2) NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE line_demo (
                line_id SERIAL PRIMARY KEY,
                invoice_id INTEGER NOT NULL REFERENCES invoice_demo(invoice_id),
                amount NUMERIC(10,2) NOT NULL
            )
        """))
        conn.execute(text("""
            INSERT INTO invoice_demo (invoice_id, total) VALUES (1, 100), (2, 200)
        """))
        # 3 lines on invoice 1, 2 lines on invoice 2 -> naive join-sum = 300 * 5 = 1500
        conn.execute(text("""
            INSERT INTO line_demo (invoice_id, amount) VALUES
            (1, 40), (1, 60), (2, 90), (2, 110)
        """))  # line amounts SUM EXACTLY to their header totals (100/200) —
        # lets measure-equivalence probes learn Invoice.Total == SUM(amount)
        # without any tolerance fudging; also used by semantic-layer tests.
        conn.commit()
    engine.dispose()


def teardown():
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS line_demo CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS invoice_demo CASCADE"))
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


FANOUT_SQL = """```sql
SELECT SUM(d.total) FROM invoice_demo d
JOIN line_demo l ON l.invoice_id = d.invoice_id
```"""

GOOD_SQL = """```sql
SELECT SUM(total) AS revenue FROM invoice_demo
"""

# FK/schema mirror of the fixture — used by direct validator/repair unit checks.
fk = [
    {"table": "line_demo", "columns": ["invoice_id"],
     "references_table": "invoice_demo", "references_columns": ["invoice_id"]},
]
schemas = {
    "invoice_demo": [{"name": "invoice_id", "type": "INT", "primary_key": True},
                     {"name": "total", "type": "NUMERIC", "primary_key": False}],
    "line_demo": [{"name": "line_id", "type": "INT", "primary_key": True},
                  {"name": "invoice_id", "type": "INT", "primary_key": False},
                  {"name": "amount", "type": "NUMERIC", "primary_key": False}],
}


async def main():
    setup()
    try:
        llm = FakeLLM([
            '{"tables": ["invoice_demo"]}',   # pick_relevant_tables (4 tables > 3 threshold)
            FANOUT_SQL,           # generate_sql attempt 1: the fan-out join
            "The total revenue is $300.",  # format_answer (NO retry needed:
            # repair_fanout_join drops the offending join deterministically)
        ])
        agent = SQLAgent(db_url=DB_URL, llm=llm, dialect="PostgreSQL",
                         max_retries=1, use_cache=False)
        answer, sql, metrics = await agent.run("What is the total revenue?")

        print("ANSWER:", answer)
        print("FINAL SQL:", " ".join(sql.split()))
        print(metrics.summary())

        assert metrics.semantic_rejections == 1, (
            f"expected exactly 1 semantic rejection (the fan-out), "
            f"got {metrics.semantic_rejections}"
        )
        # House philosophy in action: deterministic fix beats hoping the
        # model self-corrects — zero LLM retries for a provably-safe repair.
        assert metrics.retries == 0, (
            f"expected 0 retries (deterministic join-drop), got {metrics.retries}"
        )
        # Final query must be the repaired one: single table, no line_demo join.
        ast = sqlglot.parse_one(sql, read="postgres")
        joined = {t.name for t in ast.find_all(sqlglot.exp.Table)}
        assert "line_demo" not in joined, f"fan-out join should be gone, tables={joined}"
        assert metrics.answer_verified is True

        print("\n--- fan-out join detected AND dropped deterministically within the same attempt ---")

        # Unit check: repair leaves load-bearing joins untouched.
        from core_agent import repair_fanout_join
        load_bearing = ("SELECT d.total, l.amount FROM invoice_demo d "
                        "JOIN line_demo l ON l.invoice_id = d.invoice_id")
        assert repair_fanout_join(load_bearing, fk, schemas, "postgres") == load_bearing, (
            "repair must NOT drop a join whose child columns are used in SELECT"
        )
        print("unit control passed: load-bearing child join left untouched")

        # ---- Negative controls: legitimate shapes MUST NOT be flagged ----
        legit = [
            ("child-side measure", FANOUT_SQL.replace("d.total", "l.amount")),
            ("grouped by parent PK",
             "SELECT d.invoice_id, SUM(d.total) FROM invoice_demo d "
             "JOIN line_demo l ON l.invoice_id = d.invoice_id GROUP BY d.invoice_id"),
            ("no child join at all", GOOD_SQL),
            # The canonical per-group row-count pattern: grouped COUNT over a
            # parent->child chain. This EXACT shape regressed once (the
            # detector flagged it after join-path repair inserted the bridge),
            # breaking test_join_path_repair — locked here forever.
            ("grouped COUNT counts children per group",
             "SELECT customers.name, COUNT(*) AS item_count FROM customers "
             "JOIN orders ON customers.customer_id = orders.customer_id "
             "JOIN order_items ON order_items.order_id = orders.order_id "
             "GROUP BY customers.name"),
        ]
        for label, q in legit:
            try:
                validate_aggregation_fanout(q, fk, schemas, "postgres")
                print(f"negative control passed (not flagged): {label}")
            except SemanticValidationError as e:
                raise AssertionError(f"FALSE POSITIVE on '{label}': {e}")
    finally:
        teardown()


asyncio.run(main())
