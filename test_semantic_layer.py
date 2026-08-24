import asyncio

import sqlglot
from sqlglot import exp

from core_agent import SQLAgent, Metrics, _resolve_table_aliases, \
    apply_measure_optimization, SemanticValidationError
from sql_semantics import classify_grains, infer_question_grain, \
    build_fk_maps, semantic_diff, summarize_diff, descendants
from sqlalchemy import create_engine, text

from db_targets import PG_URL as DB_URL


def setup():
    """Three-level fixture: customer -> invoice(header w/ stored TOTAL)
    -> line(items whose qty*amount sum EXACTLY to their header total).
    Line counts differ from invoice counts per customer, so grain errors
    are observable; exact totals make equivalence-learning probes exact."""
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        for t in ("line_demo", "invoice_demo", "customer_demo"):
            conn.execute(text(f"DROP TABLE IF EXISTS {t} CASCADE"))
        conn.execute(text("""
            CREATE TABLE customer_demo (
                customer_id SERIAL PRIMARY KEY,
                name TEXT NOT NULL
            )"""))
        conn.execute(text("""
            CREATE TABLE invoice_demo (
                invoice_id SERIAL PRIMARY KEY,
                customer_id INTEGER NOT NULL REFERENCES customer_demo(customer_id),
                total NUMERIC(10,2) NOT NULL
            )"""))
        conn.execute(text("""
            CREATE TABLE line_demo (
                line_id SERIAL PRIMARY KEY,
                invoice_id INTEGER NOT NULL REFERENCES invoice_demo(invoice_id),
                quantity INTEGER NOT NULL,
                amount NUMERIC(10,2) NOT NULL
            )"""))
        conn.execute(text("INSERT INTO customer_demo VALUES (1,'Alice'),(2,'Bob')"))
        # Alice: 3 invoices / 300; Bob: 1 invoice / 200
        conn.execute(text("""
            INSERT INTO invoice_demo VALUES
            (1,1,100),(2,1,150),(3,1,50),(4,2,200)
        """))
        # every invoice's lines sum exactly to its header total
        conn.execute(text("""
            INSERT INTO line_demo (invoice_id, quantity, amount) VALUES
            (1,2,30),(1,5,8),           -- 60+40  = 100
            (2,3,30),(2,10,6),          -- 90+60  = 150
            (3,1,50),                   -- 50     =  50
            (4,4,25),(4,10,10)          -- 100+100= 200
        """))
        conn.commit()
    engine.dispose()


def teardown():
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        for t in ("line_demo", "invoice_demo", "customer_demo"):
            conn.execute(text(f"DROP TABLE IF EXISTS {t} CASCADE"))
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


# ---------------------------------------------------------------------------
# Pure units: grain classification + question mapping
# ---------------------------------------------------------------------------

_MINI_SCHEMAS = {
    "customers": [{"name": "id", "type": "INTEGER", "primary_key": True}],
    "orders": [{"name": "order_id", "type": "INTEGER", "primary_key": True},
               {"name": "customer_id", "type": "INTEGER", "primary_key": False}],
    "order_lines": [{"name": "line_id", "type": "INTEGER", "primary_key": True},
                    {"name": "order_id", "type": "INTEGER", "primary_key": False},
                    {"name": "price", "type": "NUMERIC", "primary_key": False}],
}
_MINI_FKS = [
    {"table": "orders", "references_table": "customers"},
    {"table": "order_lines", "references_table": "orders"},
]


def test_grain_classification():
    global _MINI_GRAINS
    _MINI_GRAINS = g = classify_grains(_MINI_SCHEMAS, _MINI_FKS)
    assert g["customers"].role == "dimension", g["customers"].role
    assert g["orders"].role == "event_root", g["orders"].role
    assert g["order_lines"].role == "detail", g["order_lines"].role
    assert g["order_lines"].depth == g["orders"].depth + 1
    print("grain classification: dimension/event_root/detail correct")


def test_question_grain_mapping():
    adj, chm = build_fk_maps(_MINI_FKS)
    cases = [
        ("Which customers placed the most orders?", "customers", "orders"),
        ("How many orders are there?", None, "orders"),
        ("count the order_lines", None, "order_lines"),
        ("List customers", "customers", "customers"),
    ]
    for q, entity, want in cases:
        got = infer_question_grain(q, entity, _MINI_GRAINS, adj)
        got_t = got[0] if got else None
        assert got_t == want, f"{q!r}: expected {want}, got {got_t}"
    print(f"question->grain mapping: {len(cases)}/{len(cases)} correct")


# ---------------------------------------------------------------------------
# End-to-end: purchases counted at the WRONG grain gets repaired in-attempt
# ---------------------------------------------------------------------------

LINE_COUNT_SQL = """```sql
SELECT c.name AS name, COUNT(l.line_id) AS purchases
FROM line_demo l
JOIN invoice_demo i ON i.invoice_id = l.invoice_id
JOIN customer_demo c ON c.customer_id = i.customer_id
GROUP BY c.name ORDER BY purchases DESC LIMIT 5
```"""

DETAIL_ARITH_SQL = """```sql
SELECT SUM(l.amount * l.quantity) AS revenue
FROM line_demo l
JOIN invoice_demo d ON d.invoice_id = l.invoice_id
```"""

HEADER_TOTAL_SQL = """```sql
SELECT SUM(d.total) AS revenue FROM invoice_demo d
```"""


async def test_purchases_grain_fix():
    print("=== purchases: COUNT(line rows) rejected, retargeted to invoice PK ===")
    llm = FakeLLM([
        '{"tables": ["customer_demo", "invoice_demo", "line_demo"],'
        ' "metric": "COUNT", "entity": "customer_demo"}',
        LINE_COUNT_SQL,                                # wrong grain
        "Alice made the most purchases.",              # format_answer
    ])
    agent = SQLAgent(db_url=DB_URL, llm=llm, dialect="PostgreSQL",
                     max_retries=1, use_cache=False)
    answer, sql, metrics = await agent.run(
        "Which customers have made the most purchases?")
    print("ANSWER:", answer)
    print("FINAL SQL:", " ".join(sql.split()))
    print(metrics.summary())

    assert metrics.semantic_rejections == 1, "grain mismatch should be caught"
    assert metrics.retries == 0, (
        f"retarget is deterministic — expected 0 retries, got {metrics.retries}"
    )
    ast = sqlglot.parse_one(sql, read="postgres")
    count_args = [a.this.sql(dialect="postgres")
                  for a in ast.find_all(exp.Count)]
    assert any("invoice" in a and "line" not in a for a in count_args), (
        f"COUNT should target the invoice grain now: {count_args}"
    )
    assert not any("line" in a for a in count_args), f"line grain survived: {count_args}"
    print("\n--- purchases counted at INVOICE grain, fixed deterministically ---")


# ---------------------------------------------------------------------------
# Equivalence learning (retry pair) -> persisted -> optimizer rewrite
# ---------------------------------------------------------------------------

async def test_learn_then_optimize():
    print("=== retry pair teaches Invoice.Total == SUM(amount*quantity) ===")
    # Cache isolation: earlier runs may have already persisted this
    # equivalence; this scenario must observe the LEARNING event itself.
    import os
    cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              ".schema_cache.json")
    if os.path.exists(cache_file):
        os.remove(cache_file)

    # Run A: detail-arithmetic attempt rejected by the plan's column check,
    # header-total attempt succeeds -> paired probe learns the equivalence.
    llm = FakeLLM([
        '{"tables": ["customer_demo", "invoice_demo", "line_demo"],'
        ' "metric": "SUM", "metric_column": "total", "entity": "invoice_demo"}',
        DETAIL_ARITH_SQL,                              # rejected: wrong measure source
        HEADER_TOTAL_SQL,                              # corrected
        "The total revenue is $500.",                  # format_answer
    ])
    agent = SQLAgent(db_url=DB_URL, llm=llm, dialect="PostgreSQL",
                     max_retries=1, use_cache=True)
    answer, sql, metrics = await agent.run("What is the total revenue?")
    print(metrics.summary())
    assert metrics.semantic_rejections >= 1 and metrics.retries == 1
    eq = agent.cache.get_db_meta(DB_URL, "measure_equiv") or []
    assert any(e["parent_col"] == "total" and e["detail_table"] == "line_demo"
               for e in eq), f"equivalence should be learned+persisted: {eq}"
    assert any("learned equivalence" in n for n in metrics.optimization_notes)
    print("equivalence learned and persisted\n")

    # Run B: fresh agent, model repeats the detail-arithmetic form — the
    # optimizer must rewrite it to the learned header-total formulation.
    llm2 = FakeLLM([
        '{"tables": ["customer_demo", "invoice_demo", "line_demo"]}',
        DETAIL_ARITH_SQL,
        "The total revenue is $500.",
    ])
    agent2 = SQLAgent(db_url=DB_URL, llm=llm2, dialect="PostgreSQL",
                      max_retries=1, use_cache=True)
    answer2, sql2, metrics2 = await agent2.run("What is the total revenue?")
    print("OPTIMIZED SQL:", " ".join(sql2.split()))
    print(metrics2.summary())
    ast = sqlglot.parse_one(sql2, read="postgres")
    tables = {t.name for t in ast.find_all(exp.Table)}
    assert "line_demo" not in tables, f"detail join should be gone: {tables}"
    assert any(isinstance(a, exp.Sum) for a in ast.find_all(exp.AggFunc))
    assert any("optimizer:" in n for n in metrics2.optimization_notes), (
        f"optimizer note missing: {metrics2.optimization_notes}"
    )
    assert metrics2.retries == 0 and metrics2.semantic_rejections == 0
    print("safe-shape rewrite fired on the learned equivalence\n")


def test_optimizer_grouped_safety():
    print("=== grouped queries are NEVER auto-rewritten (safety) ===")
    grouped = ("SELECT d.customer_id, SUM(l.amount * l.quantity) "
               "FROM line_demo l JOIN invoice_demo d "
               "ON d.invoice_id = l.invoice_id GROUP BY d.customer_id")
    eq = [{"detail_table": "line_demo",
           "expr_sql": "l.amount * l.quantity",
           "parent_table": "invoice_demo", "parent_col": "total"}]
    out = apply_measure_optimization(grouped, eq, _SCHEMAS_OPT, {}, Metrics(),
                                     "postgres")
    assert out == grouped, f"grouped query must stay untouched, got: {out}"
    print("untouched as expected\n")


_SCHEMAS_OPT = {
    "invoice_demo": [{"name": "invoice_id", "type": "INT", "primary_key": True},
                     {"name": "total", "type": "NUMERIC", "primary_key": False}],
    "line_demo": [{"name": "line_id", "type": "INT", "primary_key": True},
                  {"name": "amount", "type": "NUMERIC", "primary_key": False}],
}


# ---------------------------------------------------------------------------
# Semantic diff units
# ---------------------------------------------------------------------------

def test_semantic_diff():
    d0 = semantic_diff("SELECT SUM(a) FROM t", " SELECT sum(a) FROM t ",
                       "postgres")
    assert not d0["changed"] and d0["tags"] == ["SYNTAX_ONLY"]
    d1 = semantic_diff(
        "SELECT SUM(l.qty * l.price) FROM lines l",
        "SELECT SUM(h.total) FROM head h", "postgres")
    assert "AGGREGATE_CHANGE" in d1["tags"], d1
    d2 = semantic_diff(
        "SELECT name FROM a JOIN b ON b.x = a.x",
        "SELECT name FROM a", "postgres")
    assert "JOIN_DROPPED" in d2["tags"], d2
    print(f"semantic diff: syntax-only / aggregate-change / join-drop detected "
          f"({summarize_diff(d1)})")


async def main():
    test_grain_classification()
    test_question_grain_mapping()
    test_semantic_diff()
    test_optimizer_grouped_safety()
    setup()
    try:
        await test_purchases_grain_fix()
        await test_learn_then_optimize()
    finally:
        teardown()
    print("--- semantic layer verified: grain fix + learning + safe optimizer ---")


asyncio.run(main())
