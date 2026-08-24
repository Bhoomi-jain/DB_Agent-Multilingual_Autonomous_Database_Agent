#!/usr/bin/env python3
"""
Seed the baseline test database that the db-agent test suite assumes exists.

The suite was originally developed against a dev sandbox where `testdb` was
seeded by hand (PROJECT_HANDOFF.md section 7), so no seed script was ever
committed. On a fresh machine every test fails before reaching any agent
logic. This recreates that baseline exactly.

What it creates - and why each detail matters:

  customers / orders / order_items, and NOTHING else.
      Exactly three tables is load-bearing. `pick_relevant_tables()` skips
      its LLM call when a DB has <= 3 tables, and test_retry,
      test_exhausted_retries and test_cache all depend on that skip
      (test_cache asserts exactly 5 cache misses = 1 list_tables + 3
      describe_table + 1 list_foreign_keys). A stray 4th table silently
      breaks ~5 tests - the exact test-pollution bug in section 6.17.

  Real declared FOREIGN KEYs.
      orders.customer_id -> customers.customer_id
      order_items.order_id -> orders.order_id
      `list_foreign_keys` drives _bridge_tables(), validate_join_semantics(),
      repair_join_path() and repair_missing_joins(). Without declared FKs
      those passes have an empty graph and the repair tests cannot pass.
      MySQL tables are forced to InnoDB for the same reason - MyISAM parses
      FK syntax and then ignores it.

  unit_price as NUMERIC/DECIMAL, not FLOAT.
      Postgres NUMERIC arrives through the MCP JSON layer as a *string*
      ("59.88"). That is the bug in section 6.16 and what
      test_hybrid_agent's `assert str(expected) == "59.88"` pins down.
      A float column would quietly bypass that code path.

  Five orders, though the baseline only has items for four.
      test_hybrid_agent inserts order_items rows referencing order_id 5;
      with only four orders that INSERT dies on the FK constraint.

  item_id as the order_items primary key.
      test_alias_repair uses COUNT(OI.item_id) and
      test_missing_join_repair uses COUNT(order_items.item_id).

Data is chosen so the assertions in the suite hold: Alice is the only
Canadian customer (test_sqlite_repro expects COUNT = 1,
test_answer_verification expects a single row), and Alice leads both on
completed revenue (109.91 vs Bob 99.98 vs Carol 9.99) and on item count
(9 vs 2 vs 1), which is what test_cache and test_retry assert via
`"Alice" in answer`.

Usage
-----
    python seed_testdb.py --target postgres
    python seed_testdb.py --target all
    python seed_testdb.py --target postgres --clean-only

Connection strings come from db_targets.py and can be overridden without
editing any file (the tests read the same module, so one export retargets
seeder and suite together):

    DB_AGENT_PG_URL     postgresql+psycopg2://postgres:postgres@localhost/testdb
    DB_AGENT_MYSQL_URL  mysql+pymysql://root:rootpass@127.0.0.1/testdb
    DB_AGENT_SQLITE_URL sqlite:///test_sqlite.db
"""

import argparse
import sys

from sqlalchemy import create_engine, event, inspect, text

from db_targets import PG_URL, MYSQL_URL, SQLITE_URL

BASE_TABLES = ["order_items", "orders", "customers"]  # child-first drop order

# Tables created by self-contained tests. A test that crashes between its
# setup() and teardown() leaves one behind, which pushes the DB past the
# 3-table threshold and breaks unrelated tests. Always swept before seeding.
EPHEMERAL_TABLES = [
    "shipments",        # test_bridging
    "site_settings",    # test_semantic_validation
    "product_sales",    # test_tie_aware_ranking
    "products",         # test_hybrid_agent
] + [f"artist_extra_{i}" for i in range(1, 8)]  # test_large_schema

DDL = {
    "postgres": [
        """CREATE TABLE customers (
               customer_id SERIAL PRIMARY KEY,
               name        TEXT NOT NULL,
               country     TEXT NOT NULL)""",
        """CREATE TABLE orders (
               order_id    SERIAL PRIMARY KEY,
               customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
               status      TEXT NOT NULL)""",
        """CREATE TABLE order_items (
               item_id      SERIAL PRIMARY KEY,
               order_id     INTEGER NOT NULL REFERENCES orders(order_id),
               product_name TEXT NOT NULL,
               quantity     INTEGER NOT NULL,
               unit_price   NUMERIC(10,2) NOT NULL)""",
    ],
    "mysql": [
        """CREATE TABLE customers (
               customer_id INT AUTO_INCREMENT PRIMARY KEY,
               name        VARCHAR(255) NOT NULL,
               country     VARCHAR(255) NOT NULL) ENGINE=InnoDB""",
        """CREATE TABLE orders (
               order_id    INT AUTO_INCREMENT PRIMARY KEY,
               customer_id INT NOT NULL,
               status      VARCHAR(64) NOT NULL,
               FOREIGN KEY (customer_id) REFERENCES customers(customer_id)) ENGINE=InnoDB""",
        """CREATE TABLE order_items (
               item_id      INT AUTO_INCREMENT PRIMARY KEY,
               order_id     INT NOT NULL,
               product_name VARCHAR(255) NOT NULL,
               quantity     INT NOT NULL,
               unit_price   DECIMAL(10,2) NOT NULL,
               FOREIGN KEY (order_id) REFERENCES orders(order_id)) ENGINE=InnoDB""",
    ],
    "sqlite": [
        """CREATE TABLE customers (
               customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
               name        TEXT NOT NULL,
               country     TEXT NOT NULL)""",
        """CREATE TABLE orders (
               order_id    INTEGER PRIMARY KEY AUTOINCREMENT,
               customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
               status      TEXT NOT NULL)""",
        """CREATE TABLE order_items (
               item_id      INTEGER PRIMARY KEY AUTOINCREMENT,
               order_id     INTEGER NOT NULL REFERENCES orders(order_id),
               product_name TEXT NOT NULL,
               quantity     INTEGER NOT NULL,
               unit_price   NUMERIC(10,2) NOT NULL)""",
    ],
}

CUSTOMERS = [(1, "Alice", "Canada"), (2, "Bob", "USA"), (3, "Carol", "UK")]
ORDERS = [(1, 1, "completed"), (2, 1, "completed"), (3, 2, "completed"),
          (4, 3, "completed"), (5, 2, "completed")]
ORDER_ITEMS = [(1, "Widget", 3, 9.99), (1, "Gadget", 1, 29.99),
               (2, "Widget", 5, 9.99), (3, "Gizmo", 2, 49.99),
               (4, "Widget", 1, 9.99)]


def make_engine(kind, url):
    engine = create_engine(url, pool_pre_ping=True)
    if kind == "sqlite":
        # SQLAlchemy does not enable SQLite FK enforcement by default, and the
        # repair passes need the FKs to be visible and honoured.
        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn, _record):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
    return engine


def drop_all(conn, kind):
    cascade = " CASCADE" if kind == "postgres" else ""
    if kind == "mysql":
        conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
    for tbl in EPHEMERAL_TABLES + BASE_TABLES:
        conn.execute(text(f"DROP TABLE IF EXISTS {tbl}{cascade}"))
    if kind == "mysql":
        conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))


def seed(conn, kind):
    for stmt in DDL[kind]:
        conn.execute(text(stmt))

    conn.execute(
        text("INSERT INTO customers (customer_id, name, country) VALUES (:i, :n, :c)"),
        [{"i": i, "n": n, "c": c} for i, n, c in CUSTOMERS],
    )
    conn.execute(
        text("INSERT INTO orders (order_id, customer_id, status) VALUES (:o, :c, :s)"),
        [{"o": o, "c": c, "s": s} for o, c, s in ORDERS],
    )
    conn.execute(
        text("INSERT INTO order_items (order_id, product_name, quantity, unit_price) "
             "VALUES (:o, :p, :q, :u)"),
        [{"o": o, "p": p, "q": q, "u": u} for o, p, q, u in ORDER_ITEMS],
    )

    if kind == "postgres":
        # Explicit PKs leave SERIAL sequences at 1; later inserts would collide.
        for tbl, col in (("customers", "customer_id"), ("orders", "order_id")):
            conn.execute(text(
                f"SELECT setval(pg_get_serial_sequence('{tbl}', '{col}'), "
                f"(SELECT MAX({col}) FROM {tbl}))"
            ))


def verify(engine, kind):
    """Assert the seeded DB satisfies what the suite's assertions require."""
    problems = []
    with engine.connect() as conn:
        tables = sorted(inspect(engine).get_table_names())
        if tables != ["customers", "order_items", "orders"]:
            problems.append(
                f"expected exactly 3 base tables, found {len(tables)}: {tables} "
                "(a 4th table breaks pick_relevant_tables' <=3 skip)"
            )

        n_canada = conn.execute(text(
            "SELECT COUNT(*) FROM customers WHERE country = 'Canada'")).scalar()
        if n_canada != 1:
            problems.append(f"expected exactly 1 Canadian customer, got {n_canada}")

        top = conn.execute(text("""
            SELECT c.name, SUM(oi.quantity * oi.unit_price) AS revenue
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            JOIN order_items oi ON oi.order_id = o.order_id
            WHERE o.status = 'completed'
            GROUP BY c.name ORDER BY revenue DESC
        """)).fetchall()
        if not top or top[0][0] != "Alice":
            problems.append(f"expected Alice to lead on revenue, got {top}")

        items = conn.execute(text("""
            SELECT c.name, SUM(oi.quantity) AS n
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            JOIN order_items oi ON oi.order_id = o.order_id
            GROUP BY c.name ORDER BY n DESC
        """)).fetchall()
        if not items or items[0][0] != "Alice":
            problems.append(f"expected Alice to lead on item count, got {items}")

        n_orders = conn.execute(text("SELECT COUNT(*) FROM orders")).scalar()
        if n_orders < 5:
            problems.append(
                f"expected >=5 orders (test_hybrid_agent references order_id 5), got {n_orders}")

        insp = inspect(engine)
        found = set()
        for tbl in ("orders", "order_items"):
            for fk in insp.get_foreign_keys(tbl):
                found.add((tbl, fk["referred_table"]))
        for want in (("orders", "customers"), ("order_items", "orders")):
            if want not in found:
                problems.append(
                    f"missing declared FK {want[0]} -> {want[1]}; the repair passes "
                    "build their join graph from these"
                )

        revenue = {name: str(rev) for name, rev in top}
    return problems, tables, revenue


def do_target(kind, url, clean_only=False):
    label = kind.upper()
    print(f"\n=== {label}  ({url.split('@')[-1] if '@' in url else url})")
    try:
        engine = make_engine(kind, url)
        with engine.connect() as conn:
            drop_all(conn, kind)
            if not clean_only:
                seed(conn, kind)
            conn.commit()
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {str(exc).splitlines()[0]}")
        if "password authentication failed" in str(exc):
            print("  hint: the DSN the tests hardcode expects user 'postgres' with")
            print("        password 'postgres'. Either set that password, or export")
            print("        DB_AGENT_PG_URL and use --target postgres again.")
        if "Can't connect to MySQL" in str(exc) or "Connection refused" in str(exc):
            print("  hint: the server does not appear to be running.")
        if 'database "testdb" does not exist' in str(exc) or "Unknown database" in str(exc):
            print("  hint: create it first, e.g.  createdb testdb")
        return False
    finally:
        try:
            engine.dispose()
        except Exception:
            pass

    if clean_only:
        print("  cleaned (base + ephemeral test tables dropped)")
        return True

    engine = make_engine(kind, url)
    try:
        problems, tables, revenue = verify(engine, kind)
    finally:
        engine.dispose()

    print(f"  tables : {tables}")
    print(f"  revenue: {revenue}")
    if problems:
        print("  SELF-CHECK FAILED:")
        for p in problems:
            print(f"    - {p}")
        return False
    print("  self-check passed")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["postgres", "mysql", "sqlite", "all"],
                    default="all")
    ap.add_argument("--clean-only", action="store_true",
                    help="drop base + leftover test tables, do not re-seed")
    args = ap.parse_args()

    targets = [("postgres", PG_URL), ("mysql", MYSQL_URL), ("sqlite", SQLITE_URL)]
    if args.target != "all":
        targets = [t for t in targets if t[0] == args.target]

    results = {kind: do_target(kind, url, args.clean_only) for kind, url in targets}

    print("\n" + "=" * 60)
    for kind, ok in results.items():
        print(f"  {kind:10} {'OK' if ok else 'FAILED'}")
    ok_count = sum(results.values())
    print(f"  {ok_count}/{len(results)} target(s) ready")

    # SQLite alone is enough for test_sqlite_repro; Postgres is needed by 12
    # tests and MySQL only by test_cache.
    sys.exit(0 if ok_count == len(results) else 1)


if __name__ == "__main__":
    main()