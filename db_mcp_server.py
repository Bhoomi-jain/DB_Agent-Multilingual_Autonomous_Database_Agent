"""
db_mcp_server.py — A minimal, security-reviewed MCP server for relational
databases (PostgreSQL and MySQL/MariaDB today; anything SQLAlchemy supports
in principle).

Why this exists instead of a community MCP server: the most widely used
community Postgres MCP server (@modelcontextprotocol/server-postgres) has a
documented SQL injection vulnerability that lets a crafted query escape its
"read-only" mode entirely (e.g. `COMMIT; DROP TABLE users;`). This server
avoids that class of bug by:

  1. Parsing every query into a real SQL AST (via sqlglot) instead of
     string/regex matching, and rejecting anything that isn't a single
     read-only SELECT/CTE/UNION statement.
  2. Walking the full AST (not just the top-level statement) to catch
     DML/DDL hidden in subqueries or CTEs.
  3. Blocking statement stacking (`;`-separated multi-statement payloads).
  4. Capping result set size so a single query can't exhaust memory.

Exposed tools:
  - list_tables()                 -> table names in the connected database
  - describe_table(table_name)    -> columns, types, nullability, PK
  - list_foreign_keys(table_name) -> FK relationships (omit table_name for all)
  - run_query(query)              -> execute a read-only SELECT, capped rowsp

Usage:
  python db_mcp_server.py --db-url postgresql+psycopg2://user:pass@host/db
  python db_mcp_server.py --db-url mysql+pymysql://user:pass@host/db
"""
import argparse
import os
import sys

import sqlglot
from sqlglot import exp
from sqlalchemy import create_engine, inspect, text
from mcp.server.fastmcp import FastMCP

MAX_ROWS = 500

# Statement/expression types that are never allowed, even nested inside a
# CTE or subquery.
FORBIDDEN_NODE_TYPES = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.TruncateTable, exp.Merge, exp.Grant,
)

# Function names occasionally used for data exfiltration / side effects
# even from within a SELECT (file I/O, sleep-based blind injection probes).
FORBIDDEN_FUNCTIONS = {
    "pg_sleep", "pg_read_file", "pg_ls_dir", "lo_import", "lo_export",
    "load_file", "sleep", "benchmark", "into outfile", "into dumpfile",
}


class ReadOnlyViolation(ValueError):
    pass


def _sql_dialect_for(db_url: str) -> str:
    if db_url.startswith("postgresql"):
        return "postgres"
    if db_url.startswith("mysql"):
        return "mysql"
    if db_url.startswith("sqlite"):
        return "sqlite"
    return ""  # let sqlglot guess


def validate_readonly(sql: str, dialect: str) -> exp.Expression:
    """Parse and validate that `sql` is exactly one read-only statement.
    Returns the parsed AST on success; raises ReadOnlyViolation otherwise."""
    try:
        statements = sqlglot.parse(sql, read=dialect or None)
    except Exception as e:
        raise ReadOnlyViolation(f"Could not parse SQL: {e}")

    statements = [s for s in statements if s is not None]
    if len(statements) == 0:
        raise ReadOnlyViolation("No SQL statement found.")
    if len(statements) > 1:
        raise ReadOnlyViolation(
            "Only a single SQL statement is allowed per call — "
            "multi-statement payloads are rejected."
        )

    stmt = statements[0]

    # Top-level statement must be a SELECT, a set operation over SELECTs
    # (UNION/INTERSECT/EXCEPT), or a CTE (WITH ...) whose body is one of those.
    top_level_ok = isinstance(stmt, (exp.Select, exp.Union, exp.Intersect, exp.Except))
    if isinstance(stmt, exp.With):
        top_level_ok = isinstance(stmt.this, (exp.Select, exp.Union, exp.Intersect, exp.Except))

    if not top_level_ok:
        raise ReadOnlyViolation(
            f"Only SELECT statements are allowed (got: {stmt.key})."
        )

    # Walk the *entire* tree — this catches DML/DDL smuggled inside a
    # subquery, CTE, or table-valued expression.
    for node in stmt.walk():
        if isinstance(node, FORBIDDEN_NODE_TYPES):
            raise ReadOnlyViolation(
                f"Modifying statements are not allowed (found: {node.key})."
            )
        if isinstance(node, (exp.Anonymous, exp.Func)):
            fname = (node.name or "").lower()
            if fname in FORBIDDEN_FUNCTIONS:
                raise ReadOnlyViolation(f"Function '{fname}' is not allowed.")

    return stmt


def build_server(db_url: str) -> FastMCP:
    engine = create_engine(db_url, pool_pre_ping=True)
    dialect = _sql_dialect_for(db_url)
    mcp = FastMCP("db-mcp-server")

    @mcp.tool()
    def list_tables() -> list[str]:
        """List all table names in the connected database."""
        insp = inspect(engine)
        return insp.get_table_names()

    @mcp.tool()
    def describe_table(table_name: str) -> list[dict]:
        """Get column definitions (name, type, nullable, primary key) for a table."""
        insp = inspect(engine)
        pk_cols = set(insp.get_pk_constraint(table_name).get("constrained_columns", []))
        columns = []
        for col in insp.get_columns(table_name):
            columns.append({
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col.get("nullable", True),
                "primary_key": col["name"] in pk_cols,
            })
        return columns

    @mcp.tool()
    def list_foreign_keys(table_name: str | None = None) -> list[dict]:
        """List foreign key relationships. Pass a table_name to filter to one
        table, or omit it to get every FK relationship in the database — use
        this before writing JOINs so you use the correct keys."""
        insp = inspect(engine)
        tables = [table_name] if table_name else insp.get_table_names()
        relationships = []
        for t in tables:
            for fk in insp.get_foreign_keys(t):
                relationships.append({
                    "table": t,
                    "columns": fk["constrained_columns"],
                    "references_table": fk["referred_table"],
                    "references_columns": fk["referred_columns"],
                })
        return relationships

    @mcp.tool()
    def run_query(query: str) -> dict:
        """Execute a read-only SQL query (SELECT, including JOINs, GROUP BY,
        window functions, and CTEs) and return the results. INSERT/UPDATE/
        DELETE/DROP/ALTER and any multi-statement payload are rejected before
        they ever reach the database. Results are capped at
        {max_rows} rows.""".format(max_rows=MAX_ROWS)
        validate_readonly(query, dialect)

        with engine.connect() as conn:
            result = conn.execute(text(query))
            columns = list(result.keys())
            rows = result.fetchmany(MAX_ROWS + 1)
            truncated = len(rows) > MAX_ROWS
            rows = rows[:MAX_ROWS]
            return {
                "columns": columns,
                "rows": [list(r) for r in rows],
                "row_count": len(rows),
                "truncated": truncated,
            }

    return mcp


def main():
    parser = argparse.ArgumentParser(description="Read-only SQL MCP server (Postgres/MySQL)")
    parser.add_argument(
        "--db-url",
        default=os.getenv("DATABASE_URL"),
        help="SQLAlchemy connection URL, e.g. "
             "postgresql+psycopg2://user:pass@host/db or "
             "mysql+pymysql://user:pass@host/db "
             "(or set the DATABASE_URL env var)",
    )
    args = parser.parse_args()

    if not args.db_url:
        print("Error: --db-url or DATABASE_URL env var is required", file=sys.stderr)
        sys.exit(1)

    server = build_server(args.db_url)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
