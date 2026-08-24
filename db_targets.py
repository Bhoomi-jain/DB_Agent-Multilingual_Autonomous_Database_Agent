"""
db_targets.py — Single source of truth for the test suite's database DSNs.

Every test_* file used to hardcode the same connection strings (12 copies of
the Postgres URL alone), which meant pointing the suite at a different server
meant editing twelve files or sed-ing them (see SETUP_TESTS.md section 1).
These constants are env-overridable instead:

    export DB_AGENT_PG_URL="postgresql+psycopg2://user:pass@localhost/testdb"
    export DB_AGENT_MYSQL_URL="mysql+pymysql://user:pass@localhost/testdb"
    export DB_AGENT_SQLITE_URL="sqlite:////absolute/path/test_sqlite.db"

Defaults match exactly what seed_testdb.py seeds and what the tests asserted
before this module existed, so an unconfigured machine behaves identically.
"""

import os
import socket

PG_URL = os.getenv("DB_AGENT_PG_URL", "postgresql+psycopg2://postgres:postgres@localhost/testdb")
MYSQL_URL = os.getenv("DB_AGENT_MYSQL_URL", "mysql+pymysql://root:rootpass@127.0.0.1/testdb")
SQLITE_URL = os.getenv("DB_AGENT_SQLITE_URL", "sqlite:///test_sqlite.db")

# The dialect labels core_agent/seed_testdb use for these backends.
PG_DIALECT = "PostgreSQL"
MYSQL_DIALECT = "MySQL"
SQLITE_DIALECT = "SQLite"


def mysql_reachable(host: str = "127.0.0.1", port: int = 3306, timeout: float = 1.0) -> bool:
    """True if something is listening on the MySQL port. TCP reachability is
    not full authentication, but it's enough to decide whether to *attempt*
    MySQL — the failure mode of a wrong password is a clean error either way."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def pick_backend():
    """Return (db_url, dialect, label): MySQL when its server is up, else the
    same Postgres testdb the rest of the suite uses.

    Only test_cache.py needs MySQL specifically (its cache-miss arithmetic is
    backend-agnostic — identical schema shape on any engine with the three
    baseline tables) — so rather than fail that one test whenever MariaDB
    isn't running, fall back to Postgres and say so in the label."""
    if mysql_reachable():
        return MYSQL_URL, MYSQL_DIALECT, "MySQL"
    return PG_URL, PG_DIALECT, "Postgres (MySQL not reachable)"
