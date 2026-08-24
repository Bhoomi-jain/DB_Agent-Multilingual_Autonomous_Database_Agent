#!/usr/bin/env python3
"""
Run every test_*.py in this directory and write one pasteable report.

Drop this file in your project root (next to core_agent.py) and run:

    uv run python run_tests.py           # or: python run_tests.py

It writes test_report.txt next to itself. Paste/attach that file back.

Stdlib only - no new dependencies. Each test runs in its own subprocess so
one crash can't take the rest down.
"""

import argparse
import datetime
import glob
import os
import socket
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "test_report.txt")
CACHE_FILE = os.path.join(HERE, ".schema_cache.json")

# Tests that mutate shared DB state run last; the self-contained one runs first.
PREFERRED_FIRST = ["test_sqlite_repro.py"]

KEY_PACKAGES = [
    "sqlglot", "sqlalchemy", "langchain", "langgraph", "langchain_core",
    "langchain_mcp_adapters", "mcp", "chromadb", "sklearn", "rich",
    "langdetect", "psycopg2", "pymysql",
]

# NOTE: deliberately no probe for ChromaDB. vector_store.py uses chromadb's
# embedded EphemeralClient/PersistentClient (in-process), so there is no
# server listening on port 8000 — probing it always reported a misleading
# "NOT reachable" that looked like a missing service but was never one.
PROBES = [
    ("PostgreSQL", "localhost", 5432),
    ("MySQL", "127.0.0.1", 3306),
    ("Ollama", "127.0.0.1", 11434),
]


def tcp_probe(host, port, timeout=1.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return "reachable"
    except Exception as exc:  # noqa: BLE001 - report whatever went wrong
        return f"NOT reachable ({type(exc).__name__})"


def preflight():
    out = []
    out.append(f"timestamp      : {datetime.datetime.now().isoformat(timespec='seconds')}")
    out.append(f"python         : {sys.version.split()[0]} ({sys.executable})")
    out.append(f"platform       : {sys.platform}")
    out.append(f"cwd            : {HERE}")
    out.append("")
    out.append("package versions:")
    for name in KEY_PACKAGES:
        try:
            mod = __import__(name)
            out.append(f"  {name:24} {getattr(mod, '__version__', 'installed (no __version__)')}")
        except Exception as exc:  # noqa: BLE001
            out.append(f"  {name:24} MISSING ({type(exc).__name__})")
    out.append("")
    out.append("service reachability:")
    for label, host, port in PROBES:
        out.append(f"  {label:12} {host}:{port:<6} {tcp_probe(host, port)}")
    return "\n".join(out)


def discover():
    found = sorted(os.path.basename(p) for p in glob.glob(os.path.join(HERE, "test_*.py")))
    first = [t for t in PREFERRED_FIRST if t in found]
    return first + [t for t in found if t not in first]


def classify(proc):
    if proc.returncode == 0:
        return "PASS"
    blob = (proc.stderr or "") + (proc.stdout or "")
    if "AssertionError" in blob:
        return "FAIL(assert)"
    if "Traceback" in blob:
        return "ERROR"
    return f"ERROR(rc={proc.returncode})"


def tail(text, limit):
    if not text:
        return "(empty)"
    lines = text.splitlines()
    if len(lines) <= limit:
        return "\n".join(lines)
    hidden = len(lines) - limit
    return f"... [{hidden} earlier lines omitted] ...\n" + "\n".join(lines[-limit:])


def clear_cache():
    """Remove .schema_cache.json before each test.

    PROJECT_HANDOFF.md section 7 requires clearing this between
    cache-sensitive tests, and CACHE_TTL_SECONDS is 300: re-running the
    suite within five minutes would otherwise let test_cache and test_retry
    see cache HITS where they assert exactly 5 misses, producing a failure
    that looks like a caching bug but is really stale fixture state.
    """
    removed = []
    for path in (CACHE_FILE, CACHE_FILE + ".tmp"):
        try:
            os.remove(path)
            removed.append(os.path.basename(path))
        except FileNotFoundError:
            pass
        except OSError as exc:
            removed.append(f"{os.path.basename(path)} (could not remove: {exc})")
    return removed


HINTS = (
    ("password authentication failed",
     "Postgres rejected the hardcoded DSN. Set the postgres password to "
     "'postgres', or export DB_AGENT_PG_URL and re-seed."),
    # Deliberately NOT the substring "does not exist": retry-loop tests
    # intentionally provoke real DB errors whose text contains it
    # (e.g. "column c.no_such_column does not exist"), which made every
    # fully-passing run look like it had a missing-database problem.
    ("UndefinedTable",
     "The database or a table is missing - run: python seed_testdb.py --target all"),
    ("no such table",
     "SQLite fixture not seeded - run: python seed_testdb.py --target sqlite"),
    ("Can't connect to MySQL",
     "MySQL/MariaDB is not running (only test_cache.py needs it)."),
    ("Unknown database",
     "The 'testdb' database does not exist yet - create it, then seed."),
)


def hints_for(blob):
    return [msg for needle, msg in HINTS if needle in blob]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=240, help="per-test timeout, seconds")
    ap.add_argument("--tail", type=int, default=60, help="output lines kept per test")
    ap.add_argument("--only", nargs="*", help="run only these test files")
    ap.add_argument("--keep-cache", action="store_true",
                    help="do NOT clear .schema_cache.json between tests "
                         "(the default clears it; see clear_cache)")
    args = ap.parse_args()

    tests = args.only or discover()
    chunks = ["=" * 72, "ENVIRONMENT", "=" * 72, preflight(), ""]
    results = []
    all_output = []

    for name in tests:
        cleared = [] if args.keep_cache else clear_cache()
        print(f"--- running {name} ...", flush=True)
        try:
            proc = subprocess.run(
                [sys.executable, name],
                cwd=HERE, capture_output=True, text=True, timeout=args.timeout,
            )
            status = classify(proc)
            stdout, stderr = proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            status = f"TIMEOUT(>{args.timeout}s)"
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")

        results.append((name, status))
        all_output.append((stdout or "") + (stderr or ""))
        print(f"    {status}", flush=True)

        chunks += [
            "=" * 72,
            f"{name}  ->  {status}",
            "=" * 72,
            f"(pre-test cache clear: {', '.join(cleared) if cleared else 'nothing to remove'})",
            "--- stdout ---",
            tail(stdout, args.tail),
            "--- stderr ---",
            tail(stderr, args.tail),
            "",
        ]

    width = max(len(n) for n in tests) if tests else 20
    summary = ["=" * 72, "SUMMARY", "=" * 72]
    summary += [f"  {n:<{width}}  {s}" for n, s in results]
    passed = sum(1 for _, s in results if s == "PASS")
    summary += ["", f"  {passed}/{len(results)} passed"]

    found_hints = []
    for msg in hints_for("\n".join(all_output)):
        if msg not in found_hints:
            found_hints.append(msg)
    if found_hints:
        summary += ["", "  Likely setup problems (not code bugs):"]
        summary += [f"    - {m}" for m in found_hints]
    summary += [""]

    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary + chunks))

    print("\n".join(summary))
    print(f"\nFull report written to: {REPORT}")
    print("Paste or attach that file and I'll work through the failures.")


if __name__ == "__main__":
    main()