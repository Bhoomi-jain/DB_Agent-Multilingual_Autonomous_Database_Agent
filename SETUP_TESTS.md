# Getting the db-agent test suite running on a fresh machine

Your run produced `0/14 passed`, but **none of those failures were code bugs** —
no test got far enough to execute a single line of the repair pipeline. Three
environment problems account for all fourteen.

| Cause | Tests affected |
|---|---|
| Postgres rejected `postgres:postgres` (`FATAL: password authentication failed`) | 12 |
| MySQL/MariaDB not running | 1 (`test_cache.py`) |
| No seeded `test_sqlite.db` → `no such table: customers` | 1 (`test_sqlite_repro.py`) |

Underneath all three is one root gap: **the baseline `customers` / `orders` /
`order_items` schema was seeded by hand in the old dev sandbox
(PROJECT_HANDOFF.md §7) and no seed script was ever committed.** On a fresh
machine there is nothing to seed it. `seed_testdb.py` fills that gap.

---

## 1. Point Postgres at the DSN the tests hardcode

The DSN `postgresql+psycopg2://postgres:postgres@localhost/testdb` is
hardcoded in 12 test files, so the lowest-friction fix is to make the server
match it rather than edit twelve files:

```bash
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"
sudo -u postgres createdb testdb          # skip if it already exists
```

If local connections are still refused after setting the password, `pg_hba.conf`
is likely set to `peer` or `ident` for local TCP. Change the `127.0.0.1/32` and
`::1/128` lines to `scram-sha-256` and `sudo systemctl reload postgresql`.

Prefer not to touch the server? Export a DSN instead — but note the tests
themselves still hardcode theirs, so this only affects `seed_testdb.py`:

```bash
export DB_AGENT_PG_URL="postgresql+psycopg2://youruser:yourpass@localhost/testdb"
```

To make the *tests* honour it too, replace the literal in each file:

```bash
sed -i 's|"postgresql+psycopg2://postgres:postgres@localhost/testdb"|os.getenv("DB_AGENT_PG_URL", "postgresql+psycopg2://postgres:postgres@localhost/testdb")|' test_*.py
```

(Then add `import os` where it's missing. I'd rather do this properly as a
shared `db_targets.py` module than by `sed` — say the word.)

## 2. Start MariaDB (only `test_cache.py` needs it)

```bash
sudo systemctl start mariadb     # or: sudo service mysql start
sudo mysql -e "ALTER USER 'root'@'localhost' IDENTIFIED BY 'rootpass';
               CREATE DATABASE IF NOT EXISTS testdb;"
```

Skipping this costs you exactly one test.

## 3. Seed the baseline fixture

```bash
python seed_testdb.py --target all      # or --target postgres
```

It drops and recreates the three base tables, inserts the baseline rows, and
then **self-checks** that the result satisfies what the suite asserts: exactly
three tables, exactly one Canadian customer, Alice leading on both revenue and
item count, at least five orders, and both foreign keys actually declared. If a
self-check fails it tells you which assertion would have broken.

Details that are load-bearing, not incidental:

- **Exactly three tables.** `pick_relevant_tables()` skips its LLM call at
  ≤ 3 tables, and `test_cache` asserts precisely 5 cache misses. A stray
  fourth table breaks about five tests — the §6.17 pollution bug.
- **Real declared foreign keys**, InnoDB on MySQL and `PRAGMA foreign_keys=ON`
  on SQLite. `_bridge_tables()`, `validate_join_semantics()`,
  `repair_join_path()` and `repair_missing_joins()` all build their graph from
  these; without them the repair tests cannot pass.
- **`unit_price` is NUMERIC/DECIMAL, not FLOAT.** Postgres NUMERIC arrives
  through MCP's JSON layer as the string `"59.88"` — that's §6.16, and
  `test_hybrid_agent` pins it with `assert str(expected) == "59.88"`.
- **Five orders** even though the baseline only has items for four:
  `test_hybrid_agent` inserts `order_items` referencing `order_id` 5.

`--clean-only` drops the base tables plus any leftovers from a test that
crashed between its `setup()` and `teardown()` (`shipments`, `site_settings`,
`product_sales`, `products`, `artist_extra_1..7`). Worth running if a later
run fails oddly — a stray table is exactly the §6.17 failure mode.

## 4. Re-run

```bash
python run_tests.py
```

The runner now deletes `.schema_cache.json` before **every** test. This matters
more than it looks: `CACHE_TTL_SECONDS = 300`, so re-running the suite within
five minutes would let `test_cache` and `test_retry` see cache *hits* where
they assert exactly 5 misses — a failure that reads like a caching bug but is
just stale fixture state. `--keep-cache` opts out. The summary now also flags
recognisable setup problems separately from real failures.

---

## What to expect on the next run

With Postgres and the fixture in place, most of the suite should pass — these
tests all passed in the original sandbox against this same baseline. Two worth
watching:

- **`test_exhausted_retries.py`** previously failed with
  `expected 2 retries, got 0`, and its "Correctly raised after exhausting
  retries" line was *spurious* — the `RuntimeError` came from `get_tables()`
  during schema discovery, not from the retry loop, and the `except
  RuntimeError` caught it indistinguishably. Since `run()` does schema
  discovery *before* entering the retry loop, any discovery failure looks
  identical to retry exhaustion from the test's vantage point. It should pass
  once Postgres works, but that assertion is weaker than it appears and is
  worth tightening to check the error message too.

- **`test_sqlite_repro.py`** has no assertions at all — it only prints, so it
  passes as long as nothing raises. Its `FakeLLM` is also scripted with just
  two responses (one SQL, one answer), so any failed first attempt makes
  attempt 2 pop the *answer* string and feed it to the SQL parser. That's what
  produced `Could not parse SQL: ... There is 1 customer from Canada.` in your
  report. Seeding fixes the trigger, but the script stays one retry deep.

Send me the new `test_report.txt` and I'll work through whatever's genuinely
broken in the code.
