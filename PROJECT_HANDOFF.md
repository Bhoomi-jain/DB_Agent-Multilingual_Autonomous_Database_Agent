# PROJECT_HANDOFF.md

Everything a contributor (or future-me) needs to pick this project up:
where it stands, why the architecture looks the way it does, the full
history of observed failures and what was built in response, and what to do
next. For environment setup on a fresh machine, read [SETUP_TESTS.md](SETUP_TESTS.md)
alongside this file.

---

## 1. Status snapshot

**Working state:** the full test suite passes — 21/21 end-to-end tests
against live database servers (`python run_tests.py`, report written to
`test_report.txt`). Postgres is required by most tests; MySQL only by
`test_cache.py` (which now falls back to Postgres when MariaDB is down);
SQLite by `test_multilingual.py` + `test_sqlite_repro.py`.

**Entry points, newest to oldest:**

| File | What it is | Status |
|---|---|---|
| `core_agent.py` | Explicit control-loop text-to-SQL engine + CLI. The heart of the project. | Current |
| `multilingual.py` | Language detect/translate wrapper around SQLAgent; restores the v0.1 namesake feature on the new engine | Current |
| `hybrid_agent.py` | RAG+SQL router: sql / semantic / hybrid routes; vector search constrains the SQL half | Current |
| `production_agent.py` | Agentic `create_agent()` variant for Postgres/MySQL; also home of `build_llm()` shared by everything | Current |
| `agent.py` | Original v0.1 (LangChain ReAct loop + upstream `mcp-server-sqlite`) | **Deprecated**, kept for reference only |

**Versioning:** `pyproject.toml` says 0.2.0 (the RAG+SQL phase). The
multilingual port and test-infrastructure work described here would
reasonably be 0.3.0.

**Default LLM:** `llama3.2:latest` via `build_llm()` (overridable with
`--model` / `$LLM_MODEL`). Switched from qwen3:4b after live runs showed
qwen3 emitting chain-of-thought despite `"think": false` — including an
orphan-`</think>` shape and fully tag-less rambling that could consume the
entire num_predict budget before any answer text (see §6.10 and the
truncation guard in `format_answer`). Llama 3.2 has no thinking mode, so
answers come out clean by construction; qwen3 remains selectable for
harder reasoning workloads but REQUIRES a raised budget (`--max-tokens
1024`) — at the 512 default all three retry attempts were observed to
fail on truncated think-blocks alone.

---

## 2. Architecture decisions and their rationale

### 2.1 Explicit control loop instead of an agent loop

`core_agent.py` deliberately does NOT use LangChain's `create_agent()`.
An agentic tool-loop spends 5–10+ LLM calls per question wandering the
schema turn by turn, and you can't tell which call failed or why. The core
pipeline is a fixed sequence with exactly 3 LLM calls on the happy path:

1. `get_tables()` — table NAMES only (cheap; skipped entirely from cache)
2. `pick_relevant_tables()` — one LLM call, names only, no schema fetched yet;
   ALSO extracts a structured query plan (metric/entity/ranking/grouping) in
   the same call; **skipped entirely when the DB has ≤ 3 tables** (plan is
   then None and validation degrades to the lexical checks — load-bearing
   for tests, see §4.2)
3. `generate_sql()` — one LLM call, prompt contains ONLY the relevant tables'
   schemas plus auto-derived fact/dimension hints from the FK graph
4. validation + deterministic repair (local, no LLM): read-only AST → join
   semantics (+BFS path repair) → fan-out detection → metric intent →
   plan-vs-SQL consistency (incl. ranking direction/LIMIT) → qualifier
   resolution → grouping intent — all feeding the bounded retry with
   actionable messages — THEN the tie-aware top-N rewrite as a trusted
   final mechanical transform
5. `execute_sql()` — one MCP tool call
6. bounded retry loop — real DB error text fed back verbatim
7. `format_answer()` — one LLM call
8. answer verification — numbers must trace to real results

### 2.2 Self-built MCP server instead of a community one

`db_mcp_server.py` exists because the popular community servers trust that
a query *looks* read-only. This one parses every statement into a sqlglot
AST, allows only a single top-level SELECT/CTE/set-op, walks the whole tree
for DML/DDL smuggled in subqueries, rejects `;`-stacked payloads, blocks
exfiltration/sleep functions, and caps result rows. The agent process itself
never holds a DB connection — every access, including schema discovery,
is an MCP tool call over stdio.

One persistent MCP session is held per `SQLAgent.run()` rather than letting
langchain-mcp-adapters spawn a fresh subprocess + handshake per tool call
(that default produced most of the wall-clock overhead in early profiling).

### 2.3 Deterministic repair before LLM retry

The guiding rule, learned from watching retries fail: **if a mistake has a
deterministic fix, apply it in code; never hope the model self-corrects.**
Real observation driving this: an undeclared alias (`C.name`) was retried
three times with the model making the *identical* mistake each time. Every
repair function is deliberately conservative — it acts only when the fix is
unambiguous (single FK path, single column-membership match) and otherwise
leaves the query alone for validation/retry to handle loudly. The complete
failure-class → countermeasure table is in README.md.

### 2.4 Schema caching, invalidated loudly

`SchemaCache` is a JSON file keyed by db_url with per-entry TTL (300 s),
version-stamped (`CACHE_SCHEMA_VERSION = 2`) so format changes discard old
files instead of crashing on them — which happened for real. `run_tests.py`
deletes the cache before EVERY test because re-runs within the TTL would
turn asserted cache misses into hits.

### 2.5 Cosine distance in the vector store

chromadb defaults to squared-L2, whose scale tracks embedding magnitude.
TF-IDF+SVD vectors happened to sit where the hybrid threshold (1.2) worked;
real dense models produce L2 distances in the thousands and silently matched
nothing. `VectorStore` now forces `hnsw:space=cosine` (bounded [0, 2],
scale-independent) and `DEFAULT_DISTANCE_THRESHOLD = 1.0`. When nothing
passes the threshold, the closest rejected match is printed so a miscalibrated
threshold is distinguishable from a genuine no-match.

### 2.6 Multilingual as a wrapper, not integration

Translation is orthogonal to SQL reasoning, so it wraps any `(answer, sql,
metrics)`-shaped agent rather than living inside `run()`. Two reliability
details: translation cost is folded into the SAME Metrics summary (honest
reporting), and figures surviving answer-verification are re-checked after
back-translation because LLM translation can silently corrupt them
("59.88" → "59,88").

---

## 3. Bug-history compendium (observed failure → fix)

The historical numbering referenced in older docs (§6.x of a lost handoff)
is preserved here for continuity.

| Ref | Observed failure | Fix |
|---|---|---|
| §6.1 | Wrong-dialect SQL: T-SQL `SELECT TOP n` against SQLite | `_try_repair` transpiles via sqlglot from alternate dialects |
| §6.2 | Hallucinated join `Artist.ArtistId = Invoice.CustomerId` ran and returned confident nonsense | `validate_join_semantics` + `_bridge_tables` |
| §6.3 | Same class, but a real multi-hop path existed | `repair_join_path` BFS-rewrites through bridge tables |
| §6.4 | Undeclared alias `C.name`; model repeated identical broken SQL 3× | `repair_undefined_aliases` |
| §6.5 | Model resolved "no such column" by DELETING the InvoiceLine reference — query silently changed meaning | `repair_missing_joins` + `validate_all_qualifiers_resolved` |
| §6.6 | Final answer editorialized: "(likely a typo)" next to a real value | answer verification + corrective retry |
| §6.7 | Over-fetching: whole DB schema described up front regardless of question | describe_table only for picked tables (`test_large_schema` pins tool-call count) |
| §6.8 | langchain-mcp-adapters opened new session per tool call | persistent session in `_connect()` |
| §6.9 | MCP execution errors arrived as ordinary text, invisible to the retry loop | `_call_tool` raises `RuntimeError` on `Error executing tool` |
| §6.10 | Qwen3 leaked `<think>` blocks into .content despite think=false | `_strip_thinking` defensively strips them |
| §6.11 | Old cache-file format crashed new code | `CACHE_SCHEMA_VERSION` guard discards stale formats |
| §6.12 | Naive "first SELECT keyword" extraction grabbed narration prose | candidate spans all validated by real parser |
| §6.13 | LIMIT N arbitrarily dropped rank ties | `rewrite_top_n_with_ties` RANK() rewrite |
| §6.14 | "average X per customer" answered without GROUP BY — plausible wrong number | `validate_grouping_intent` actionable rejection |
| §6.15 | Answer verification false-positive on band name "U2" / list indices | letter-boundary regex + list-marker stripping |
| §6.16 | Postgres NUMERIC arrives through MCP JSON as string `"59.88"`; never matched answer text | `_extract_numbers_from_rows` float-parses strings; fixture pins NUMERIC type |
| §6.17 | A stray 4th test table flipped the ≤3-tables skip and broke ~5 unrelated tests | `seed_testdb.py` sweeps ephemeral tables; runner clears cache per test |
| §6.18 | chromadb default squared-L2 made distance thresholds meaningless across embedders | forced cosine space + diagnostic printout (§2.5) |
| §6.19 | `test_exhausted_retries` couldn't distinguish discovery crash from true exhaustion | assertion now requires the "Failed after" message |
| §6.20 | Fan-out double-count: `SUM(Invoice.Total) JOIN InvoiceLine` returned ~9× real revenue — valid FK join, multiplied rows | `validate_aggregation_fanout` (grouped-grain / DISTINCT / child-measure escapes; grouped-COUNT is a deliberate, documented blind spot owned by the metric layer) |
| §6.21 | Metric confusion: "customers who *spent* the most" answered with COUNT(invoices)=7-each | `validate_metric_intent` (lexical) + plan-layer `validate_plan_matches_sql` (structural — catches keyword-unanticipated phrasings) + ranking direction/LIMIT checks |
| §6.22 | Answer verification false positives: comma-grouped "20,848.62" split into two tokens; "195.10" ≠ raw 195.1 as strings | extractors normalized to floats with thousands-comma stripping; decimal citations match at their stated precision, integer claims stay strict |
| §6.23 | Live-tuning the evaluation layer against llama3.2 taught three rules: (a) plans from small models are NOISY — ranking/metric claims are only enforced when lexically corroborated by the question, hallucinated metric_columns fall back to numeric-any; (b) T-SQL `+` string concat silently becomes numeric addition on SQLite (names collapsed to 0 → answer layer hallucinated names to fill the void) — `repair_string_concat` rewrites text-typed `+` chains to `\|\|`; (c) fan-out detection v2 is LINEAGE-based: only tables in the FROM-root's FK-ancestor line get inflated by a joined child; detail-side measures (`SUM(line.qty * price)`) are line-grain scaled and immune — the first structural draft false-positived on exactly that shape |
| §6.24 | Three more live findings on llama3.2: (a) plan said ASC for a "MOST" question and the word-only corroboration let it kill 3 attempts of correct DESC SQL — direction is now POLARITY-corroborated (`most/top/best` forces DESC, `least/lowest` forces ASC); (b) unqualified `FirstName + ' ' + LastName` slipped the concat repair (no alias to resolve types from) — unique-name schema resolution added; (c) `SUM(IL.Total)` (nonexistent column) passed every static layer and died as a cryptic sqlite error — `validate_columns_exist` now checks qualified/unqualified existence + ambiguity against fetched schemas with close-match suggestions, and whitelists SELECT-list aliases used in ORDER BY. Known remaining blind spot: COUNT-grain confusion ("purchases" counted as InvoiceLines instead of Invoices — same tie-set here, mislabeled figures) needs plan-level grain semantics; accepted for llama3.2-class models |
| §6.25 | v3 integration bugs caught by the suite mid-build: `(0,0)` tuple used as a falsy guard (non-empty tuples are truthy); sqlglot stores Select-FROM under key `from_`; underscore-splitting made `order_lines` fragment-match `orders`; inverted child→parent map lookups; probe SQL lost its SUM wrapper or reused dead aliases; self-referencing FKs (Employee↔Employee) sent depth fixpoints into an infinite loop | fixes are the shipped code — the lesson is that every one was caught by a failing assert within seconds of introduction |

---

### 3.1 Semantic layer (grain inference · measure optimizer · attempt diffing)

Added after live llama3.2 benchmarking exposed two more failure families:

- **Grain inference** (`sql_semantics.py`): FK-topology classification
  (event_root / detail / dimension / standalone + depth), question-to-grain
  mapping (stem matching + entity-document heuristic + activity-verb
  gating), and static effective-count-grain extraction shared with the
  fan-out detector via `root_and_detail_side`. `validate_count_grain`
  rejects document-vs-line confusion; `repair_count_grain` deterministically
  retargets COUNT to the document PK and demotes detail joins/promotes the
  document table into FROM.
- **Measure-equivalence learning**: when a retry pair moves from detail
  arithmetic to a lineage column, one paired whole-table probe verifies
  equality (0.5% tolerance); matches persist to `.schema_cache.json` v3
  (`measure_equiv`). The optimizer then rewrites UNGROUPED scalar
  SUM(detail-arithmetic) to SUM(stored column) — provably safe shapes only;
  grouped queries are annotated, never rewritten.
- **Semantic diff**: every retry emits structural tags
  (AGGREGATE_CHANGE / JOIN_DROPPED / FILTER_CHANGED / ...) into
  `Metrics.attempt_diffs` and the report.

Known blind spots (documented, deliberate): grouped-COUNT numeric-vs-grain
confusion; answer verification checks figures only, not name strings.

### 3.2 Measurable layer (v3): profile · execution validation · classification · benchmark

- **schema_profile.py**: one-probe-per-table statistical model (row_count,
  numeric min/max/avg/ndistinct), persisted in cache **v4** (`profile`).
  Derived: identifier-vs-measure, confirmed-1:N edges (child/parent row
  ratio >= 1.5), event depths. Self-referencing FKs ignored; hard round-cap.
- **Graph-first resolution** (`resolve_measure_source`): measure/grain
  expectations come from plan.tables/entity + graph geometry + column
  stats. Stem matching DEMOTED to last-resort fallback (weak-signal flag).
  `sold/sales` forces the sales-lineage fact table as source (overrides
  noun-derived shallower picks like Track-for-"tracks sold").
- **Execution-based validation**: post-execute, pre-format gate rejects
  statistically impossible results (negative non-negative sums, AVG outside
  profiled range, COUNT > total rows).
- **FailureClass taxonomy** wired at every site; Metrics carries attempts /
  repairs_applied / repairs_skipped / failure_classes.
- **Repair governance**: budget of 2 deterministic repairs per query;
  beyond it repairs are skipped and the bounded LLM retry takes over.
- **benchmark.py** + `benchmark/gold_chinook.json` (~14 hand-verified
  Chinook golds): exact-match accuracy, execution accuracy, retry rate,
  hallucination rate, failure-class breakdown; `--baseline prev.json`
  prints before/after deltas.

## 4. Test infrastructure

### 4.1 How it runs

`run_tests.py` executes each `test_*.py` in its own subprocess (one crash
can't take down the rest), clears `.schema_cache.json` first, classifies
failures (assert vs error vs timeout), recognizes known setup problems, and
writes a pasteable `test_report.txt`.

Tests are end-to-end against REAL databases — no mocked DB layer. The LLM
is scripted (`FakeLLM` pops pre-written responses in order), which makes
runs deterministic and fast (~0.8 s each).

### 4.2 Load-bearing fixture details

- **Exactly three base tables** (`customers`/`orders`/`order_items`): the
  ≤3-tables skip in `pick_relevant_tables` means several tests script NO
  table-pick response; a stray fourth table desyncs every scripted FakeLLM.
  Self-contained tests create/drop their extras in setup/teardown.
- **Real declared FKs** (InnoDB on MySQL): the entire repair machinery
  builds its join graph from `list_foreign_keys`.
- **NUMERIC not FLOAT** for unit_price: pins the §6.16 string-typing path.
- **Five orders though four have items**: `test_hybrid_agent` inserts rows
  referencing order_id 5.
- **Alice leads** both completed-revenue and item count; she is the only
  Canadian customer — several assertions depend on both facts.

### 4.3 Centralized DSNs

All connection strings live in `db_targets.py`, env-overridable
(`DB_AGENT_PG_URL`, `DB_AGENT_MYSQL_URL`, `DB_AGENT_SQLITE_URL`). Tests
import; nobody hardcodes. `pick_backend()` prefers MySQL when its port is
up and falls back to Postgres otherwise.

### 4.4 Known remaining weaknesses (accepted-for-now)

- Scripted-response coupling: any change to schema size or the ≤3-tables
  threshold shifts FakeLLM response order (fails loudly as IndexError, but
  cryptically). Since the merged plan call, >3-table tests script plan
  OBJECTS ({"tables": [...]}) rather than bare arrays — old-format scripts
  fail open to all-tables and trip tool-count assertions like
  test_large_schema's.
- Scripts, not pytest: top-level `asyncio.run(main())` would execute at
  pytest collection time. Fine under `run_tests.py`; a pytest migration
  needs fixtures + collection guards.
- Exact arithmetic assertions (5 cache misses, 6 tool calls) pin internal
  behavior — they break intentionally when the pipeline changes, but that
  cuts both ways.
- Ollama dense embeddings are far less exercised than TF-IDF (no live-model
  test); the cosine-space fix is verified synthetically.

---

## 5. Environment pointers

- Fresh-machine setup (Postgres auth, pg_hba, MariaDB, seeding): SETUP_TESTS.md
- Fixture seeding with self-check: `python seed_testdb.py --target all`
- Full suite: `python run_tests.py` → `test_report.txt`
- Service expectations: PostgreSQL :5432 reachable for 13 tests; MySQL :3306 optional; Ollama :11434 only for live-model runs; ChromaDB runs embedded (no server, no port)

---

## 6. Roadmap — suggested next steps

1. **CI**: `.github/` is empty. A workflow installing deps + starting
   postgres service containers + `python seed_testdb.py && python run_tests.py`
   would lock in the currently-manual verification loop.
2. **pytest migration**: keep scripts runnable standalone but add pytest
   wrappers/fixtures so IDEs and standard tooling can collect them.
3. **Commit hygiene**: working tree carries the Phase-2 fixes (cosine space,
   threshold diagnostics, MySQL fallback), the whole test infrastructure,
   multilingual port, and these docs — commit in logical chunks.
4. **Conversation memory**: every entry point is single-shot; multi-turn
   follow-ups ("and Germany?") are the natural next UX step.
5. **Write-path story**: everything is read-only by design. A guarded,
   confirmed, transaction-scoped write mode is a product decision more than
   a technical one — decide deliberately.
6. ~~Live-model evaluation harness~~ **DONE** — `benchmark.py` +
   `benchmark/gold_chinook.json` (14 verified golds) now report exact-match,
   execution accuracy, retry rate, hallucination rate and failure-class
   breakdowns; use `--baseline` for before/after deltas. Next increment:
   widen the gold set and sweep models (qwen3 vs llama3.2).
7. **Embedding provider parity**: exercise the Ollama embedding path with a
   live-model test once a CI runner can host one.
