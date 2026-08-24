# DB-Agent — Multilingual Autonomous Database Agent

A natural-language-to-SQL query agent with an **explicit, inspectable control loop** — no ReAct black box. Ask questions about your database in any language, in plain English or with meaning-based ("find products described as durable") searches, and get back verified answers.

Every LLM call and tool call happens exactly where the code says it does, in the order it says, with metrics to prove it. Wrong SQL is not just retried — whole classes of mistakes are **detected and repaired deterministically** before they ever reach the database.

## Features

- **Explicit control loop** (`core_agent.py`) — schema discovery → table picking → SQL generation → validation → execution → bounded retry → answer formatting. Happy path = 3 LLM calls, regardless of database size.
- **Genuinely multilingual** (`multilingual.py`) — detects the question's language, reasons over the database in English, replies in the language you asked in. Translation cost is included in reported metrics.
- **RAG + SQL hybrid** (`hybrid_agent.py`) — routes each question: pure structured data → SQL; pure meaning-based lookup → vector search; both (e.g. "total revenue from eco-friendly products") → vector search narrows row IDs first, then SQL aggregates over exactly those rows.
- **Self-built, security-reviewed MCP server** (`db_mcp_server.py`) — every query is parsed into a real AST (sqlglot) and rejected unless it is a single read-only SELECT/CTE/UNION. Catches DML/DDL smuggled inside subqueries, blocks `;`-stacked payloads, caps result size. This exists because the most widely used community Postgres MCP server has a documented injection escape.
- **Deterministic SQL repair pipeline** — see below.
- **Schema caching** — table list / column schemas / foreign keys cached per-database with a TTL, so repeat CLI invocations skip discovery entirely.
- **Metrics on every run** — LLM calls, tool calls, cache hits/misses, retries, semantic rejections, answer verification status, per-step timings.
- **Statistical schema profile** (`schema_profile.py`) — row counts, column ranges and confirmed 1:N edges power execution-based validation (impossible results rejected before answering).
- **Accuracy harness** (`benchmark.py`) — exact-match / execution accuracy, retry and hallucination rates against 14 hand-verified Chinook golds, with before/after baselines.
- Pluggable LLM backend: local Ollama (default, `llama3.2`) or Anthropic API.

## The repair pipeline

A text-to-SQL model makes *predictable classes* of mistakes. Each observed class got a targeted, deterministic countermeasure in `core_agent.py`:

| Failure class | Countermeasure |
|---|---|
| Wrong-dialect syntax (`SELECT TOP n` on SQLite) | `_try_repair` — sqlglot transpiles from alternate dialects |
| Undeclared table alias (`C.name`, never declared) | `repair_undefined_aliases` — resolves by column membership |
| Hallucinated join (`Artist.ArtistId = Invoice.CustomerId`) | `validate_join_semantics` + `repair_join_path` — BFS over real FK graph rewrites multi-hop joins |
| **Join fan-out double-counting** (`SUM(Invoice.Total)` ⋈ InvoiceLine ≈ 9× inflation) | `validate_aggregation_fanout` — lineage-based: flags aggregating FROM-root-ancestor columns while a FK-child is joined (detail-side measures are line-grain scaled and immune); auto-repair drops the provably-unneeded child join |
| **T-SQL `+` string concat** silently becoming numeric addition | `repair_string_concat` — text-typed `+` chains rewritten to `\|\|` |
| **Metric confusion** ("customers who *spent* most" answered with COUNT or quantity-SUM) | `validate_metric_intent` (lexical + integer-measure guard) + `validate_plan_matches_sql` (structural, vs the LLM's own declared query plan) |
| Table referenced but never joined | `repair_missing_joins` — inserts the missing JOIN chain |
| Structurally valid but answers the wrong question ("average X per customer" without GROUP BY) | `validate_grouping_intent` — rejects with an actionable error for the retry loop |
| Ranking done wrong (no ORDER BY / wrong direction / LIMIT overshoot) | plan-vs-SQL ranking checks — top-N must sort by the planned metric in the planned direction |
| **Grain confusion** ("most purchases" counted as line-items) | `validate_count_grain` + `repair_count_grain` — FK-topology grain inference retargets COUNT to the document level |
| Recomputed measures (`SUM(qty × price)`) where a stored total exists | semantic optimizer — learns `Invoice.Total ≡ Σ(qty×price)` via paired probes (persisted to cache), then safely rewrites ungrouped scalar forms; grouped shapes get annotations |
| Opaque retries | `semantic_diff` — every retry logs structural tags (`AGGREGATE_CHANGE`, `JOIN_DROPPED`, …) into metrics and the report |
| Arbitrary top-N cutoffs dropping ties | `rewrite_top_n_with_ties` — RANK() rewrite so boundary ties are kept |
| Answer text editorializing / fabricating figures | `verify_answer` — every number must trace to real results (comma/rounding-tolerant), one corrective retry |

Before SQL is even generated, Step 2 produces a **structured query plan** (`{tables, metric, metric_column, entity, ranking, grouping}`) from a single merged LLM call — generation is conditioned on it and validation enforces it. Schema context also carries auto-derived **fact/dimension hints** ("Invoice is transactional — totals already aggregated per event") so small models stop fanning out fact tables against their line items.

Anything that can't be fixed deterministically is fed back to the model as a specific, bounded retry — never silently guessed at.

## Architecture

```
 you ──► detect language ──► translate question to English (if needed)
                                     │
                                     ▼
                     ┌─── route (hybrid_agent.py only) ───┐
                     │        │              │            │
                     ▼        ▼              ▼            │
                   SQL    SEMANTIC       HYBRID           │
                     │        │         vector search    │
                     │        │         → row IDs        │
                     ▼        ▼              │            │
                core_agent.SQLAgent  ◄───────┘            │
                1. list tables (cached)                   │
                2. pick relevant tables (LLM, names only) │
                3. bridge FK gaps via BFS                 │
                4. generate SQL (filtered schema only)    │
                5. validate + deterministic repairs       │
                6. execute via read-only MCP server       │
                7. bounded retry w/ real DB error text    │
                8. format + verify answer numbers         │
                                     │
                                     ▼
                     English answer ──► translate back ──► you
```

All database access goes through `db_mcp_server.py` over stdio MCP — the agent never opens a DB connection itself. Embeddings for the hybrid agent come from a pluggable provider: fully-local TF-IDF+SVD (no model download) or Ollama dense embeddings (`nomic-embed-text`). Vector similarity uses explicit cosine distance so one relevance threshold is meaningful across embedding providers.

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) running with a model pulled (`ollama pull llama3.2` — the default; other models work via `--model`, but "thinking" models like `qwen3:4b` also need `--max-tokens 1024` or their reasoning exhausts the default budget before any answer is produced) — or set `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY`

### Installation

```bash
git clone https://github.com/Bhoomi-jain/DB_Agent—Multilingual_Autonomous_Database_Agent.git
cd DB_Agent—Multilingual_Autonomous_Database_Agent

# Using uv (recommended)
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .

# Or standard pip
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
```

No API key required for the default local setup.

## Usage

Four entry points share the same engine:

```bash
# Explicit-control-loop agent against any SQLAlchemy-supported database
python core_agent.py "Which customer generated the most revenue?" \
    --db-url "postgresql+psycopg2://user:pass@localhost/mydb" --metrics

# Multilingual wrapper (defaults to the bundled Chinook sample DB)
python multilingual.py "कनाडा से कितने ग्राहक हैं?"
python multilingual.py "¿Cuáles son los 5 álbumes más vendidos?" --db-url sqlite:///chinook.db

# RAG + SQL hybrid over a table with a free-text column
python hybrid_agent.py "Total revenue from products described as eco-friendly?" \
    --db-url "$DATABASE_URL" \
    --vector-table products --vector-text-column description --vector-id-column product_id

# Agentic variant (LangChain create_agent loop) for Postgres/MySQL
python production_agent.py "Top 5 customers by revenue?" --db-url "$DATABASE_URL"
```

Useful flags: `--provider ollama|anthropic`, `--model`, `--max-retries`, `--no-cache`, `--verbose` (step-by-step logs), `--metrics`.

The bundled `chinook.db` is the classic [Chinook](https://github.com/lerocha/chinook-database) digital-media-store sample (gitignored — download with `curl -L -o chinook.db https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite`).

## Testing

The suite (21 scripts) runs each test in its own subprocess against real databases — no mocked DB layer — using scripted fake LLMs so tests are deterministic:

```bash
# One-time: create + seed the fixture databases the suite expects
python seed_testdb.py --target all          # postgres + mysql + sqlite

# Run everything, writes test_report.txt
python run_tests.py
```

Connection targets are centralized in `db_targets.py` and overridable without editing tests:

```bash
export DB_AGENT_PG_URL="postgresql+psycopg2://user:pass@localhost/testdb"
```

Fresh-machine setup details (Postgres auth, MariaDB, pg_hba): see [SETUP_TESTS.md](SETUP_TESTS.md). Project history and design rationale: see [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md).

## Measuring accuracy

`benchmark.py` runs a curated set of 14 Chinook questions with hand-verified gold answers through the live pipeline and reports the numbers:

```
exact-match accuracy   : 71.4%      # gold figure(s) appear in the answer
execution accuracy     : 78.6%      # executed result == gold result
retry rate             : 35.7%
hallucination rate     : 7.1%       # verification / column-hallucination events
failure classes        : AGGREGATION_ERROR, COLUMN_HALLUCINATION, ...
repairs applied/skipped: 4/0        # deterministic-repair budget usage
```

```bash
python benchmark.py --out run1.json            # run + save
python benchmark.py --baseline run1.json       # before/after deltas
```

Every question is classified on failure (join / grain / aggregation / column-hallucination / execution / verification), so regressions say *what* broke, not just that something did. The statistical profile that powers execution-validation is built once per database and cached alongside schemas.

## Project Structure

```
├── core_agent.py          # Explicit text-to-SQL control loop + repair pipeline + CLI
├── sql_semantics.py       # Grain inference, measure-source resolution, semantic diff
├── schema_profile.py      # Statistical DB model: column stats, 1:N edges, ranges
├── multilingual.py        # Language detect/translate wrapper around SQLAgent + CLI
├── hybrid_agent.py        # RAG+SQL router: sql / semantic / hybrid + CLI
├── production_agent.py    # Agentic (create_agent) variant, LLM factory shared by all
├── db_mcp_server.py       # Security-reviewed read-only MCP server (AST validated)
├── vector_store.py        # chromadb-backed semantic store, pluggable embedders
├── benchmark.py           # Accuracy harness: exact-match / exec-accuracy / retry /
│                          # hallucination rates + failure-class breakdown vs golds
├── benchmark/             # gold_chinook.json (hand-verified answers) + results
├── db_targets.py          # Central, env-overridable test DSNs
├── seed_testdb.py         # Recreates the baseline test fixture (self-checking)
├── run_tests.py           # Suite runner: subprocess isolation + report generation
├── SETUP_TESTS.md         # Fresh-machine environment setup guide
├── PROJECT_HANDOFF.md     # Design decisions, bug-history compendium, roadmap
├── test_*.py              # 21 self-contained end-to-end tests
├── chinook.db             # Sample DB (gitignored)
└── pyproject.toml / uv.lock
```

## Requirements

All dependencies are pinned in `pyproject.toml`: langchain, langgraph, langchain-ollama, langchain-anthropic, langchain-mcp-adapters, mcp, langdetect, sqlalchemy, psycopg2-binary, pymysql, sqlglot, chromadb, scikit-learn, httpx, python-dotenv, rich.

> Note: the original release depended on Anthropic's archived `mcp-server-sqlite`. It is no longer needed — `db_mcp_server.py` replaces it (and fixes the read-only escape its community Postgres sibling was vulnerable to).

## License

MIT

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/), [LangGraph](https://github.com/langchain-ai/langgraph), and [sqlglot](https://github.com/tobymao/sqlglot)
- Vector search via [ChromaDB](https://www.trychroma.com/)
- Sample data from the [Chinook Database](https://github.com/lerocha/chinook-database)
- Powered by [Ollama](https://ollama.com/) and Meta Llama 3.2 (default model)
