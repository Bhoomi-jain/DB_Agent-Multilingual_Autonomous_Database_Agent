"""
core_agent.py — Explicit, inspectable text-to-SQL control loop.

No langchain `create_agent()`, no ReAct black box. Every LLM call and every
tool call happens exactly where this code says it does, in this order:

  1. get_tables()            — list tables (cached across CLI runs)
  2. pick_relevant_tables()  — 1 cheap LLM call, table NAMES only, not schema
  3. generate_sql()          — 1 LLM call, using only the filtered schema
  4. validate_sql()          — local, no LLM/network call (fast, cheap)
  5. execute_sql()           — 1 MCP tool call
  6. retry loop              — explicit `for` loop with the DB error fed
                                back into the next generate_sql() call,
                                bounded by --max-retries (not "hope the
                                model notices and retries on its own")
  7. format_answer()         — 1 LLM call

Happy path = 3 LLM calls total, regardless of how many tables the DB has.
Compare to an agentic tool-loop, which can spend 5-10+ LLM calls per
question exploring the schema turn by turn.

Performance:
  - Schema filtering: describe_table/list_foreign_keys results are trimmed
    to only the tables picked as relevant before they ever go in a prompt.
  - Caching: table list / column schemas / FKs are cached to a local JSON
    file keyed by db_url, with a TTL — so the *next* CLI invocation against
    the same database skips list_tables/describe_table/list_foreign_keys
    entirely (these rarely change between runs of a CLI tool).

Reliability:
  - Every retry is an explicit, counted loop iteration with the actual DB
    error text fed back to the model — not a prompt instruction hoping the
    model self-corrects.
  - Every step logs what happened; metrics (LLM calls, tool calls, cache
    hits/misses, retries, per-step timing) are collected and reported.
"""
import os
import sys
import json
import time
import hashlib
import asyncio
import argparse
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
import sqlglot
from sqlglot import exp
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler

from db_mcp_server import validate_readonly, ReadOnlyViolation
from production_agent import build_llm  # reuse the ollama/anthropic factory

load_dotenv()
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_path=False, show_time=True)],
)
logger = logging.getLogger("core_agent")
logger.setLevel(logging.WARNING)  # quiet by default; --verbose turns this up

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".schema_cache.json")
CACHE_TTL_SECONDS = 300
DEFAULT_MAX_RETRIES = 2
CACHE_SCHEMA_VERSION = 2  # bump whenever SchemaCache's on-disk shape changes


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    llm_calls: int = 0
    tool_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    retries: int = 0
    semantic_rejections: int = 0
    answer_verified: bool = True
    verification_retries: int = 0
    step_timings: dict = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)

    def record(self, step: str, seconds: float):
        self.step_timings[step] = self.step_timings.get(step, 0.0) + seconds

    def summary(self) -> str:
        total = time.time() - self.start_time
        lines = [
            f"LLM calls:     {self.llm_calls}",
            f"Tool calls:    {self.tool_calls}",
            f"Cache hits:    {self.cache_hits}",
            f"Cache misses:  {self.cache_misses}",
            f"Retries used:  {self.retries}",
            f"Semantic rejections: {self.semantic_rejections}",
            f"Answer verified: {self.answer_verified}"
            + (f" (after {self.verification_retries} retry)" if self.verification_retries else ""),
            f"Total time:    {total:.2f}s",
        ]
        if self.step_timings:
            lines.append("Breakdown:")
            for step, secs in sorted(self.step_timings.items(), key=lambda x: -x[1]):
                lines.append(f"  {step}: {secs:.2f}s")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Schema cache (file-backed, so it survives across separate CLI invocations)
# ---------------------------------------------------------------------------

class SchemaCache:
    def __init__(self, path: str = CACHE_FILE, ttl: int = CACHE_TTL_SECONDS):
        self.path = path
        self.ttl = ttl
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    data = json.load(f)
            except Exception:
                return {}
            # Guard against a cache file written by an older, incompatible
            # SchemaCache format (this is exactly what caused a real crash:
            # an old whole-schema cache entry being read as the new
            # per-key {"value":..., "fetched_at":...} shape). If the
            # version doesn't match, treat it as empty rather than crash —
            # it'll just refill itself as cache misses on this run.
            if data.get("_version") != CACHE_SCHEMA_VERSION:
                logger.info("[cache] schema version changed — discarding old cache file")
                return {"_version": CACHE_SCHEMA_VERSION}
            return data
        return {"_version": CACHE_SCHEMA_VERSION}

    def _save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._data, f)
        os.replace(tmp, self.path)

    @staticmethod
    def _key(db_url: str) -> str:
        return hashlib.sha256(db_url.encode()).hexdigest()[:16]

    def _fresh(self, entry: Optional[dict]) -> bool:
        return bool(entry) and (time.time() - entry["fetched_at"] <= self.ttl)

    def get_tables(self, db_url: str) -> Optional[list]:
        entry = self._data.get(self._key(db_url), {}).get("tables")
        return entry["value"] if self._fresh(entry) else None

    def set_tables(self, db_url: str, tables: list):
        self._data.setdefault(self._key(db_url), {})["tables"] = {
            "value": tables, "fetched_at": time.time()
        }
        self._save()

    def get_table_schema(self, db_url: str, table: str) -> Optional[list]:
        entry = self._data.get(self._key(db_url), {}).get("table_schemas", {}).get(table)
        return entry["value"] if self._fresh(entry) else None

    def set_table_schema(self, db_url: str, table: str, schema: list):
        db_entry = self._data.setdefault(self._key(db_url), {})
        db_entry.setdefault("table_schemas", {})[table] = {
            "value": schema, "fetched_at": time.time()
        }
        self._save()

    def get_foreign_keys(self, db_url: str) -> Optional[list]:
        entry = self._data.get(self._key(db_url), {}).get("foreign_keys")
        return entry["value"] if self._fresh(entry) else None

    def set_foreign_keys(self, db_url: str, fks: list):
        self._data.setdefault(self._key(db_url), {})["foreign_keys"] = {
            "value": fks, "fetched_at": time.time()
        }
        self._save()


# ---------------------------------------------------------------------------
# Small parsing helpers (no LLM structured-output framework — just regex)
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    """Some Ollama/model versions still emit <think>...</think> reasoning
    inline in .content even when reasoning=False ("think": false) is
    requested — the flag isn't universally honored depending on Ollama
    server version. Strip it defensively so downstream parsing isn't
    corrupted by chain-of-thought prose that was never meant to be output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _sqlglot_dialect(dialect_label: str) -> str:
    return {"PostgreSQL": "postgres", "MySQL": "mysql", "SQLite": "sqlite"}.get(dialect_label, "")


# Alternate dialects to try repairing a candidate against when it fails to
# parse under the actual target dialect. Covers the mistake actually
# observed in practice: a model trained heavily on T-SQL/MSSQL examples
# emitting "SELECT TOP n" against a SQLite/Postgres/MySQL database, which
# don't support that syntax (they use LIMIT).
_REPAIR_SOURCE_DIALECTS = ["tsql", "postgres", "mysql", "sqlite", "oracle", ""]


def _try_repair(candidate: str, target_dialect: str) -> Optional[str]:
    """`candidate` failed to parse as target_dialect SQL. Check whether it's
    valid SQL in a *different* dialect (a real, observed LLM failure mode —
    e.g. SQL Server's `TOP n` on a SQLite database) and if so, transpile it
    to the target dialect via sqlglot rather than discarding a query that
    was actually correct in spirit, just the wrong dialect."""
    for src in _REPAIR_SOURCE_DIALECTS:
        if src == target_dialect:
            continue
        try:
            transpiled = sqlglot.transpile(candidate, read=src, write=target_dialect or None)[0]
            validate_readonly(transpiled, target_dialect)
            logger.info(f"[repair] '{src}'-dialect SQL auto-transpiled to "
                        f"target dialect: {transpiled[:120]}")
            return transpiled
        except Exception:
            continue
    return None


def extract_sql(text: str, dialect: str = "") -> str:
    text = _strip_thinking(text)

    # Collect every span that could plausibly be the query: a fenced code
    # block if present, plus regex-found SELECT/CTE spans as a fallback for
    # when the model narrates instead of fencing its answer. A naive "find
    # the first SELECT/WITH keyword" heuristic (an earlier version of this
    # function) was unreliable: the model's narration itself contains the
    # bare English words "select" ("...we need to select the right
    # columns...") and "with" ("...with columns like Country..."), which
    # matched and produced garbage. So every candidate — fenced or not — is
    # run through the real SQL parser/read-only validator before being
    # trusted; the parser is ground truth here, not a keyword guess.
    candidates = []
    m = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        candidates.append(m.group(1).strip())
    else:
        m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        if m:
            candidates.append(m.group(1).strip())

    candidates += re.findall(r"(?is)\bWITH\s+\w+\s+AS\s*\(.*?(?=;|\n\s*\n|$)", text)
    candidates += re.findall(r"(?is)\bSELECT\b.*?(?=;|\n\s*\n|$)", text)

    seen = set()
    fallback = None
    # Longest first: a CTE match is a superset of the plain-SELECT match
    # for the same query, so trying it first avoids returning a truncated
    # inner fragment when a full CTE is actually present.
    for c in sorted(candidates, key=len, reverse=True):
        c = c.strip().rstrip(";")
        if not c or c in seen:
            continue
        seen.add(c)
        if fallback is None:
            fallback = c  # best-effort answer if nothing validates at all
        try:
            validate_readonly(c, dialect)
            return c  # confirmed: this candidate actually parses as valid read-only SQL
        except ReadOnlyViolation:
            repaired = _try_repair(c, dialect)
            if repaired:
                return repaired
            continue

    return fallback or text.strip()


def extract_json_list(text: str) -> list:
    text = _strip_thinking(text)
    # Try every bracketed span, not just the first — thinking-mode prose can
    # contain stray brackets before the actual answer. [^\[\]]* avoids
    # overmatching across unrelated bracket pairs.
    for m in re.finditer(r"\[[^\[\]]*\]", text, re.DOTALL):
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue
    return []


# langchain-mcp-adapters returns every MCP tool result as a list of content
# blocks, one block per top-level element the tool returned server-side (a
# Python list of 3 dicts becomes 3 separate blocks; a single dict becomes
# one block). Each block's "text" is either a raw string or JSON. These two
# helpers reassemble the original shape our db_mcp_server.py tools return.

def _parse_block(block) -> object:
    text = block.get("text") if isinstance(block, dict) else block
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def unwrap_list(raw) -> list:
    """Use for tools that return a Python list (list_tables, describe_table,
    list_foreign_keys)."""
    if isinstance(raw, list):
        return [_parse_block(b) for b in raw]
    return raw if isinstance(raw, list) else [raw]


def unwrap_single(raw):
    """Use for tools that return a single Python dict (run_query)."""
    if isinstance(raw, list):
        parsed = [_parse_block(b) for b in raw]
        return parsed[0] if len(parsed) == 1 else parsed
    return raw


# ---------------------------------------------------------------------------
# Semantic validation — this is a DIFFERENT check from validate_readonly.
# validate_readonly confirms the SQL is syntactically valid and safe to run.
# It says nothing about whether a JOIN condition actually corresponds to a
# real relationship in the schema. A query can be perfectly valid, safe
# SQL and still be nonsense — e.g. `JOIN Invoice I ON A.ArtistId =
# I.CustomerId`, which ran without error and returned a confidently wrong
# answer. This section catches that class of mistake before execution.
# ---------------------------------------------------------------------------

class SemanticValidationError(ValueError):
    pass


def _resolve_table_aliases(ast: exp.Expression) -> dict:
    """Map every alias (or bare table name, if unaliased) used in the query
    to its real table name, e.g. {'A': 'Artist', 'I': 'Invoice'}."""
    aliases = {}
    for table in ast.find_all(exp.Table):
        name = table.name
        alias = table.alias or name
        aliases[alias] = name
    return aliases


def _fk_pairs(foreign_keys: list) -> set:
    """(table, column, ref_table, ref_column) tuples, both directions, for
    O(1) "is this actually a declared FK relationship" checks."""
    pairs = set()
    for fk in foreign_keys:
        for col, ref_col in zip(fk["columns"], fk["references_columns"]):
            pairs.add((fk["table"], col, fk["references_table"], ref_col))
            pairs.add((fk["references_table"], ref_col, fk["table"], col))
    return pairs


def validate_join_semantics(sql: str, foreign_keys: list, dialect: str = "") -> None:
    """Walk every JOIN's ON condition and confirm each column=column
    equality actually matches a declared foreign-key relationship — not
    just that both tables exist in the schema, but that THESE SPECIFIC
    columns are a real FK pair. This is what catches
    `JOIN Invoice ON Artist.ArtistId = Invoice.CustomerId`: both tables are
    real, both columns are real, but that specific pairing was never a
    declared relationship anywhere in the schema.

    Deliberately conservative: only flags simple `col = col` equalities it
    can fully resolve to real tables via alias tracking. Anything more
    exotic (function calls, literals, multi-condition ON clauses it can't
    cleanly split) is left alone rather than risking a false positive that
    blocks a legitimate query — the goal is to catch the observed failure
    mode, not to become a general SQL correctness prover."""
    try:
        ast = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return  # unparseable SQL is validate_readonly's job to catch, not this

    aliases = _resolve_table_aliases(ast)
    fk_pairs = _fk_pairs(foreign_keys)

    for join in ast.find_all(exp.Join):
        on = join.args.get("on")
        if on is None:
            continue  # e.g. CROSS JOIN with no ON clause — nothing to check
        for eq in on.find_all(exp.EQ):
            left, right = eq.this, eq.expression
            if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
                continue  # not a plain column=column predicate — skip, don't guess
            lt = aliases.get(left.table, left.table)
            rt = aliases.get(right.table, right.table)
            if not lt or not rt:
                continue  # couldn't resolve a table qualifier — skip
            lc, rc = left.name, right.name
            if (lt, lc, rt, rc) not in fk_pairs:
                raise SemanticValidationError(
                    f"JOIN condition '{lt}.{lc} = {rt}.{rc}' does not match any "
                    f"declared foreign-key relationship in the schema. {lt} and "
                    f"{rt} are not related through these columns — check the "
                    f"FK: lines in the schema for the real relationship, and "
                    f"whether an intermediate table is needed to connect them."
                )


def _fk_graph_and_edges(foreign_keys: list):
    """Build (adjacency graph, edge->columns map) from the FK list — the
    same undirected graph structure used for schema bridging, reused here
    to actually rewrite a bad join rather than just deciding which tables'
    schemas to show the model."""
    graph: dict = {}
    edge_cols: dict = {}
    for fk in foreign_keys:
        a, b = fk["table"], fk["references_table"]
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
        for ca, cb in zip(fk["columns"], fk["references_columns"]):
            edge_cols.setdefault(a, {})[b] = (ca, cb)
            edge_cols.setdefault(b, {})[a] = (cb, ca)
    return graph, edge_cols


def _bfs_table_path(start: str, goal: str, graph: dict) -> Optional[list]:
    if start == goal:
        return [start]
    visited = {start}
    queue = [[start]]
    while queue:
        path = queue.pop(0)
        node = path[-1]
        for neighbor in graph.get(node, ()):
            if neighbor == goal:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return None


def repair_join_path(sql: str, foreign_keys: list, dialect: str = "") -> str:
    """When a JOIN condition doesn't match a real FK relationship (the
    class of mistake validate_join_semantics rejects), check whether a
    real multi-hop FK path exists between the two tables via BFS over the
    actual foreign-key graph — the same graph/BFS logic already used for
    schema bridging — and if so, rewrite the query to insert the missing
    intermediate table(s) with correct join conditions, computed entirely
    from the schema's real relationships. Nothing here is specific to any
    particular pair of tables; it's the same generic path-finding applied
    to SQL rewriting instead of just schema selection.

    Deliberately conservative, same as the other repair functions in this
    file: only rewrites a single top-level SELECT with a simple single-
    equality ON clause per join; anything more exotic (UNION, WITH,
    compound ON conditions) is left untouched rather than risking an
    incorrect rewrite — validate_join_semantics will still catch and
    clearly report those cases for the normal retry loop to handle."""
    try:
        ast = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return sql

    if not isinstance(ast, exp.Select):
        return sql  # only handle simple SELECT; don't guess on UNION/WITH

    aliases = _resolve_table_aliases(ast)
    fk_pairs = _fk_pairs(foreign_keys)
    graph, edge_cols = _fk_graph_and_edges(foreign_keys)
    existing_table_names = {t.name for t in ast.find_all(exp.Table)}

    joins = list(ast.args.get("joins") or [])
    inserted_joins = []
    changed = False

    for join in joins:
        on = join.args.get("on")
        if on is None:
            continue
        eqs = list(on.find_all(exp.EQ))
        if len(eqs) != 1:
            continue  # compound ON condition — don't guess which part is the join key
        left, right = eqs[0].this, eqs[0].expression
        if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
            continue

        lt = aliases.get(left.table, left.table)
        rt = aliases.get(right.table, right.table)
        if not lt or not rt or (lt, left.name, rt, right.name) in fk_pairs:
            continue  # already valid, or unresolvable — nothing to repair

        path = _bfs_table_path(lt, rt, graph)
        if not path or len(path) <= 2:
            continue  # no FK path exists at all — repair can't help; let validation reject it

        # Walk the path, inserting a JOIN for each intermediate table that
        # isn't already present in the query, chaining each new join off
        # the previous table in the path using the real FK columns.
        prev_table, prev_qualifier = lt, left.table
        ok = True
        for mid in path[1:-1]:
            if mid in existing_table_names:
                # Already joined elsewhere in the query under its own name
                # — chain through it via its real table name as qualifier.
                prev_table, prev_qualifier = mid, mid
                continue
            if prev_table not in edge_cols or mid not in edge_cols.get(prev_table, {}):
                ok = False
                break
            col_prev, col_mid = edge_cols[prev_table][mid]
            inserted_joins.append(exp.Join(
                this=exp.Table(this=exp.to_identifier(mid)),
                on=exp.EQ(
                    this=exp.column(col_prev, table=prev_qualifier),
                    expression=exp.column(col_mid, table=mid),
                ),
            ))
            existing_table_names.add(mid)
            prev_table, prev_qualifier = mid, mid

        if not ok or prev_table not in edge_cols or rt not in edge_cols.get(prev_table, {}):
            continue  # couldn't fully resolve every hop — leave this join alone

        col_prev2, col_target = edge_cols[prev_table][rt]
        join.set("on", exp.EQ(
            this=exp.column(col_prev2, table=prev_qualifier),
            expression=exp.column(col_target, table=right.table),
        ))
        changed = True
        logger.info(f"[repair] rebuilt join path {lt} -> {rt} via {' -> '.join(path)}")

    if not changed:
        return sql

    # Splice the newly-inserted intermediate joins in before the joins list
    # they support (order among JOIN clauses doesn't affect SQL semantics
    # as long as every referenced table is declared somewhere in the list).
    ast.set("joins", inserted_joins + joins)
    return ast.sql(dialect=dialect or None)


# ---------------------------------------------------------------------------
# Missing-join repair — a FOURTH distinct class of mistake, different from
# repair_join_path above. repair_join_path fixes a WRONG join edge (two
# tables joined via columns that aren't a real FK). This handles a table
# referenced by column (e.g. `COUNT(InvoiceLine.TrackId)`) that was never
# joined into the query AT ALL — no JOIN clause for it anywhere. This is
# what let a query silently change from "tracks sold" (needs InvoiceLine)
# to "tracks in catalog" (doesn't) after a failed attempt: the model
# resolved a "no such column" error by deleting the InvoiceLine reference
# entirely instead of adding the join, and nothing caught that the query's
# meaning had silently changed. Uses the same FK graph/BFS as the other
# repairs — genuinely computed, not hardcoded to any specific table.
# ---------------------------------------------------------------------------

def repair_missing_joins(sql: str, foreign_keys: list, table_schemas: dict, dialect: str = "") -> str:
    """Find column qualifiers that name a REAL table (present in
    table_schemas) but aren't declared anywhere in FROM/JOIN, and insert
    the FK-path-shortest JOIN chain connecting it to a table already in the
    query. Conservative like the other repairs: only acts when a real FK
    path exists; leaves anything ambiguous or disconnected for
    validate_all_qualifiers_resolved to flag explicitly."""
    try:
        ast = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return sql
    if not isinstance(ast, exp.Select):
        return sql

    tables = list(ast.find_all(exp.Table))
    declared = {t.name for t in tables} | {t.alias for t in tables if t.alias}
    present_real_names = {t.name for t in tables}

    cols = list(ast.find_all(exp.Column))
    used_qualifiers = {c.table for c in cols if c.table}

    # A qualifier that names a real table in the schema but isn't declared
    # anywhere in this query — as opposed to a garbage/undefined alias,
    # which repair_undefined_aliases handles separately.
    missing = {q for q in used_qualifiers if q not in declared and q in table_schemas}
    if not missing:
        return sql

    graph, edge_cols = _fk_graph_and_edges(foreign_keys)
    joins = list(ast.args.get("joins") or [])
    inserted = []
    changed = False

    for missing_table in missing:
        best_path = None
        for present in present_real_names:
            path = _bfs_table_path(present, missing_table, graph)
            if path and (best_path is None or len(path) < len(best_path)):
                best_path = path
        if not best_path or len(best_path) < 2:
            continue  # no FK connection to anything already in the query

        anchor_name = best_path[0]
        anchor_qualifier = next(
            (t.alias or t.name for t in tables if t.name == anchor_name), anchor_name
        )
        prev_table, prev_qualifier = anchor_name, anchor_qualifier
        ok = True
        for hop in best_path[1:]:
            # Newly added tables use their real name as qualifier, matching
            # how they were already referenced in SELECT/WHERE/etc.
            hop_qualifier = hop
            if prev_table not in edge_cols or hop not in edge_cols.get(prev_table, {}):
                ok = False
                break
            col_prev, col_hop = edge_cols[prev_table][hop]
            inserted.append(exp.Join(
                this=exp.Table(this=exp.to_identifier(hop)),
                on=exp.EQ(
                    this=exp.column(col_prev, table=prev_qualifier),
                    expression=exp.column(col_hop, table=hop_qualifier),
                ),
            ))
            present_real_names.add(hop)
            prev_table, prev_qualifier = hop, hop_qualifier

        if ok:
            changed = True
            logger.info(f"[repair] added missing JOIN for referenced-but-unjoined "
                        f"table '{missing_table}' via {' -> '.join(best_path)}")

    if not changed:
        return sql

    ast.set("joins", joins + inserted)
    return ast.sql(dialect=dialect or None)


# ---------------------------------------------------------------------------
# Tie-aware ranking rewrite — addresses "the agent assumes ORDER BY + LIMIT
# = correct top-k". A plain LIMIT N is an arbitrary cutoff: if rows N and
# N+1 are tied on the ranking metric, LIMIT N keeps one and drops the other
# for no principled reason. The fix is a generic structural SQL rewrite —
# not specific to any table or query — using RANK() OVER (...) instead of
# a raw LIMIT, so ties at the boundary are included rather than silently
# broken. This is standard SQL ranking practice, not a heuristic.
# ---------------------------------------------------------------------------

def rewrite_top_n_with_ties(sql: str, dialect: str = "") -> str:
    """Rewrite `... ORDER BY <expr> LIMIT N` into a query using
    `RANK() OVER (ORDER BY <expr>) AS __rnk` filtered to `__rnk <= N`, so
    ties at the cutoff are included instead of arbitrarily cut.

    Conservative like the other repairs here: only rewrites when the
    ORDER BY key is a single, simple identifier (a column or an alias
    defined in the SELECT list) that the outer window function can safely
    reference from the wrapped subquery. If the ORDER BY uses a raw
    unaliased expression (e.g. `ORDER BY SUM(x)` with no output alias for
    that value), rewriting could produce invalid or incorrect SQL, so this
    bails out and leaves the original LIMIT-based query untouched rather
    than risk it — same "don't guess" philosophy as the other repairs."""
    try:
        ast = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return sql
    if not isinstance(ast, exp.Select):
        return sql

    limit_node = ast.args.get("limit")
    order_node = ast.args.get("order")
    if limit_node is None or order_node is None:
        return sql

    order_expressions = order_node.expressions
    if len(order_expressions) != 1:
        return sql  # multi-key ORDER BY — don't guess how ties interact across keys

    try:
        n = int(str(limit_node.expression.this))
    except (AttributeError, ValueError):
        return sql  # non-literal LIMIT (e.g. a bound parameter) — leave alone

    order_col = order_expressions[0].this
    if not isinstance(order_col, exp.Column):
        return sql  # not a simple column/alias reference — bail out, don't guess

    order_key = order_col.name

    def _output_name(e):
        if isinstance(e, exp.Alias):
            return e.alias
        if isinstance(e, exp.Column):
            return e.name  # e itself IS the column for a bare "SELECT revenue"
        return None

    select_aliases = {_output_name(e) for e in ast.expressions}
    select_aliases.discard(None)
    if order_key not in select_aliases:
        return sql  # ORDER BY key isn't exposed as an output column of this
        # query — the outer window function couldn't reference it after
        # wrapping, so don't attempt the rewrite

    order_by_sql = order_node.sql(dialect=dialect or None)
    order_by_expr = re.sub(r"(?i)^order by\s+", "", order_by_sql)

    inner = ast.copy()
    inner.set("limit", None)
    inner.set("order", None)
    inner_sql = inner.sql(dialect=dialect or None)

    wrapped = (
        f"SELECT * FROM ("
        f"SELECT sub.*, RANK() OVER (ORDER BY {order_by_expr}) AS __rnk "
        f"FROM ({inner_sql}) AS sub"
        f") AS ranked WHERE __rnk <= {n} ORDER BY __rnk"
    )

    try:
        reparsed = sqlglot.parse_one(wrapped, read=dialect or None)
        return reparsed.sql(dialect=dialect or None)
    except Exception:
        return sql  # rewrite didn't produce valid SQL — don't risk it


def validate_all_qualifiers_resolved(sql: str, dialect: str = "") -> None:
    """Stricter semantic check, run AFTER the repair functions above.
    validate_join_semantics only checks that EXISTING JOIN conditions in
    the query are real FK relationships — it says nothing about whether a
    referenced table was joined at all. This closes that gap: every column
    qualifier used anywhere in the query (SELECT, WHERE, GROUP BY, ORDER
    BY, inside aggregate functions, ...) must resolve to a table or alias
    actually declared in FROM/JOIN. Anything still unresolved at this point
    is genuinely unfixable automatically (repair_missing_joins already had
    its chance) and needs a real LLM retry with a specific, actionable
    error rather than a cryptic database-level failure."""
    try:
        ast = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return  # unparseable SQL is validate_readonly's job, not this

    if not isinstance(ast, exp.Select):
        return  # only enforced on simple SELECT, consistent with the repairs above

    tables = list(ast.find_all(exp.Table))
    declared = {t.name for t in tables} | {t.alias for t in tables if t.alias}

    cols = list(ast.find_all(exp.Column))
    used_qualifiers = {c.table for c in cols if c.table}
    unresolved = used_qualifiers - declared

    if unresolved:
        raise SemanticValidationError(
            f"Column qualifier(s) {sorted(unresolved)} are referenced in the "
            f"query but no matching table appears anywhere in FROM/JOIN. ADD "
            f"a JOIN clause for the missing table(s) — do NOT remove the "
            f"column reference, and do NOT change what the query measures "
            f"just to make the error go away."
        )


# ---------------------------------------------------------------------------
# Grouping-intent validation — a FIFTH distinct class of mistake, and a
# different KIND of bug from everything above: those all caught SQL that
# was structurally broken (wrong join, missing join, undeclared alias).
# This one is structurally FINE — it runs, it returns a plausible-looking
# number — but the number answers a different question than the one asked.
# A classic text-to-SQL failure: "average invoice total per customer"
# matches "average" + "invoice total" but drops "per customer", producing
# a single ungrouped AVG() across every invoice instead of one average per
# customer. The answer_verification step can't catch this — the number IS
# real, pulled straight from a real query result. Only checking the
# QUESTION's language against the SQL's STRUCTURE can catch it.
# ---------------------------------------------------------------------------

_GROUPING_INTENT_PATTERNS = [
    re.compile(r"\bper\s+(\w+)", re.IGNORECASE),
    re.compile(r"\bfor each\s+(\w+)", re.IGNORECASE),
    re.compile(r"\beach\s+(\w+)", re.IGNORECASE),
    re.compile(r"\bby\s+(\w+)\s*(?:,|and\b|$)", re.IGNORECASE),
]


def _grouping_intent_terms(question: str) -> list:
    """Extract the noun following a grouping-intent phrase ('per customer'
    -> 'customer'), for both detection and building an actionable error
    message. Not schema-aware by design — matching is purely on the
    question's language; the caller decides what to do with the term."""
    terms = []
    for pattern in _GROUPING_INTENT_PATTERNS:
        for m in pattern.finditer(question):
            terms.append(m.group(1))
    return terms


def validate_grouping_intent(question: str, sql: str, dialect: str = "") -> None:
    """If the question uses 'per X' / 'for each X' / 'by X' language
    (implying one result row per X), but the query aggregates WITHOUT a
    GROUP BY at all, that's almost certainly wrong — an ungrouped aggregate
    collapses everything into a single row, contradicting a per-entity
    breakdown. Doesn't attempt to guess or auto-repair which column to
    group by (too easy to guess wrong and silently produce a different
    mistake); raises with the specific term detected so the retry prompt
    can point the model at exactly what was missed."""
    terms = _grouping_intent_terms(question)
    if not terms:
        return

    try:
        ast = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return  # unparseable SQL is validate_readonly's job, not this
    if not isinstance(ast, exp.Select):
        return

    has_aggregate = any(
        isinstance(n, (exp.Sum, exp.Avg, exp.Count, exp.Min, exp.Max))
        for n in ast.walk()
    )
    has_group_by = ast.args.get("group") is not None

    if has_aggregate and not has_group_by:
        raise SemanticValidationError(
            f"The question says \"{terms[0]}\" — implying one result per "
            f"{terms[0]}, e.g. \"per customer\" means one row per customer "
            f"— but the query aggregates with NO GROUP BY at all, which "
            f"collapses everything into a single overall value instead. "
            f"Add a GROUP BY on the column that corresponds to \"{terms[0]}\"."
        )


# ---------------------------------------------------------------------------
# Answer verification — a best-effort check that the FINAL ANSWER TEXT
# actually reflects the query results, rather than the model editorializing,
# "correcting" a value it thought looked like a typo, or otherwise drifting
# from the data. This can't catch every kind of hallucination, but it
# directly targets an observed failure: the model appending fabricated
# commentary like "(likely a typo)" instead of trusting the DB result.
# ---------------------------------------------------------------------------

def _strip_list_markers(text: str) -> str:
    """Remove leading list markers ('1. ', '2) ', etc.) so they aren't
    mistaken for data values during number extraction — done via actual
    list-marker pattern matching, not a blanket 'small numbers are probably
    an index' guess (which turned out to let real hallucinated small counts,
    e.g. a fabricated '15', slip through unflagged)."""
    return re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)


def _extract_numbers(text: str) -> set:
    text = _strip_list_markers(text)
    # Negative lookbehind/lookahead for a letter excludes digits embedded in
    # an alphanumeric token (e.g. the "2" in "U2", a real band name) from
    # being treated as a data figure to verify — a real observed false
    # positive that flagged a correct answer as unverified.
    # rstrip('.') handles any remaining trailing punctuation — a real
    # decimal like "49.62" is unaffected since \d* already consumed the
    # digits after the dot, so there's no trailing "." left to strip.
    return {n.rstrip(".") for n in re.findall(r"(?<![A-Za-z])-?\d+\.?\d*(?![A-Za-z])", text)}


def _extract_numbers_from_rows(result: dict) -> set:
    nums = set()
    for row in result.get("rows", []):
        for val in row:
            if isinstance(val, (int, float)):
                nums.add(str(val))
            elif isinstance(val, str):
                # MCP's JSON serialization turns non-JSON-native types (e.g.
                # Postgres Decimal/NUMERIC results) into strings — a real
                # observed bug: "59.88" arrived as a string, not a float,
                # and was silently never matched against the answer text.
                try:
                    float(val)
                    nums.add(val)
                except ValueError:
                    pass
    # The row count itself is a legitimate, commonly-stated fact ("there
    # are N results") even when it isn't a literal cell value anywhere.
    nums.add(str(len(result.get("rows", []))))
    return nums


def verify_answer(answer: str, result: dict) -> list:
    """Returns the list of numeric figures in `answer` that could NOT be
    traced back to any value actually present in the query results (or the
    row count). Empty list = every number in the answer is grounded in
    real data."""
    answer_nums = _extract_numbers(answer)
    row_nums = _extract_numbers_from_rows(result)
    return sorted(answer_nums - row_nums)


# ---------------------------------------------------------------------------
# Alias repair — a THIRD distinct class of mistake from dialect syntax
# (extract_sql's _try_repair) and join semantics (validate_join_semantics).
# This one is subtler: the query parses fine and every JOIN is a real FK
# relationship, but SELECT/GROUP BY/ORDER BY reference a table alias (e.g.
# "A.Name") that was NEVER declared in FROM/JOIN ("JOIN Artist ON ..." with
# no "AS A"). sqlglot's parser doesn't do identifier resolution, so this
# passes both earlier validators cleanly and only fails at the database —
# and in practice, error-message retries didn't fix it (the model repeated
# the identical mistake three times). This repairs it deterministically
# instead of hoping the model self-corrects.
# ---------------------------------------------------------------------------

def repair_undefined_aliases(sql: str, dialect: str, table_schemas: dict) -> str:
    """If a column qualifier (e.g. the "A" in "A.Name") isn't declared as
    any real table name or alias in FROM/JOIN, try to identify which
    un-aliased table it was meant to refer to and attach that alias to it.
    Disambiguation is two-stage: first by actual column membership (which
    candidate table's real schema contains every column referenced under
    that qualifier — authoritative, since we already fetched real schemas),
    then, if still ambiguous, by name-prefix match. Only applies the fix
    when exactly one candidate remains; never guesses under real ambiguity."""
    try:
        ast = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return sql

    tables = list(ast.find_all(exp.Table))
    declared = {t.name for t in tables} | {t.alias for t in tables if t.alias}

    cols = list(ast.find_all(exp.Column))
    used_qualifiers = {c.table for c in cols if c.table}
    undefined = used_qualifiers - declared
    if not undefined:
        return sql

    changed = False
    for qualifier in undefined:
        referenced_cols = {c.name for c in cols if c.table == qualifier}
        candidates = [t for t in tables if not t.alias]

        matches = [
            t for t in candidates
            if referenced_cols.issubset({c["name"] for c in table_schemas.get(t.name, [])})
        ]
        if len(matches) > 1:
            narrowed = [t for t in matches if t.name.lower().startswith(qualifier.lower())]
            if len(narrowed) == 1:
                matches = narrowed

        if len(matches) == 1:
            table_node = matches[0]
            original_name = table_node.name
            table_node.set("alias", exp.TableAlias(this=exp.to_identifier(qualifier)))
            changed = True
            # Once a table has an alias, some engines (SQLite included)
            # reject any further reference to it by its original bare
            # name — the whole query has to consistently use the alias.
            # Rewrite every other column reference that used the original
            # table name so the repaired query is actually internally
            # consistent, not just individually parseable.
            for c in cols:
                if c.table == original_name:
                    c.set("table", exp.to_identifier(qualifier))
            logger.info(f"[repair] undeclared alias '{qualifier}' resolved to "
                        f"table '{original_name}' and repaired")

    return ast.sql(dialect=dialect or None) if changed else sql


# ---------------------------------------------------------------------------
# The control loop itself
# ---------------------------------------------------------------------------

class SQLAgent:
    """Explicit, non-agentic text-to-SQL pipeline. See module docstring."""

    def __init__(self, db_url: str, llm, dialect: str,
                 max_retries: int = DEFAULT_MAX_RETRIES,
                 cache_ttl: int = CACHE_TTL_SECONDS,
                 use_cache: bool = True):
        self.db_url = db_url
        self.llm = llm
        self.dialect = dialect
        self.max_retries = max_retries
        self.cache = SchemaCache(ttl=cache_ttl) if use_cache else None
        self.metrics = Metrics()
        self._tools = {}
        self._session_cm = None

    async def _connect(self):
        """Hold ONE persistent MCP session open for the whole run(), instead
        of letting langchain-mcp-adapters open a brand-new subprocess + MCP
        handshake for every individual tool call. client.get_tools() docs
        say plainly: "A new session will be created for each tool call" —
        that's what produced the repeated ListToolsRequest/CallToolRequest
        pairs in the logs and most of the wall-clock overhead. We instead
        open client.session(...) once and reuse it for every tool call in
        this run."""
        server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db_mcp_server.py")
        client = MultiServerMCPClient({
            "db": {
                "command": sys.executable,
                "args": [server_script, "--db-url", self.db_url],
                "transport": "stdio",
            }
        })
        self._session_cm = client.session("db")
        session = await self._session_cm.__aenter__()
        tools = await load_mcp_tools(session)
        self._tools = {t.name: t for t in tools}

    async def _disconnect(self):
        if self._session_cm is not None:
            await self._session_cm.__aexit__(None, None, None)
            self._session_cm = None

    # Which unwrap helper each MCP tool's result needs (see unwrap_list /
    # unwrap_single docstrings above).
    _RESULT_SHAPE = {
        "list_tables": "list",
        "describe_table": "list",
        "list_foreign_keys": "list",
        "run_query": "single",
    }

    async def _call_tool(self, name: str, **kwargs):
        t0 = time.time()
        raw = await self._tools[name].ainvoke(kwargs)

        # langchain-mcp-adapters returns MCP execution errors
        # (CallToolResult(isError=True)) as ordinary text content by
        # default, rather than raising — that's meant for an agentic loop
        # where the model reads the error and self-corrects. We're not
        # using that loop, so surface it as a real exception ourselves,
        # which is what lets our explicit retry loop (step 6) catch it.
        if isinstance(raw, list):
            for block in raw:
                text = block.get("text") if isinstance(block, dict) else block
                if isinstance(text, str) and text.startswith("Error executing tool"):
                    self.metrics.tool_calls += 1
                    self.metrics.record(f"tool:{name}", time.time() - t0)
                    raise RuntimeError(text)

        shape = self._RESULT_SHAPE.get(name, "single")
        result = unwrap_list(raw) if shape == "list" else unwrap_single(raw)
        self.metrics.tool_calls += 1
        self.metrics.record(f"tool:{name}", time.time() - t0)
        logger.info(f"[tool] {name}({kwargs}) -> {str(result)[:200]}")
        return result

    async def _call_llm(self, prompt: str, step: str) -> str:
        t0 = time.time()
        result = await self.llm.ainvoke(prompt)
        self.metrics.llm_calls += 1
        self.metrics.record(f"llm:{step}", time.time() - t0)
        text = result.content if hasattr(result, "content") else str(result)
        logger.info(f"[llm:{step}] {text[:200]}")
        return text

    # ---- Step 1a: get table NAMES only (cheap, cached) ----
    async def get_tables(self) -> list:
        if self.cache:
            cached = self.cache.get_tables(self.db_url)
            if cached is not None:
                self.metrics.cache_hits += 1
                logger.info("[cache] table list hit — skipped list_tables")
                return cached
            self.metrics.cache_misses += 1

        tables = await self._call_tool("list_tables")
        if self.cache:
            self.cache.set_tables(self.db_url, tables)
        return tables

    # ---- Step 1b: get schema for ONE table only (cheap, cached per-table) ----
    async def get_table_schema(self, table: str) -> list:
        if self.cache:
            cached = self.cache.get_table_schema(self.db_url, table)
            if cached is not None:
                self.metrics.cache_hits += 1
                logger.info(f"[cache] schema hit for '{table}' — skipped describe_table")
                return cached
            self.metrics.cache_misses += 1

        schema = await self._call_tool("describe_table", table_name=table)
        if self.cache:
            self.cache.set_table_schema(self.db_url, table, schema)
        return schema

    # ---- Step 1c: get ALL foreign keys once, cached (cheap single call
    # regardless of table count — filtered down to relevant pairs later) ----
    async def get_foreign_keys(self) -> list:
        if self.cache:
            cached = self.cache.get_foreign_keys(self.db_url)
            if cached is not None:
                self.metrics.cache_hits += 1
                logger.info("[cache] foreign key hit — skipped list_foreign_keys")
                return cached
            self.metrics.cache_misses += 1

        fks = await self._call_tool("list_foreign_keys")
        if self.cache:
            self.cache.set_foreign_keys(self.db_url, fks)
        return fks

    # ---- Step 2: pick relevant tables (names only — cheap) ----
    async def pick_relevant_tables(self, question: str, all_tables: list) -> list:
        if len(all_tables) <= 3:
            return all_tables  # not worth an LLM call for a tiny schema

        prompt = (
            f"Database tables: {all_tables}\n\n"
            f"Question: {question}\n\n"
            "Which tables are needed to answer this? Include any table "
            "needed to JOIN to the answer, even if not named directly. "
            "Respond with ONLY a JSON array of table names, nothing else."
        )
        text = await self._call_llm(prompt, "pick_tables")
        picked = [t for t in extract_json_list(text) if t in all_tables]
        if not picked:
            logger.warning(
                "[pick_tables] Could not parse a usable table list from the model's "
                "response — falling back to ALL tables (safe, but defeats schema "
                "filtering for this call). Response was: " + text[:150]
            )
        return picked or all_tables  # fail open, not closed

    # ---- Step 3: generate SQL ----
    async def generate_sql(self, question: str, schema_context: str, error_context: str = "") -> str:
        retry_note = (
            f"\n\nThe previous attempt failed — fix this exact error:\n{error_context}\n"
            "IMPORTANT: fix the error by correcting the query's STRUCTURE "
            "(e.g. add the missing JOIN clause for a table that error "
            "mentions). Do NOT fix it by deleting or simplifying away the "
            "part of the query that caused the error — that changes what "
            "the query measures and produces a wrong answer that runs "
            "without error. If a column from table X caused the error, the "
            "fix is almost always \"join table X correctly\", not \"stop "
            "using table X\"."
            if error_context else ""
        )
        prompt = (
            f"Write a single read-only {self.dialect} SELECT query.\n\n"
            f"Schema (only the relevant tables):\n{schema_context}\n\n"
            f"Question: {question}{retry_note}\n\n"
            "Rules:\n"
            "- Output exactly one SELECT statement (JOINs/GROUP BY/window "
            "functions/CTEs are fine) in a ```sql code block, nothing else.\n"
            "- Use only the columns and FK join keys shown above.\n"
            "- Only JOIN two tables if an explicit \"FK:\" line above connects "
            "them (directly, or through another table in the schema above). "
            "Never invent a join between columns that aren't listed as a FK "
            "relationship, even if the column names look similar.\n"
            "- Match the exact metric the question asks for: \"how many X\" "
            "or \"most sold\" means COUNT or SUM of a quantity/count column, "
            "NOT a price/total/revenue column, unless the question actually "
            "asks about money.\n"
            f"- Use {self.dialect} syntax specifically, NOT SQL Server/T-SQL "
            f"syntax — e.g. use LIMIT n at the end of the query to limit "
            f"rows, never `SELECT TOP n`, which {self.dialect} does not "
            "support.\n"
            "- Unless asked for a specific row count, LIMIT 20 and ORDER BY "
            "the most relevant metric.\n\n"
            "Respond with the ```sql code block ONLY. No explanation, no "
            "reasoning, no restating the question — just the fenced query."
        )
        text = await self._call_llm(prompt, "generate_sql")
        return extract_sql(text, _sqlglot_dialect(self.dialect))

    # ---- Step 4: validate — local AST check, no LLM/network call ----
    def validate_sql(self, sql: str):
        validate_readonly(sql, _sqlglot_dialect(self.dialect))

    # ---- Step 5: execute ----
    async def execute_sql(self, sql: str) -> dict:
        return await self._call_tool("run_query", query=sql)

    # ---- Step 7: format final answer ----
    async def format_answer(self, question: str, sql: str, result: dict, correction_note: str = "") -> str:
        prompt = (
            f"Question: {question}\n\n"
            f"SQL used: {sql}\n\n"
            f"Result columns: {result.get('columns')}\n"
            f"Result rows: {result.get('rows')}\n\n"
            "Answer the question directly and concisely, citing the key "
            "numbers. Report the data EXACTLY as returned — do not correct, "
            "second-guess, or annotate values you think look unusual or "
            "like typos (e.g. do not add notes like \"(likely a typo)\"). "
            "Real data often contains legitimate values that look "
            "unexpected; trust the query result verbatim rather than "
            "editorializing about it. Don't just repeat the raw table."
            f"{correction_note}"
        )
        return await self._call_llm(prompt, "format_answer")

    def _build_schema_context(self, table_schemas: dict, foreign_keys: list, relevant: list) -> str:
        """Schema filtering: only the picked tables' columns, only FKs that
        connect two picked tables — this is what keeps the prompt small.
        `table_schemas` here only ever contains entries for `relevant`
        tables in the first place (see run()), so this is filtering FKs,
        not re-filtering columns."""
        lines = []
        for t in relevant:
            cols = ", ".join(f"{c['name']} {c['type']}" for c in table_schemas[t])
            lines.append(f"{t}({cols})")
        for fk in foreign_keys:
            if fk["table"] in relevant and fk["references_table"] in relevant:
                lines.append(
                    f"FK: {fk['table']}.{fk['columns']} -> "
                    f"{fk['references_table']}.{fk['references_columns']}"
                )
        return "\n".join(lines)

    @staticmethod
    def _build_fk_graph(foreign_keys: list) -> dict:
        graph: dict = {}
        for fk in foreign_keys:
            a, b = fk["table"], fk["references_table"]
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)
        return graph

    @classmethod
    def _bridge_tables(cls, seed_tables: list, foreign_keys: list) -> list:
        """Ensure every pair of seed tables ends up connected by an actual
        FK-joinable path, adding any intermediate "bridge" tables via BFS
        shortest path over the foreign-key graph.

        This is what prevents a weaker model from hallucinating a direct
        join between two tables that aren't really related — a real,
        observed failure: joining Artist directly to Invoice ("ON
        A.ArtistId = I.CustomerId", which is nonsense) when the real
        relationship is Artist -> Album -> Track -> InvoiceLine -> Invoice.
        pick_relevant_tables only picked the two tables the question named
        by keyword, so the model was never even shown the bridge tables and
        had no correct join available to write. The fix is computed here in
        code — the actual join path is deterministic graph structure, not
        something that should depend on the LLM correctly guessing it."""
        if len(seed_tables) <= 1:
            return list(seed_tables)

        graph = cls._build_fk_graph(foreign_keys)
        result = set(seed_tables)

        def bfs_path(start, goal):
            if start == goal:
                return [start]
            visited = {start}
            queue = [[start]]
            while queue:
                path = queue.pop(0)
                node = path[-1]
                for neighbor in graph.get(node, ()):
                    if neighbor == goal:
                        return path + [neighbor]
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(path + [neighbor])
            return None  # no FK path connects these two tables at all

        seeds = list(seed_tables)
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                path = bfs_path(seeds[i], seeds[j])
                if path:
                    result.update(path)

        return list(result)

    async def run(self, question: str):
        await self._connect()
        try:
            # Step 1: table NAMES only — cheap, cached, no per-table cost yet
            tables = await self.get_tables()

            # Step 2: pick relevant tables from names alone (no schema fetched yet)
            seed_tables = await self.pick_relevant_tables(question, tables)

            # Fetch the FK graph (single cheap cached call, whole-DB) BEFORE
            # deciding the final table set, so we can bridge any gap between
            # the seed tables with the tables that actually connect them.
            foreign_keys = await self.get_foreign_keys()
            relevant = self._bridge_tables(seed_tables, foreign_keys)
            if set(relevant) != set(seed_tables):
                bridged_in = set(relevant) - set(seed_tables)
                logger.info(f"[schema] bridging gap between picked tables — "
                            f"adding connecting table(s) via FK graph: {bridged_in}")

            logger.info(f"[schema] {len(tables)} tables total -> fetching schema for "
                        f"{len(relevant)} relevant: {relevant}")

            # Only NOW fetch describe_table, and only for the relevant subset —
            # this is the fix for over-fetching the whole database's schema
            # up front regardless of what the question actually needs.
            table_schemas = {t: await self.get_table_schema(t) for t in relevant}
            schema_context = self._build_schema_context(table_schemas, foreign_keys, relevant)

            error_context = ""
            sql = ""
            result = None
            last_error = None

            # Step 6: explicit, bounded retry loop
            for attempt in range(self.max_retries + 1):
                sql = await self.generate_sql(question, schema_context, error_context)
                # Deterministic alias repair — fixes a real, observed
                # failure (undeclared "A."/"T." aliases) that error-message
                # retries alone did not fix even after 3 identical attempts.
                sql = repair_undefined_aliases(sql, _sqlglot_dialect(self.dialect), table_schemas)
                # Deterministic missing-join repair — fixes a table
                # referenced by column (e.g. an aggregate over
                # InvoiceLine.TrackId) that was never joined into the query
                # at all. Runs BEFORE join-semantics validation below so
                # that by the time that check runs, every referenced table
                # is actually present.
                sql = repair_missing_joins(sql, foreign_keys, table_schemas, _sqlglot_dialect(self.dialect))
                try:
                    self.validate_sql(sql)
                    # Semantic/FK validation — a DIFFERENT check from
                    # validate_sql above. Syntax validation confirms the SQL
                    # is well-formed and safe; this confirms every JOIN
                    # condition corresponds to a REAL declared relationship,
                    # not just that both tables happen to exist. This is
                    # what catches a hallucinated join like
                    # `Artist.ArtistId = Invoice.CustomerId` — valid SQL,
                    # real tables, real columns, but never a real
                    # relationship anywhere in the schema.
                    try:
                        validate_join_semantics(sql, foreign_keys, _sqlglot_dialect(self.dialect))
                    except SemanticValidationError as e:
                        self.metrics.semantic_rejections += 1
                        # Before giving up on this attempt (and burning a
                        # full LLM retry), check whether a real FK path
                        # exists between the two tables via BFS over the FK
                        # graph — computed generically, not hardcoded to
                        # any specific pair — and if so, deterministically
                        # rewrite the query to route through the correct
                        # intermediate table(s) instead of hoping the model
                        # figures out a multi-hop join itself.
                        repaired = repair_join_path(sql, foreign_keys, _sqlglot_dialect(self.dialect))
                        if repaired == sql:
                            raise  # nothing we could deterministically fix
                        sql = repaired
                        self.validate_sql(sql)
                        validate_join_semantics(sql, foreign_keys, _sqlglot_dialect(self.dialect))
                    # Stricter check, run last: catches anything STILL
                    # referencing a table that isn't actually joined —
                    # repair_missing_joins already had its chance above, so
                    # anything flagged here is genuinely unfixable
                    # automatically (e.g. no FK path connects it to
                    # anything in the query) and needs a real LLM retry
                    # with a specific, actionable error.
                    validate_all_qualifiers_resolved(sql, _sqlglot_dialect(self.dialect))
                    # Grouping-intent check — a DIFFERENT kind of bug from
                    # everything above: this SQL is structurally fine, but
                    # answers a different question than what was asked
                    # (e.g. "average X per customer" without a GROUP BY).
                    # No auto-repair here — guessing the wrong grouping
                    # column would just trade one silent mistake for
                    # another — but a specific, actionable error forces a
                    # real retry instead of shipping a plausible wrong number.
                    validate_grouping_intent(question, sql, _sqlglot_dialect(self.dialect))
                    # Tie-aware ranking rewrite — applied LAST, only after
                    # every validation above has confirmed the flat query is
                    # correct. This wraps the query in a subquery (RANK()
                    # OVER (...) WHERE rnk <= N), and none of the validators
                    # above understand derived-table aliases — validating
                    # the simple pre-rewrite query first and treating this
                    # rewrite as a trusted final mechanical transform avoids
                    # that whole class of false rejection.
                    sql = rewrite_top_n_with_ties(sql, _sqlglot_dialect(self.dialect))
                    result = await self.execute_sql(sql)
                    last_error = None
                    break
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"[attempt {attempt + 1}/{self.max_retries + 1} failed] {last_error}")
                    error_context = f"Query: {sql}\nError: {last_error}"
                    if attempt < self.max_retries:
                        self.metrics.retries += 1

            if last_error:
                raise RuntimeError(
                    f"Failed after {self.max_retries + 1} attempt(s). Last error: {last_error}"
                )

            answer = await self.format_answer(question, sql, result)

            # Step 8 (new): answer verification. A syntactically valid,
            # semantically valid, correctly executed query can still get a
            # bad final answer if the LLM editorializes when writing it up
            # — an observed real failure was the model appending a
            # fabricated "(likely a typo)" note instead of trusting the
            # DB's actual value. Check that every number in the answer
            # traces back to the real result set; if not, give the model
            # one corrective retry with the mismatch called out explicitly.
            unverified = verify_answer(answer, result)
            self.metrics.answer_verified = not unverified
            if unverified:
                logger.warning(f"[verify] Answer contains number(s) not found in the "
                                f"query results: {unverified} — retrying format_answer once.")
                self.metrics.verification_retries += 1
                correction_note = (
                    "\n\nIMPORTANT: Your previous answer included the number(s) "
                    f"{unverified}, which do NOT appear anywhere in the result rows "
                    "above. Use ONLY the exact values shown in the rows — do not "
                    "alter, round unexpectedly, invent, or 'correct' any figure."
                )
                answer = await self.format_answer(question, sql, result, correction_note)
                still_unverified = verify_answer(answer, result)
                self.metrics.answer_verified = not still_unverified
                if still_unverified:
                    logger.warning(f"[verify] Still unverified after retry: {still_unverified}")
                    answer += (
                        "\n\n⚠️ Note: this answer could not be fully verified against "
                        "the raw query results — please double-check the figures above."
                    )

            return answer, sql, self.metrics
        finally:
            await self._disconnect()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Explicit-control-loop text-to-SQL agent (no create_agent()).",
    )
    parser.add_argument("question", type=str)
    parser.add_argument("--db-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "ollama"),
                         choices=["ollama", "anthropic"])
    parser.add_argument("--model", default=os.getenv("LLM_MODEL"))
    parser.add_argument(
        "--think", action="store_true",
        help="Enable Qwen3's extended thinking mode (Ollama only). Off by "
             "default — it adds significant latency for little benefit on "
             "straightforward SQL generation.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Cap on generated tokens per LLM call (Ollama's num_predict).",
    )
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--cache-ttl", type=int, default=CACHE_TTL_SECONDS)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Show step-by-step logs")
    parser.add_argument("--metrics", action="store_true", help="Show call/timing metrics after answering")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)

    if not args.db_url:
        console.print("[bold red]Error:[/bold red] --db-url or DATABASE_URL env var is required")
        sys.exit(1)

    dialect = "PostgreSQL" if args.db_url.startswith("postgresql") else \
              "MySQL" if args.db_url.startswith("mysql") else \
              "SQLite" if args.db_url.startswith("sqlite") else "SQL"

    console.print(Panel(f"[bold cyan]Question:[/bold cyan] {args.question}", border_style="cyan"))

    try:
        llm = build_llm(args.provider, args.model, reasoning=args.think, max_tokens=args.max_tokens)
        agent = SQLAgent(
            args.db_url, llm, dialect,
            max_retries=args.max_retries,
            cache_ttl=args.cache_ttl,
            use_cache=not args.no_cache,
        )
        answer, sql, metrics = asyncio.run(agent.run(args.question))

        console.print(Panel(f"[bold green]Answer:[/bold green]\n\n{answer}", border_style="green"))
        console.print(Panel(f"[dim]{sql}[/dim]", title="SQL used", border_style="dim"))
        if args.metrics:
            console.print(Panel(metrics.summary(), title="Metrics", border_style="blue"))
    except Exception as e:
        console.print(Panel(f"[bold red]Error:[/bold red]\n\n{str(e)}", border_style="red"))
        sys.exit(1)


if __name__ == "__main__":
    main()