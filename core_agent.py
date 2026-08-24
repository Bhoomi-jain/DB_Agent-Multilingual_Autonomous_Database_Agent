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
from sql_semantics import (
    AGG_TYPES as _SEM_AGG_TYPES,
    classify_grains,
    infer_question_grain,
    build_fk_maps,
    descendants,
    root_and_detail_side,
    resolve_measure_source,
    semantic_diff,
    summarize_diff,
)
import schema_profile as schema_profile_mod
from schema_profile import (
    build_profile,
    validate_execution_result,
    ExecutionSanityError,
)

load_dotenv()
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_path=False, show_time=True)],
)
# ---------------------------------------------------------------------------
# Failure classification (observability layer) — every rejection, repair,
# and execution error gets ONE category so runs can be quantified:
# benchmark.py reports exact-match/exec accuracy, retry rate, hallucination
# rate and this breakdown side by side.
# ---------------------------------------------------------------------------

from collections import Counter as _Counter

class FailureClass:
    SYNTAX_ERROR = "SYNTAX_ERROR"
    JOIN_ERROR = "JOIN_ERROR"
    COLUMN_HALLUCINATION = "COLUMN_HALLUCINATION"
    GRAIN_ERROR = "GRAIN_ERROR"
    AGGREGATION_ERROR = "AGGREGATION_ERROR"
    MEASURE_SOURCE_ERROR = "MEASURE_SOURCE_ERROR"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    VERIFICATION_ERROR = "VERIFICATION_ERROR"
    NONE = "NONE"


def classify_error(message: str, default: str = FailureClass.NONE) -> str:
    m = (message or "")
    checks = [
        ("Grain mismatch", FailureClass.GRAIN_ERROR),
        ("does not match any declared foreign-key", FailureClass.JOIN_ERROR),
        ("does not exist on table", FailureClass.COLUMN_HALLUCINATION),
        ("is not provided by any table", FailureClass.COLUMN_HALLUCINATION),
        ("AMBIGUOUS", FailureClass.JOIN_ERROR),
        ("measures the wrong thing", FailureClass.AGGREGATION_ERROR),
        ("monetary AMOUNT", FailureClass.AGGREGATION_ERROR),
        ("HOW MANY", FailureClass.AGGREGATION_ERROR),
        ("NO aggregation", FailureClass.AGGREGATION_ERROR),
        ("requires", FailureClass.AGGREGATION_ERROR),
        ("Only SELECT statements", FailureClass.SYNTAX_ERROR),
        ("Could not parse SQL", FailureClass.SYNTAX_ERROR),
        ("Impossible", FailureClass.EXECUTION_ERROR),
        ("Error executing tool", FailureClass.EXECUTION_ERROR),
        ("could not be fully verified", FailureClass.VERIFICATION_ERROR),
        ("not found in the query results", FailureClass.VERIFICATION_ERROR),
    ]
    for needle, cls in checks:
        if needle in m:
            return cls
    return default


logger = logging.getLogger("core_agent")
logger.setLevel(logging.WARNING)  # quiet by default; --verbose turns this up

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".schema_cache.json")
CACHE_TTL_SECONDS = 300
DEFAULT_MAX_RETRIES = 2
CACHE_SCHEMA_VERSION = 4  # v4: adds statistical 'profile' (schema_profile)


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
    attempt_diffs: list = field(default_factory=list)      # semantic SQL diffs per retry
    optimization_notes: list = field(default_factory=list) # optimizer/learning annotations
    attempts: int = 0                                       # generate attempts used
    repairs_applied: int = 0                                # deterministic repairs applied (budgeted)
    repairs_skipped: int = 0                                # skipped by budget
    failure_classes: dict = field(default_factory=dict)     # class -> count

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
        if self.attempt_diffs:
            lines.append("Attempt diffs:")
            lines += [f"  retry {i+1}: {summarize_diff(d)}" for i, d in enumerate(self.attempt_diffs)]
        if self.optimization_notes:
            lines.append("Optimizer notes:")
            lines += [f"  - {n}" for n in self.optimization_notes]
        if self.failure_classes:
            lines.append("Failure classes:")
            lines += [f"  - {k}: {v}" for k, v in sorted(self.failure_classes.items())]
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


    def get_db_meta(self, db_url: str, key: str):
        """Namespaced per-database metadata (e.g. learned measure
        equivalences). Returns None when absent."""
        return self._data.get(self._key(db_url), {}).get("meta", {}).get(key)

    def set_db_meta(self, db_url: str, key: str, value):
        entry = self._data.setdefault(self._key(db_url), {})
        entry.setdefault("meta", {})[key] = value
        self._save()


# ---------------------------------------------------------------------------
# Small parsing helpers (no LLM structured-output framework — just regex)
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    """Some Ollama/model versions still emit chain-of-thought inline in
    .content even when reasoning=False ("think": false) is requested — the
    flag isn't universally honored depending on Ollama server version.
    Strip it defensively so downstream parsing isn't corrupted by reasoning
    prose that was never meant to be output. Three observed shapes:

      1. "<think>reasoning</think>answer"          — classic pair
      2. "<think>reasoning..." (unterminated)      — generation hit the
         num_predict cap mid-think; nothing usable follows
      3. "reasoning</think>answer" (ORPHAN close)  — observed live: some
         Ollama/qwen3 template combos swallow the OPENING tag while leaving
         the closing one, so pair-matching alone misses it entirely and the
         whole reasoning block leaked into the answer panel."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"</think>", text, re.IGNORECASE) and not re.search(r"<think>", text, re.IGNORECASE):
        text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


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
    fenced = []
    m = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        fenced.append(m.group(1).strip())
    else:
        m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        if m:
            fenced.append(m.group(1).strip())

    # Unfenced fallbacks: keyword spans to end-of-text. These can swallow
    # trailing fence markers or truncate at blank lines, so they are
    # scrubbed of fence artifacts and tried ONLY after every fenced
    # candidate (a live failure: the fence-swallowing span was longest,
    # failed validation, and the clean fenced query never got its turn).
    unfenced = [c.replace("```", "").strip()
                for c in re.findall(r"(?is)\bWITH\s+\w+\s+AS\s*\(.*?(?=;|\n\s*\n|$)", text)]
    unfenced += [c.replace("```", "").strip()
                 for c in re.findall(r"(?is)\bSELECT\b.*?(?=;|\n\s*\n|$)", text)]

    candidates = fenced + sorted(set(unfenced), key=len, reverse=True)

    seen = set()
    fallback = None
    # Longest first within each tier: a CTE match is a superset of the
    # plain-SELECT match for the same query.
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


def extract_json_object(text: str) -> dict:
    """Extract the first parseable JSON OBJECT from an LLM response.
    Companion to extract_json_list: same defensive philosophy (thinking-mode
    prose can contain stray braces), plus one extra candidate — the outermost
    {..} span — because a nested object (the plan's "ranking": {...}) defeats
    the flat [^{}]-style scan. Returns {} when nothing parses; callers are
    required to fail open."""
    text = _strip_thinking(text)
    candidates = []
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m and "{" in m.group(1):
        candidates.append(m.group(1))
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first:last + 1])
    candidates += re.findall(r"\{[^{}]*\}", text, re.DOTALL)
    for c in candidates:
        try:
            parsed = json.loads(c)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue
    return {}


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
# Fan-out detection — a SIXTH failure class, and the one behind a real
# wrong answer observed live: "What is the total revenue?" produced
# SUM(Invoice.Total) FROM Invoice JOIN InvoiceLine — valid, FK-correct SQL
# returning ~9x the true figure, because every invoice row repeats once per
# line item BEFORE the SUM sees it. validate_join_semantics passes (the
# join IS a real relationship); nothing else examined whether the JOIN
# SHAPE multiplies the aggregated table's rows.
#
# Rule, computed entirely from the AST + declared FKs (nothing hardcoded):
# flag aggregating over table X's own columns while X is joined to any of
# its FK-CHILDREN (tables that reference X), unless the query proves the
# multiplication is absent or intended:
#   - GROUP BY at X's grain (a group column resolves to X, ideally its PK)
#     or at the multiplying child's grain
#   - DISTINCT inside the aggregate
#   - every aggregated column lives on the child side (aggregating child
#     measures across the join is exactly WHY you join it)
# ---------------------------------------------------------------------------

def _children_by_fk(foreign_keys: list) -> dict:
    """parent table -> set of tables that reference it."""
    children = {}
    for fk in foreign_keys:
        children.setdefault(fk["references_table"], set()).add(fk["table"])
    return children


_AGG_TYPES = (exp.Sum, exp.Avg, exp.Count, exp.Min, exp.Max)


def validate_aggregation_fanout(sql: str, foreign_keys: list,
                                 table_schemas: dict, dialect: str = "") -> None:
    try:
        ast = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return  # unparseable SQL is validate_readonly's job, not this

    findings = _fanout_findings(ast, foreign_keys, table_schemas)
    if not findings:
        return
    f = findings[0]
    raise SemanticValidationError(
        f"This query aggregates over '{f['x']}' while ALSO joining "
        f"{sorted(f['multiplying'])}, which reference '{f['x']}' many-to-one — "
        f"each '{f['x']}' row repeats once per matching child row, so "
        f"{f['agg'].sql(dialect=dialect)} DOUBLE-COUNTS (e.g. summing "
        f"invoice totals after joining line items multiplies revenue "
        f"by items-per-invoice). Fix by ONE of: drop the unnecessary "
        f"join to {sorted(f['multiplying'])}; aggregate the child-side "
        f"measure instead; use DISTINCT inside the aggregate; or "
        f"GROUP BY at '{f['x']}''s primary key."
    )


def _fanout_findings(ast: exp.Expression, foreign_keys: list,
                     table_schemas: dict) -> list:
    """Shared detection core for validate_aggregation_fanout and
    repair_fanout_join: returns one entry per offending aggregate
    ({'x': aggregated table, 'multiplying': set of joined FK-children,
    'agg': the aggregate node}). Empty list = clean.

    Lineage-based rule (v2, after a live false positive taught the
    difference): starting from the FROM-root table B, walk UPWARD through
    declared FKs — those tables form B's lookup lineage. Aggregating any
    lineage column is inflated iff a FK-CHILD of it is also joined (the
    result then carries one row per child). Aggregating columns from BELOW
    the root (descendant/detail tables) is inherently line-grain scaled —
    SUM(line.qty * line.price) is exact revenue, never flagged, no matter
    that the detail table's lookup parents appear in the joins."""
    if not isinstance(ast, exp.Select):
        return []

    known = set(table_schemas)
    for fk in foreign_keys:
        known.add(fk["table"])
        known.add(fk["references_table"])

    aliases = _resolve_table_aliases(ast)
    children_of = _children_by_fk(foreign_keys)
    # Shared FROM-root / detail-side walker (sql_semantics) — single source
    # of truth for join-side reasoning across fan-out AND grain checks.
    base, lineage, below_root = root_and_detail_side(ast, foreign_keys, known)
    query_tables = {t.name for t in ast.find_all(exp.Table) if t.name in known}

    group_cols = []
    group = ast.args.get("group")
    if group:
        for col in group.find_all(exp.Column):
            group_cols.append((aliases.get(col.table, col.table) or None,
                               col.name))

    def _grouped_at_grain(table: str) -> bool:
        pks = {c["name"] for c in table_schemas.get(table, [])
               if c.get("primary_key")}
        for gt, name in group_cols:
            if gt == table and (not pks or name in pks):
                return True
        return False

    findings = []
    for agg in ast.find_all(*_AGG_TYPES):
        # MIN/MAX are idempotent under duplication; DISTINCT dedupes;
        # grouped COUNT is the canonical rows-per-group pattern (its
        # numeric-vs-count blind spot belongs to the metric layers).
        if isinstance(agg, (exp.Min, exp.Max)):
            continue
        if agg.args.get("distinct") is not None:
            continue
        if isinstance(agg, exp.Count) and group_cols:
            continue
        if isinstance(agg.this, exp.Star):
            arg_tables = {base} if base else set()  # star measures root grain
        else:
            arg_tables = {aliases.get(c.table, c.table)
                          for c in agg.this.find_all(exp.Column)}
        arg_tables.discard(None)

        # Pure detail-side measures scale with the line grain themselves —
        # structurally immune to fan-out.
        if arg_tables and arg_tables & below_root:
            continue
        candidates = ([base] if base else []) if not arg_tables \
            else sorted(t for t in arg_tables if t in lineage)
        if not candidates:
            continue  # nothing anchored to the duplicating lineage
        if not below_root:
            continue  # no detail table joined below: no duplication exists

        for x in candidates:
            multiplying = query_tables & children_of.get(x, set())
            if not multiplying:
                continue
            if any(_grouped_at_grain(t) for t in ({x} | multiplying)):
                continue
            findings.append({"x": x, "multiplying": multiplying, "agg": agg})
            break
    return findings


def repair_fanout_join(sql: str, foreign_keys: list,
                       table_schemas: dict, dialect: str = "") -> str:
    """Deterministic fix for the fan-out class: DROP the offending child
    joins outright — the inverse of repair_missing_joins. Safe by
    construction only when every flagged child table's columns appear
    NOWHERE outside its own JOIN ON clause (nothing else in the query
    depends on the join existing); any star, unqualified column, or
    residual finding after removal returns the original SQL untouched —
    house rule: never guess, hand it to the bounded retry instead."""
    try:
        ast = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return sql
    if not isinstance(ast, exp.Select):
        return sql

    findings = _fanout_findings(ast, foreign_keys, table_schemas)
    if not findings:
        return sql

    aliases = _resolve_table_aliases(ast)
    drop = set()
    for f in findings:
        drop |= f["multiplying"]
    if any(isinstance(s, exp.Star) for s in ast.find_all(exp.Star)):
        return sql  # star semantics unclear — don't touch the join shape

    # Columns referenced OUTSIDE any JOIN ON clause make their table
    # load-bearing (SELECT/WHERE/GROUP BY/ORDER BY depend on the join).
    on_column_ids = set()
    for j in ast.find_all(exp.Join):
        on = j.args.get("on")
        if on is not None:
            on_column_ids.update(id(c) for c in on.find_all(exp.Column))
    for c in ast.find_all(exp.Column):
        if id(c) in on_column_ids:
            continue
        resolved = aliases.get(c.table, c.table)
        if not resolved or resolved in drop:
            return sql  # unqualified attribution or direct dependency

    joins = list(ast.args.get("joins") or [])
    kept, dropped = [], []
    for j in joins:
        name = j.this.name if isinstance(j.this, exp.Table) else None
        if name in drop:
            dropped.append(name)
        else:
            kept.append(j)
    if not dropped:
        return sql

    ast.set("joins", kept)

    # Post-conditions: the fan-out is actually gone, and no surviving join
    # still references a dropped table through its ON clause.
    if _fanout_findings(ast, foreign_keys, table_schemas):
        return sql
    for j in kept:
        on = j.args.get("on")
        if on is not None:
            if {aliases.get(c.table, c.table) for c in on.find_all(exp.Column)} & set(dropped):
                return sql

    logger.info(f"[repair] dropped fan-out join(s) to {sorted(set(dropped))} — "
                f"aggregate runs on the picked table alone")
    return ast.sql(dialect=dialect or None)


_TEXT_TYPES = ("NVARCHAR", "VARCHAR", "TEXT", "CHAR", "CLOB", "STRING", "NCHAR")


def _column_type(table_schemas: dict, resolved_table: str, col_name: str) -> str:
    for c in table_schemas.get(resolved_table, []):
        if c["name"].lower() == col_name.lower():
            return str(c.get("type", "")).upper()
    return ""


def repair_string_concat(sql: str, table_schemas: dict, dialect: str = "") -> str:
    """T-SQL-style `first + last` string concatenation silently becomes
    NUMERIC ADDITION on SQLite/Postgres when executed as-is (observed live:
    customer names collapsed to 0 and the answer layer hallucinated names
    to fill the void). Where EVERY leaf operand resolves to text (text-
    typed column or string literal — handles the left-nested tree that
    `a + ' ' + b` produces), rewrite the + chain to portable || . Anything
    ambiguous is left untouched."""
    try:
        ast = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return sql
    if not isinstance(ast, exp.Select):
        return sql

    aliases = _resolve_table_aliases(ast)

    def _is_text_leaf(node) -> bool:
        if isinstance(node, exp.Literal):
            return bool(node.is_string)
        if isinstance(node, exp.Column):
            t = aliases.get(node.table, node.table)
            if not t:
                # Unqualified column (llama3.2 emits these constantly).
                # Resolvable by UNIQUE name across the fetched schema —
                # ambiguous or unknown names stay untouched, per house
                # "never guess" rule.
                owners = [tbl for tbl, cols in table_schemas.items()
                          if any(c["name"].lower() == node.name.lower()
                                 for c in cols)]
                t = owners[0] if len(owners) == 1 else None
            if not t:
                return False
            return _column_type(table_schemas, t,
                                node.name).startswith(_TEXT_TYPES)
        return False

    def _flatten_adds(node, out):
        if isinstance(node, exp.Add):
            _flatten_adds(node.this, out)
            _flatten_adds(node.expression, out)
        else:
            out.append(node)

    changed = False
    # Only TOP-level Add nodes of each chain (parent isn't an Add), so each
    # `a + ' ' + b` tree is rewritten exactly once.
    for add in [a for a in ast.find_all(exp.Add)
                if not isinstance(a.parent, exp.Add)]:
        leaves = []
        _flatten_adds(add, leaves)
        if len(leaves) >= 2 and all(_is_text_leaf(l) for l in leaves):
            new = leaves[0].copy()
            for leaf in leaves[1:]:
                new = exp.DPipe(this=new, expression=leaf.copy())
            add.replace(new)
            changed = True

    if not changed:
        return sql
    logger.info("[repair] rewrote T-SQL '+' string concatenation to '||'")
    try:
        reparsed = sqlglot.parse_one(ast.sql(dialect=dialect or None),
                                     read=dialect or None)
    except Exception:
        return sql
    return reparsed.sql(dialect=dialect or None)


# ---------------------------------------------------------------------------
# Metric-intent validation — catches the OTHER live wrong answer:
# "Which 5 customers spent the most?" answered with COUNT(Invoices).
# Every customer had 7 invoices, so the top-5 came back as arbitrary rows
# with identical counts — structurally perfect, verified numbers, entirely
# the wrong measure. Purely lexical on the QUESTION side (no extra LLM
# call), deliberately narrow to avoid false positives elsewhere in the
# suite: money terms must be present AND the aggregate set must contain no
# SUM-family function at all before COUNT is rejected (and vice versa).
# ---------------------------------------------------------------------------

_MONEY_TERM_RE = re.compile(
    r"\b(?:revenue|spent|spend|spending|sales|earnings|income|billed|turnover)\b"
    r"|\btotal\b(?!\s+(?:number|count)\b)",  # "total number of invoices" ≠ money
    re.IGNORECASE,
)

_COUNT_INTENT_RE = re.compile(r"\bhow many\b|\bnumber of\b|\bcount of\b",
                              re.IGNORECASE)

# Lexical corroboration for plan fields: small models emit noisy plans
# (observed live: llama3.2 claimed ranking.enabled=true, direction=ASC for
# "What is the average invoice total?"). A plan claim is only trusted when
# the question itself shows matching language; otherwise it's downgraded
# with a logged note — fail-open, never silently obeyed.
_RANK_HINT_RE = re.compile(
    r"\b(?:top|bottom|best|worst|first|last|most|least|highest|lowest)\b",
    re.IGNORECASE,
)

# Polarity words: which END of the ranking the question actually wants.
# Observed live: llama3.2 emitted direction=ASC for "Which country has the
# MOST customers?", and the mere-word corroboration above let that through —
# killing three attempts of perfectly-correct DESC SQL. When polarity words
# unambiguously state an end, they OVERPOWER whatever direction the plan
# claimed; only genuinely mixed/neutral questions keep the plan's say.
_RANK_DESC_RE = re.compile(
    r"\b(?:most|top|best|highest|largest|biggest|greatest|max(?:imum)?)\b",
    re.IGNORECASE,
)
_RANK_ASC_RE = re.compile(
    r"\b(?:least|lowest|fewest|smallest|bottom|worst|min(?:imum)?)\b",
    re.IGNORECASE,
)


def corroborate_ranking_direction(question: str, claimed_direction: str) -> str:
    """Return the question-credible ranking direction ('DESC'/'ASC')."""
    wants_desc = bool(_RANK_DESC_RE.search(question))
    wants_asc = bool(_RANK_ASC_RE.search(question))
    claimed = str(claimed_direction or "DESC").upper()
    if wants_desc == wants_asc:  # both or neither: no lexical verdict
        return claimed
    lex = "DESC" if wants_desc else "ASC"
    if lex != claimed:
        logger.info(f"[pick_tables] plan said {claimed} but question polarity "
                    f"says {lex} - overriding")
    return lex


def validate_metric_intent(question: str, sql: str, dialect: str = "",
                           table_schemas: Optional[dict] = None) -> None:
    expects_money = bool(_MONEY_TERM_RE.search(question))
    expects_count = bool(_COUNT_INTENT_RE.search(question)) and not expects_money
    if not (expects_money or expects_count):
        return

    try:
        ast = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return
    if not isinstance(ast, exp.Select):
        return

    aggs = list(ast.find_all(*_AGG_TYPES))
    if not aggs:
        return  # no aggregation at all: other validators cover this shape

    families = {type(a).__name__.upper() for a in aggs}
    money_family_used = bool({"SUM", "AVG", "MIN", "MAX"} & families)
    count_used = "COUNT" in families

    if expects_money and count_used and not money_family_used:
        term = _MONEY_TERM_RE.search(question).group(0)
        raise SemanticValidationError(
            f"The question asks about a monetary AMOUNT (\"{term}\"), but the "
            f"query only uses COUNT(...) — counting rows does not measure "
            f"spending/revenue/totals. Rewrite using SUM() over the numeric "
            f"amount column (e.g. SUM(<fact_table>.Total))."
        )
    if expects_count and ("SUM" in families or "AVG" in families):
        raise SemanticValidationError(
            f"The question asks HOW MANY (a count of things), but the query "
            f"uses SUM()/AVG() over amounts — that measures value, not "
            f"quantity. Use COUNT() on the entity's key column instead."
        )

    # Integer-measure guard (needs schemas; skipped without them). A money
    # question whose EVERY aggregate argument resolves to an INTEGER column
    # is measuring counts-of-things wearing a SUM costume — the live Q4
    # failure ("spent the most" answered with SUM(Quantity)). Only fires
    # when genuine decimal/currency columns exist among the relevant tables,
    # so integer-denominated data (units sold, points scored) is untouched.
    if expects_money and table_schemas and not (
            count_used and not money_family_used):
        aliases = _resolve_table_aliases(ast)

        def _is_int_column(col) -> bool:
            t = aliases.get(col.table, col.table)
            ctype = _column_type(table_schemas, t, col.name) if t else ""
            return ctype.startswith(("INT", "SMALLINT", "BIGINT", "TINYINT"))

        decimal_candidates = sorted({
            c["name"] for cols in table_schemas.values() for c in cols
            if str(c.get("type", "")).upper().startswith(("NUMERIC", "DECIMAL",
                                                          "DEC", "REAL", "FLOA",
                                                          "DOUBLE", "MONEY"))
        })
        if not decimal_candidates:
            return

        bad = True
        for agg in aggs:
            if isinstance(agg.this, exp.Star):
                bad = False  # unknown measure — give benefit of the doubt
                break
            cols = list(agg.this.find_all(exp.Column))
            if not cols or not all(_is_int_column(c) for c in cols):
                bad = False  # at least one non-integer (or unresolvable) arg
                break
        if bad:
            raise SemanticValidationError(
                f"The question asks about a monetary AMOUNT, but every "
                f"{(' / '.join(sorted(families)))}() here runs over INTEGER "
                f"columns — integers are counts/quantities, not money. Use a "
                f"currency-style column such as {decimal_candidates[:4]} "
                f"(or multiply quantity by its unit price)."
            )


# ---------------------------------------------------------------------------
# Plan-vs-SQL consistency — Phase-2 upgrade over the lexical intent check.
# Step 2's LLM call now also emits a structured QUERY PLAN (metric / entity /
# ranking / grouping). When a plan exists, the generated SQL must implement
# IT — regardless of which keywords happened to appear in the question.
# Every check activates ONLY when the plan supplies that key (absent =
# unknown = fail open), so partial plans degrade gracefully.
#
# This structurally closes the COUNT-instead-of-SUM class even for phrasings
# no regex anticipated: the model itself declared "this needs SUM over
# Total, top-5 DESC" — the SQL must comply or take a targeted retry.
# ---------------------------------------------------------------------------

def validate_plan_matches_sql(plan: Optional[dict], sql: str, dialect: str = "",
                              table_schemas: Optional[dict] = None,
                              question: Optional[str] = None) -> None:
    if not plan:
        return

    try:
        ast = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return  # unparseable SQL is validate_readonly's job, not this
    if not isinstance(ast, exp.Select):
        return

    families = {type(a).__name__.upper() for a in ast.find_all(*_AGG_TYPES)}
    metric = plan.get("metric")

    if metric == "NONE":
        if families:
            raise SemanticValidationError(
                f"The query plan says this question needs NO aggregation (a "
                f"plain listing), but the SQL uses {sorted(families)}(). "
                f"Aggregate functions answer 'how much/how many in total' "
                f"questions — remove them and select the requested rows "
                f"directly."
            )
    elif metric in {"SUM", "COUNT", "AVG", "MIN", "MAX"}:
        if not families:
            raise SemanticValidationError(
                f"The query plan requires {metric}(...) aggregation for this "
                f"question, but the SQL contains no aggregate function at "
                f"all. Apply {metric}() to the planned column."
            )
        # 'sold'-style questions legitimately surface as COUNT (documents)
        # or SUM (units): when the question contains sold/sales, accept
        # EITHER family and let the grain layer own the distinction.
        sold_relaxed = bool(question) and metric in ("COUNT", "SUM") and re.search(
            r"\b(?:sold|sales)\b", question, re.IGNORECASE)
        if metric not in families and len(families) == 1 \
                and not (sold_relaxed and "SUM" in families):
            found = next(iter(families))
            raise SemanticValidationError(
                f"The query plan calls for {metric}(), but the SQL uses "
                f"{found}() instead — a different measure answering a "
                f"different question. Rewrite using {metric}() on the "
                f"planned column."
            )
        # Column-level check: the metric must run over the planned column
        # (when the model named one), not an adjacent key/ID column. A
        # planned column living OUTSIDE the planned entity is first REMAPPED
        # to its owning table when that resolution is unique (observed
        # live: planners correctly say "sum InvoiceLine.amount" while
        # naming entity=Customer — that's a legitimate child-measure plan,
        # not noise); only unresolvable column names are discarded.
        metric_column = plan.get("metric_column")
        if metric_column and metric in families and table_schemas:
            exists_anywhere = any(
                metric_column.lower() in
                {c["name"].lower() for c in cols}
                for cols in table_schemas.values())
            if not exists_anywhere:
                logger.info(f"[plan] ignoring metric_column "
                            f"'{metric_column}': no such column in any "
                            f"fetched schema (planner hallucination)")
                metric_column = None
        if metric_column and metric in families:
            effective = metric_column.lower()
            if table_schemas:
                owners = [t for t, cols in table_schemas.items()
                          if any(c["name"].lower() == effective for c in cols)]
                if owners and effective not in {c["name"].lower() for c in table_schemas.get(plan.get("entity"), [])}:
                    if len(owners) == 1:
                        logger.info(f"[plan] metric_column '{metric_column}' "
                                    f"resolved to table '{owners[0]}' "
                                    f"(child-measure plan)")
                    else:
                        logger.info(f"[plan] ignoring metric_column "
                                    f"'{metric_column}': ambiguous across "
                                    f"{sorted(owners)}")
                        metric_column = None
        if metric_column and metric in families:
            def _agg_columns(agg_node) -> set:
                if any(isinstance(s, exp.Star) for s in agg_node.find_all(exp.Star)):
                    return {"*"}
                return {c.name.lower() for c in agg_node.find_all(exp.Column)}

            effective = metric_column.lower()
            ok = True
            strict = False
            acceptable = {effective}

            if table_schemas:
                owners = [t for t, cols in table_schemas.items()
                          if any(c["name"].lower() == effective for c in cols)]
                if len(owners) > 1:
                    # Name exists in several tables — unenforceable, fail open.
                    logger.info(f"[plan] ignoring metric_column "
                                f"'{metric_column}': ambiguous across "
                                f"{sorted(owners)}")
                    metric_column = None
                elif not owners:
                    # Column doesn't exist anywhere in the fetched schema —
                    # an hallucinated planner hint (observed live: 'amount'
                    # against Chinook). Nothing exact to enforce: accept any
                    # numeric measure among relevant tables.
                    logger.info(f"[plan] ignoring metric_column "
                                f"'{metric_column}': no such column in the "
                                f"fetched schema")
                    for cols in table_schemas.values():
                        for c in cols:
                            t = str(c.get("type", "")).upper()
                            numeric = t.startswith(("NUMERIC", "DECIMAL",
                                                    "DEC", "REAL", "FLOA",
                                                    "DOUBLE", "MONEY", "INT"))
                            if numeric and not c.get("primary_key"):
                                acceptable.add(c["name"].lower())
                else:
                    # Unique real owner.
                    if owners[0] in {t.name for t in ast.find_all(exp.Table)}:
                        strict = True
                    else:
                        # Owner exists but was joined away (e.g. fan-out
                        # repair removed the line-items join): same fallback.
                        logger.info(f"[plan] planned column '{metric_column}' "
                                    f"lives in '{owners[0]}', which is not "
                                    f"joined — falling back to numeric-any")
                        for cols in table_schemas.values():
                            for c in cols:
                                t = str(c.get("type", "")).upper()
                                numeric = t.startswith(("NUMERIC", "DECIMAL",
                                                        "DEC", "REAL", "FLOA",
                                                        "DOUBLE", "MONEY", "INT"))
                                if numeric and not c.get("primary_key"):
                                    acceptable.add(c["name"].lower())

            if metric_column is not None:
                metric_aggs = [a for a in ast.find_all(*_AGG_TYPES)
                               if type(a).__name__.upper() == metric]
                if strict:
                    target = effective
                    ok = any(target in _agg_columns(a) for a in metric_aggs)
                else:
                    ok = any(_agg_columns(a) & acceptable for a in metric_aggs)
                if metric_aggs and not ok:
                    raise SemanticValidationError(
                        f"The query plan expects {metric}() over column "
                        f"'{metric_column}', but every {metric}() in the SQL "
                        f"aggregates something else — that measures the wrong "
                        f"thing. Aggregate '{metric_column}' instead."
                    )

    ranking = plan.get("ranking")
    if isinstance(ranking, dict) and ranking.get("enabled") \
            and ranking.get("soft") and not ast.args.get("group"):
        # Soft (injected) ranking only applies to per-entity GROUPED
        # listings; whole-table aggregates have nothing to order.
        ranking = None
    if isinstance(ranking, dict) and ranking.get("enabled"):
        order_node = ast.args.get("order")
        if order_node is None:
            raise SemanticValidationError(
                f"This is a top/bottom-N ranking question per the query "
                f"plan, but the SQL has no ORDER BY. Add ORDER BY <metric> "
                f"{ranking.get('direction', 'DESC')}."
            )
        else:
            want_desc = str(ranking.get("direction") or "DESC").upper() != "ASC"
            descs = [bool(e.args.get("desc")) for e in order_node.expressions]
            if descs and all(d != want_desc for d in descs):
                direction = "DESC" if want_desc else "ASC"
                raise SemanticValidationError(
                    f"Ranking direction mismatch: the plan wants ORDER BY "
                    f"{direction} but the SQL orders the opposite way — that "
                    f"returns the WORST rows where the BEST were asked for. "
                    f"Flip the sort direction."
                )
        planned_limit = ranking.get("limit")
        if planned_limit is not None:
            limit_node = ast.args.get("limit")
            if limit_node is None:
                raise SemanticValidationError(
                    f"Ranking question asks for top {planned_limit} but the "
                    f"SQL has no LIMIT clause — add LIMIT {planned_limit}."
                )
            try:
                n = int(str(limit_node.expression.this))
            except (AttributeError, ValueError):
                n = None  # non-literal LIMIT — can't compare, don't guess
            if n is not None and n > int(planned_limit):
                raise SemanticValidationError(
                    f"LIMIT {n} exceeds the requested top {planned_limit} — "
                    f"cap it at LIMIT {planned_limit}."
                )

    entity = plan.get("entity")
    if entity:
        tables_in_sql = {t.name for t in ast.find_all(exp.Table)}
        if tables_in_sql and entity not in tables_in_sql:
            raise SemanticValidationError(
                f"The query plan identifies '{entity}' as the main table the "
                f"question is about, but it never appears in FROM/JOIN "
                f"(query touches: {sorted(tables_in_sql)}). Query the planned "
                f"table."
            )


def repair_metric_column(sql: str, plan: Optional[dict], table_schemas: dict,
                         dialect: str = "") -> str:
    """Deterministic fix for plan-vs-SQL measure confusion: when every
    condition is provably safe — the plan names a numeric column, that
    column exists in the SAME table as the currently-aggregated one (so the
    row grain is identical and only the MEASURE was mixed up, e.g.
    SUM(InvoiceLine.Quantity) vs the planned SUM(InvoiceLine.amount)) —
    swap the aggregated column. COUNT is excluded (counting rows is not
    interchangeable with summing values). Anything ambiguous returns the
    original SQL untouched."""
    if not plan:
        return sql
    metric = plan.get("metric")
    metric_column = plan.get("metric_column")
    if metric not in {"SUM", "AVG", "MIN", "MAX"} or not metric_column:
        return sql
    try:
        ast = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return sql
    if not isinstance(ast, exp.Select):
        return sql

    aliases = _resolve_table_aliases(ast)

    def _numeric(col_type: str) -> bool:
        return bool(col_type) and col_type.startswith(
            ("NUMERIC", "DECIMAL", "DEC", "REAL", "FLOA", "DOUBLE", "MONEY"))

    planned_ok = False
    changed = False
    for agg in ast.find_all(*_AGG_TYPES):
        if type(agg).__name__.upper() != metric:
            continue
        if agg.args.get("distinct") is not None or isinstance(agg.this, exp.Star):
            continue
        for col in list(agg.this.find_all(exp.Column)):
            t = aliases.get(col.table, col.table)
            if not t:
                continue
            planned_type = _column_type(table_schemas, t, metric_column)
            current_type = _column_type(table_schemas, t, col.name)
            same_table_swap = (
                col.name.lower() != metric_column.lower()
                and _numeric(planned_type) and _numeric(current_type)
            )
            # First hit proves the planned column exists where we're aggregating.
            if col.name.lower() == metric_column.lower() and _numeric(current_type):
                planned_ok = True
            if same_table_swap:
                col.set("this", exp.to_identifier(metric_column))
                planned_ok = True
                changed = True

    if not changed or not planned_ok:
        return sql
    logger.info(f"[repair] swapped aggregate argument to planned column "
                f"'{metric_column}' ({metric})")
    try:
        reparsed = sqlglot.parse_one(ast.sql(dialect=dialect or None),
                                     read=dialect or None)
    except Exception:
        return sql
    return reparsed.sql(dialect=dialect or None)


# ---------------------------------------------------------------------------
# Column-existence validation — closes stage-4D: everything upstream parsed,
# joined, and measured correctly, yet SUM(IL.Total) sailed through and died
# at the database because InvoiceLine has no Total column (observed live).
# We already hold every relevant table's schema, so reference them: every
# qualified column must exist on its resolved table; every unqualified one
# must resolve somewhere among the queried tables (2+ owners = the classic
# ambiguous-CustomerId failure, now rejected HERE with an actionable
# "qualify it" message instead of a cryptic sqlite error three lines later).
# ---------------------------------------------------------------------------

def validate_columns_exist(sql: str, table_schemas: dict, dialect: str = "") -> None:
    import difflib

    try:
        ast = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return
    if not isinstance(ast, exp.Select):
        return

    aliases = _resolve_table_aliases(ast)
    query_tables = {t.name for t in ast.find_all(exp.Table)}
    lowered = {tbl: {c["name"].lower() for c in cols}
               for tbl, cols in table_schemas.items()}
    owners_index = {}
    for tbl, names in lowered.items():
        for n in names:
            owners_index.setdefault(n, []).append(tbl)

    def _suggest(name, pool):
        hits = difflib.get_close_matches(name.lower(), list(pool), n=3, cutoff=0.6)
        return ", ".join(hits) if hits else "no similar column"

    # SELECT-list output aliases are legal references in ORDER BY / HAVING /
    # (sqlite) GROUP BY — never schema columns, so whitelist them before
    # checking anything unqualified.
    output_aliases = {e.alias.lower() for e in ast.expressions
                      if isinstance(e, exp.Alias)}

    for col in ast.find_all(exp.Column):
        qualifier = col.table
        name_l = col.name.lower()

        if not qualifier and name_l in output_aliases:
            continue  # ORDER BY <select-alias> etc.
        resolved = aliases.get(qualifier, qualifier) if qualifier else ""
        if not resolved:
            resolved = ""

        if qualifier and resolved not in lowered:
            continue  # derived table / CTE / unknown scope: not ours to check

        if resolved:
            if name_l not in lowered[resolved]:
                similar = _suggest(col.name, lowered[resolved])
                raise SemanticValidationError(
                    f"Column '{qualifier}.{col.name}' does not exist on table "
                    f"'{resolved}'. Closest existing columns: {similar}. "
                    f"Check the schema above and use an existing column."
                )
        else:
            # Unqualified: must resolve among the queried tables well enough
            # to execute. 0 owners = unknown column; 2+ owners restricted to
            # the joined set = ambiguous exactly like sqlite's error, but
            # with the fix stated up front.
            in_query = [t for t in owners_index.get(name_l, [])
                        if t in query_tables]
            if len(in_query) == 0:
                all_owners = owners_index.get(name_l, [])
                pool = set().union(*(lowered[t] for t in query_tables
                                     if t in lowered)) if query_tables else set()
                similar = _suggest(col.name, pool)
                hint = f" (it exists on {sorted(all_owners)}, which are not part of this query)" if all_owners else ""
                raise SemanticValidationError(
                    f"Column '{col.name}' is not provided by any table in "
                    f"this query{hint}. Closest existing columns: {similar}."
                )
            if len(in_query) > 1:
                raise SemanticValidationError(
                    f"Column '{col.name}' is AMBIGUOUS: it exists on several "
                    f"joined tables ({sorted(in_query)}). Qualify it with the "
                    f"intended table (e.g. {in_query[0]}.{col.name}) — note "
                    f"GROUP BY/ORDER BY references count too."
                )


# ---------------------------------------------------------------------------
# Semantic layer — grain validation/repair, measure-equivalence learning,
# and the safe-shape measure optimizer. Deterministic complements to the
# syntactic pipeline above; see sql_semantics.py for the inference cores.
# ---------------------------------------------------------------------------

def _norm_expr(s: str) -> str:
    return re.sub(r"\s+", "", s or "").lower()


def _sales_sales_hint(tables):
    return sorted({t for t in tables})[:3]


def validate_count_grain(question: str, plan: Optional[dict], sql: str,
                         table_schemas: dict, foreign_keys: list,
                         dialect: str = "", profile: Optional[dict] = None) -> None:
    """Reject COUNT-style queries whose counting grain sits BELOW the grain
    the question asks about ('most purchases' counted as InvoiceLine rows
    instead of Invoices — the live failure that motivated this layer)."""
    try:
        ast = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return
    if not isinstance(ast, exp.Select):
        return
    counts = [a for a in ast.find_all(*_AGG_TYPES)
              if isinstance(a, exp.Count)]
    if not counts:
        return

    grains = classify_grains(table_schemas, foreign_keys)
    adj, ch = build_fk_maps(foreign_keys)
    entity = plan.get("entity") if plan else None
    expected = None
    resolved = None
    if profile:
        resolved = resolve_measure_source(question, plan or {},
                                          table_schemas, foreign_keys,
                                          profile)
        if resolved and not resolved.get("weak"):
            src = resolved["source"]
            if src and src in grains:
                expected = (src, grains[src])
    if not expected:
        expected = infer_question_grain(question, entity, grains, adj)
    if not expected:
        return
    expected_table, exp_info = expected
    if not exp_info.is_eventish:
        return  # dimension-level units: nothing to compare against rows

    known = set(table_schemas)
    for fk in foreign_keys:
        known.add(fk["table"])
        known.add(fk["references_table"])
    base, lineage, below = root_and_detail_side(ast, foreign_keys, known)
    query_tables = {t.name for t in ast.find_all(exp.Table) if t.name in known}
    contributing = ({base} if base else set()) | below
    contributing &= query_tables
    if not contributing:
        return
    effective = max(contributing,
                    key=lambda t: grains[t].depth if t in grains else 0)
    eff_info = grains.get(effective)
    if not eff_info or effective == expected_table:
        return

    # 'sold'/'sales' questions must traverse the SALES lineage. A query
    # whose joins never touch an invoice-family table is counting catalog
    # rows, not sold units — the live 'tracks sold' miss. Checked FIRST:
    # even when the counted table isn't a descendant of the expected one,
    # missing sales lineage is itself the violation.
    if resolved and resolved.get("requires_sales_lineage"):
        sales_tables = set(resolved.get("sales_lineage_hint") or [])
        if sales_tables and not (query_tables & sales_tables):
            raise SemanticValidationError(
                f"The question asks about SOLD units, but this query never "
                f"joins the sales lineage {sorted(sales_tables)}. "
                f"Add the join path through it so rows represent actual "
                f"sales rather than catalog entries."
            )

    # Only a real descendant relationship is a violation: counting an
    # UNRELATED or PARENT-side table isn't a grain error.
    if effective not in descendants(expected_table, ch):
        return

    pk = exp_info.pk_cols[0] if exp_info.pk_cols else "*"
    raise SemanticValidationError(
        f"Grain mismatch: the question asks how many {expected_table}-level "
        f"units ({expected_table} rows), but this query counts rows at the "
        f"'{effective}' grain (each {expected_table} repeats once per "
        f"matching {effective} row). Count {expected_table}'s primary key "
        f"instead: COUNT({expected_table}.{pk})."
    )


def repair_count_grain(sql: str, question: str, plan: Optional[dict],
                       table_schemas: dict, foreign_keys: list,
                       dialect: str = "",
                       profile: Optional[dict] = None) -> str:
    """Deterministic companion to validate_count_grain.

    Two-step rewrite when the question targets a document grain but the SQL
    counts a detail grain below it:
      1. retarget every COUNT argument to the expected table's primary key;
      2. demote detail-side tables: drop their joins, or PROMOTE the
         expected table into FROM when the detail table is the root.
    Safety: step 2 runs only against references that REMAIN after step 1
    (the swapped-out arguments obviously referenced the detail table), and
    any surviving reference to a demoted table cancels the whole rewrite."""
    try:
        ast = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return sql
    if not isinstance(ast, exp.Select):
        return sql
    counts = [a for a in ast.find_all(*_AGG_TYPES) if isinstance(a, exp.Count)]
    if not counts:
        return sql

    grains = classify_grains(table_schemas, foreign_keys)
    adj, ch = build_fk_maps(foreign_keys)
    entity = plan.get("entity") if plan else None
    expected = None
    resolved = None
    if profile:
        resolved = resolve_measure_source(question, plan or {},
                                          table_schemas, foreign_keys,
                                          profile)
        if resolved and not resolved.get("weak") and resolved.get("source"):
            expected = (resolved["source"], grains.get(resolved["source"]))
    if not expected:
        expected = infer_question_grain(question, entity, grains, adj)
    if not expected:
        return sql
    expected_table, exp_info = expected
    if not exp_info.is_eventish or not exp_info.pk_cols:
        return sql

    known = set(table_schemas)
    for fk in foreign_keys:
        known.add(fk["table"])
        known.add(fk["references_table"])
    base, lineage, below = root_and_detail_side(ast, foreign_keys, known)
    query_tables = {t.name for t in ast.find_all(exp.Table) if t.name in known}
    contributing = ({base} if base else set()) | below
    contributing &= query_tables
    if not contributing:
        return sql
    effective = max(contributing,
                    key=lambda t: grains[t].depth if t in grains else 0)
    if effective is None or effective == expected_table \
            or effective not in descendants(expected_table, ch):
        return sql
    if expected_table not in query_tables or not exp_info.pk_cols:
        return sql

    # ---- Step 1: retarget COUNT arguments to the document PK --------------
    pk = exp_info.pk_cols[0]
    alias_of = next((t.alias for t in ast.find_all(exp.Table)
                     if t.name == expected_table and t.alias), None)
    target = exp.column(pk, table=alias_of) if alias_of else exp.column(pk)

    changed = False
    for agg in counts:
        if agg.args.get("distinct"):
            continue
        current = None
        if isinstance(agg.this, exp.Column):
            current = _resolve_table_aliases(ast).get(agg.this.table,
                                                      agg.this.table) or None
        if current is None or current in descendants(expected_table, ch):
            agg.set("this", target.copy())
            changed = True
        elif isinstance(agg.this, exp.Star):
            agg.set("this", target.copy())
            changed = True

    # ---- Step 2: demote detail-side tables --------------------------------
    # Work on a COPY: tentatively remove every below-expected detail table
    # (joins + FROM-root promotion), then verify nothing remaining still
    # references them. Any dangling reference -> whole rewrite cancelled,
    # original SQL returned for the normal LLM retry path.
    aliases_now = _resolve_table_aliases(ast)
    tentative = descendants(expected_table, ch) & query_tables
    work = ast.copy()
    w_joins = list(work.args.get("joins") or [])
    kept, dropped = [], []
    for j in w_joins:
        name = j.this.name if isinstance(j.this, exp.Table) else None
        if name in tentative:
            dropped.append(name)
        else:
            kept.append(j)
    promoted_base = base in tentative if base else False
    if promoted_base:
        new_table = exp.Table(this=exp.to_identifier(expected_table))
        if alias_of:
            new_table.set("alias",
                          exp.TableAlias(this=exp.to_identifier(alias_of)))
        from_node = work.args.get("from_") or work.args.get("from")
        if from_node is not None:
            from_node.set("this", new_table)
        kept = [j for j in kept
                if not (isinstance(j.this, exp.Table)
                        and j.this.name == expected_table)]
        dropped.append(base)
    work.set("joins", kept)

    # Dangling-reference check against the DEMOTED structure.
    w_aliases = _resolve_table_aliases(work)
    dropped_set = set(dropped)
    ok = True
    for c in work.find_all(exp.Column):
        r = w_aliases.get(c.table, c.table)
        if r and r in dropped_set:
            ok = False
            break
    if not ok or not dropped:
        return sql  # cannot demote safely / nothing actually removed

    logger.info(f"[repair] grain retargeted to {expected_table}.{pk}; "
                f"demoted detail side {sorted(dropped_set)} "
                f"(FROM promoted)" if promoted_base else
                f"[repair] grain retargeted to {expected_table}.{pk}")
    return work.sql(dialect=dialect or None)


def aliases_resolve(ast, col):
    aliases = _resolve_table_aliases(ast)
    return aliases.get(col.table, col.table) or None


def apply_measure_optimization(sql: str, equivalences, table_schemas: dict,
                               foreign_keys: list, metrics, dialect: str = "") -> str:
    """Safe-shape measure optimizer: ungrouped scalar SUM over multi-column
    arithmetic on a detail table, where a LEARNED equivalence maps that
    exact expression to a pre-aggregated lineage column -> drop the detail
    dependency and sum the stored column instead. Everything else gets an
    annotation, never a rewrite (grouped shapes fan out under naive swaps)."""
    try:
        ast = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return sql
    if not isinstance(ast, exp.Select) or ast.args.get("group"):
        return sql
    sums = [a for a in ast.find_all(*_AGG_TYPES)
            if isinstance(a, exp.Sum) and not a.args.get("distinct")]
    if len(sums) != 1:
        return sql
    agg = sums[0]
    if not isinstance(agg.this, (exp.Add, exp.Mul, exp.Sub, exp.Div)):
        return sql
    leaves = [c for c in agg.this.find_all(exp.Column)]
    if len(leaves) < 2:
        return sql
    aliases = _resolve_table_aliases(ast)
    detail_tables = {aliases.get(c.table, c.table) for c in leaves}
    if len(detail_tables) != 1 or None in detail_tables:
        return sql
    detail = next(iter(detail_tables))
    # Qualifier-free normalization so aliased model SQL matches the
    # qualifier-free expression stored by the learning probe.
    import copy as _copymod
    _bare = _copymod.deepcopy(agg.this)
    for c in _bare.find_all(exp.Column):
        c.set("table", None)
    expr_key = _norm_expr(_bare.sql(dialect=dialect))
    entry = next((e for e in (equivalences or [])
                  if e.get("detail_table") == detail
                  and _norm_expr(e.get("expr_sql", "")) == expr_key), None)
    if not entry:
        note = (f"recompute pattern SUM({detail}: "
                f"{'*'.join(c.name for c in leaves)}) detected; no learned "
                f"pre-aggregated equivalent yet (--learn-measures)")
        if note not in metrics.optimization_notes:
            metrics.optimization_notes.append(note)
        return sql

    parent, pcol = entry["parent_table"], entry["parent_col"]
    if parent not in table_schemas:
        return sql

    known = set(table_schemas)
    for fk in foreign_keys:
        known.add(fk["table"])
        known.add(fk["references_table"])
    base, _lineage, _below = root_and_detail_side(ast, foreign_keys, known)

    # CASE A: the detail table IS the FROM-root -> PROMOTE the parent into
    # FROM (reusing its existing join node + alias) and drop that join;
    # every other join stays untouched.
    if detail == base:
        all_joins = list(ast.args.get("joins") or [])
        parent_join = next((j for j in all_joins
                            if isinstance(j.this, exp.Table)
                            and j.this.name == parent), None)
        if parent_join is None:
            return sql  # parent not directly joined: can't promote safely
        # Swap the aggregate argument FIRST, using the parent's own alias
        # from the join node (it survives promotion as the new FROM).
        p_alias = parent_join.this.alias
        target_a = exp.column(pcol, table=p_alias)
        swapped = False
        for agg in sums:
            if isinstance(agg.this, (exp.Add, exp.Mul, exp.Sub, exp.Div)) \
                    and not agg.args.get("distinct"):
                agg.set("this", target_a.copy())
                swapped = True
        from_node = ast.args.get("from_") or ast.args.get("from")
        if from_node is not None:
            from_node.set("this", parent_join.this)
        kept = [j for j in all_joins if j is not parent_join]
        ast.set("joins", kept)
        if not swapped:
            return sql
        metrics.optimization_notes.append(
            f"optimizer: SUM(detail arithmetic on {detail}) rewritten to "
            f"SUM({parent}.{pcol}) via learned equivalence")
        logger.info(f"[optimizer] promoted {parent} to FROM; dropped "
                    f"detail-side aggregation over {detail}")
        try:
            reparsed = sqlglot.parse_one(ast.sql(dialect=dialect or None),
                                         read=dialect or None)
        except Exception:
            return sql
        return reparsed.sql(dialect=dialect or None)
    # The detail table must be otherwise unreferenced (safe-drop check).
    # Columns INSIDE the aggregate being replaced obviously reference it —
    # they disappear with the rewrite — so they are excluded, exactly like
    # ON-clause columns.
    protected_ids = set()
    for j in ast.find_all(exp.Join):
        on = j.args.get("on")
        if on is not None:
            protected_ids.update(id(c) for c in on.find_all(exp.Column))
    protected_ids.update(id(c) for a in sums
                         for c in a.find_all(exp.Column))
    for c in ast.find_all(exp.Column):
        if id(c) in protected_ids:
            continue
        r = aliases.get(c.table, c.table)
        if not r or r == detail:
            return sql  # referenced elsewhere / unattributable: don't touch

    h_alias = next((t.alias for t in ast.find_all(exp.Table)
                    if t.name == parent and t.alias), None)
    agg.set("this", exp.column(pcol, table=h_alias))
    joins = list(ast.args.get("joins") or [])
    kept = [j for j in joins
            if not (isinstance(j.this, exp.Table) and j.this.name == detail)]
    if len(kept) != len(joins):
        ast.set("joins", kept)
    metrics.optimization_notes.append(
        f"optimizer: SUM(detail arithmetic on {detail}) rewritten to "
        f"SUM({parent}.{pcol}) via learned equivalence")
    logger.info(f"[optimizer] rewrote recomputed measure to {parent}.{pcol}")
    try:
        reparsed = sqlglot.parse_one(ast.sql(dialect=dialect or None),
                                     read=dialect or None)
    except Exception:
        return sql
    return reparsed.sql(dialect=dialect or None)


def normalize_table_qualifiers(sql: str, dialect: str = "") -> str:
    """When a FROM/JOIN table carries an AS alias, sqlite/postgres reject
    any remaining full-table-name qualifiers ('InvoiceLine.Quantity' after
    'JOIN InvoiceLine AS IL'). llama3.2 mixes both styles constantly; this
    pass rewrites every such qualifier to the table's alias."""
    try:
        ast = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return sql
    alias_by_name = {}
    for t in ast.find_all(exp.Table):
        if t.alias and t.alias.lower() != t.name.lower():
            alias_by_name[t.name] = t.alias
    if not alias_by_name:
        return sql
    changed = False
    for c in ast.find_all(exp.Column):
        q = c.table
        if q and q in alias_by_name:
            c.set("table", exp.to_identifier(alias_by_name[q]))
            changed = True
    if not changed:
        return sql
    logger.info(f"[repair] normalized full-name qualifiers to aliases "
                f"{sorted(alias_by_name)}")
    return ast.sql(dialect=dialect or None)


def relocate_qualifiers(sql: str, table_schemas: dict,
                        dialect: str = "") -> tuple:
    """Qualifier relocation (Layer 1b): a qualified column that doesn't
    exist on its resolved table but DOES exist uniquely among the queried
    tables is remapped to that table ('IL.AlbumId' -> 'Album.AlbumId'), so
    join-path repair can then add any missing hop. Returns
    (new_sql, moved:list[str]); ambiguous/unknown names are left for
    validate_columns_exist to reject with suggestions."""
    try:
        ast = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return sql, []
    if not isinstance(ast, exp.Select):
        return sql, []

    aliases = _resolve_table_aliases(ast)
    known = set(table_schemas)
    query_tables = {t.name for t in ast.find_all(exp.Table) if t.name in known}
    cols_by_table = {t: {c["name"].lower() for c in table_schemas.get(t, [])}
                     for t in known}

    moved = []
    for col in ast.find_all(exp.Column):
        q = col.table
        if not q:
            continue
        r = aliases.get(q, q)
        if r not in cols_by_table:
            continue
        if col.name.lower() in cols_by_table[r]:
            continue
        owners = [t for t in query_tables
                  if col.name.lower() in cols_by_table[t]]
        new_owner = None
        if len(owners) == 1:
            new_owner = owners[0]
        elif len(owners) > 1:
            # Contextual disambiguation: pick the owner whose OTHER columns
            # co-occur in the same result clause cluster (e.g. GROUP BY
            # neighbors). Implemented as: owner with most sibling columns
            # referenced unqualified elsewhere in the query.
            scores = []
            for o in owners:
                siblings = cols_by_table[o] - {col.name.lower()}
                hits = sum(1 for other in ast.find_all(exp.Column)
                           if not other.table
                           and other.name.lower() in siblings)
                scores.append((hits, -len(cols_by_table[o]), o))
            scores.sort(reverse=True)
            if scores and scores[0][0] > 0:
                new_owner = scores[0][2]
        if not new_owner:
            continue
        alias_new = next((t.alias for t in ast.find_all(exp.Table)
                          if t.name == new_owner and t.alias), new_owner)
        col.set("table", exp.to_identifier(alias_new))
        moved.append(f"{q}.{col.name} -> {alias_new}.{col.name}")
    if not moved:
        return sql, []
    logger.info(f"[repair:relocation] {moved}")
    try:
        reparsed = sqlglot.parse_one(ast.sql(dialect=dialect or None),
                                     read=dialect or None)
    except Exception:
        return sql, []
    return reparsed.sql(dialect=dialect or None), moved


def repair_missing_order(sql: str, plan: Optional[dict],
                         dialect: str = "") -> str:
    """Injected (soft) rankings on GROUPED per-entity listings get a
    business-default ORDER BY <metric> DESC when the model omitted it.
    Deterministic, additive-only; bails on any ambiguity."""
    if not plan:
        return sql
    ranking = plan.get("ranking")
    if not (isinstance(ranking, dict) and ranking.get("enabled")
            and ranking.get("soft")):
        return sql
    try:
        ast = sqlglot.parse_one(sql, read=dialect or None)
    except Exception:
        return sql
    if not isinstance(ast, exp.Select):
        return sql
    if ast.args.get("order") or not ast.args.get("group"):
        return sql
    direction = "ASC" if str(ranking.get("direction") or "DESC").upper() == "ASC" else "DESC"
    # Order by the first aggregated OUTPUT ALIAS (precise), falling back to
    # the first aliased expression; bare positional refs break after the
    # tie-rewrite wraps the query.
    agg_alias = next((e.alias for e in ast.expressions
                      if isinstance(e, exp.Alias)
                      and any(isinstance(x, _AGG_TYPES)
                              for x in e.find_all(_AGG_TYPES))), None)
    if not agg_alias:
        return sql
    new_order = sqlglot.parse_one(
        f"ORDER BY {agg_alias} {direction}", read=dialect or None)
    ast.set("order", new_order)
    logger.info(f"[repair] injected ORDER BY 1 {direction} for per-entity listing")
    return ast.sql(dialect=dialect or None)


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


def _strip_list_markers(text: str) -> str:
    """Remove leading list markers ('1. ', '2) ', etc.) so they aren't
    mistaken for data values during number extraction — done via actual
    list-marker pattern matching, not a blanket 'small numbers are probably
    an index' guess (which turned out to let real hallucinated small counts,
    e.g. a fabricated '15', slip through unflagged)."""
    return re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)


# Negative lookbehind/lookahead for a letter excludes digits embedded in an
# alphanumeric token (e.g. the "2" in "U2", a real band name) from being
# treated as a data figure — a real observed false positive. Commas INSIDE a
# digit run are part of the token ("20,848.62"), commas adjacent to spaces
# are not (list separation: "7, 1").
_NUMBER_RE = re.compile(r"(?<![A-Za-z])-?\d[\d,]*(?:\.\d+)?(?![A-Za-z])")


def _to_number(token: str):
    """Normalize one numeric token to float: strips thousands separators.
    Returns None for anything that isn't actually a number after cleanup."""
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def _answer_figures(text: str) -> list:
    """Every numeric figure in `text` as (float_value, declared_decimals).
    declared_decimals is how many digits were literally written after the
    point ("195.10" -> 2, "15" -> 0) — used to allow rounded citations of
    real values without letting near-miss fabrications through."""
    text = _strip_list_markers(text)
    figures = []
    for m in _NUMBER_RE.finditer(text):
        v = _to_number(m.group(0))
        if v is None:
            continue
        tail = m.group(0).partition(".")[2]
        # Trailing "." alone yields no decimals; "49.62" -> 2.
        figures.append((v, len(tail)))
    return figures


def _extract_numbers_from_rows(result: dict) -> set:
    """Raw numeric values present in the query results, as floats.
    MCP's JSON serialization turns non-JSON-native types (e.g. Postgres
    NUMERIC) into strings — a real observed bug: "59.88" arrived as a
    string and was silently never matched; parse those too."""
    raw = set()
    for row in result.get("rows", []):
        for val in row:
            if isinstance(val, bool):
                continue  # bool is an int subclass; True->1.0 is not data
            if isinstance(val, (int, float)):
                raw.add(float(val))
            elif isinstance(val, str):
                v = _to_number(val.strip())
                if v is not None:
                    raw.add(v)
    # The row count itself is a legitimate, commonly-stated fact ("there
    # are N results") even when it isn't a literal cell value anywhere.
    raw.add(float(len(result.get("rows", []))))
    return raw


def _grounded(value: float, declared_decimals: int, raw: set) -> bool:
    """Is this answer figure traceable to the query results?

    Exact float match against any raw value first (comma/decimal-format
    differences already normalize away: "195.10" == 195.1, "20,848.62" ==
    20848.62 — the two real false-positive shapes observed live).

    Then ONE tolerance, deliberately narrow: DECIMAL citations may match a
    raw value at the precision the answer itself stated ("$5.65" citing
    5.651941...). INTEGER claims stay strict — integers are where counts
    live, and a fabricated count that merely rounds from a nearby real
    value (14.99 invoices -> "15") must keep failing, exactly like the
    historical hallucinated-'15' incident."""
    if value in raw:
        return True
    if declared_decimals >= 1:
        k = min(declared_decimals, 4)
        return any(round(r, k) == value for r in raw)
    return False


def verify_answer(answer: str, result: dict) -> list:
    """Returns the numeric figures in `answer` that could NOT be traced
    back to any value actually present in the query results (or the row
    count). Empty list = every figure in the answer is grounded in real
    data (up to legitimate rounding at the answer's own precision)."""
    raw = _extract_numbers_from_rows(result)
    unverified = [v for v, dec in _answer_figures(answer) if not _grounded(v, dec, raw)]
    return sorted(unverified)


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
                 use_cache: bool = True,
                 learn_measures: bool = False):
        self.db_url = db_url
        self.llm = llm
        self.dialect = dialect
        self.max_retries = max_retries
        self.cache = SchemaCache(ttl=cache_ttl) if use_cache else None
        self.learn_measures = learn_measures
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
        # Strip <think> blocks at the single choke point every LLM consumer
        # goes through. extract_sql/extract_json_list strip again defensively,
        # but format_answer's output goes straight to the user — and live
        # runs showed raw chain-of-thought leaking into the answer panel.
        text = _strip_thinking(text)
        logger.info(f"[llm:{step}] {text[:200]}")
        return text

    # ---- Governed deterministic-repair entry point ------------------------
    REPAIR_BUDGET = 2  # max deterministic repairs per query (user directive)

    def _repair_budget_left(self) -> bool:
        return self.metrics.repairs_applied < self.REPAIR_BUDGET

    def _tag_failure(self, exc: Exception):
        cls = classify_error(str(exc),
                             default=FailureClass.EXECUTION_ERROR
                             if "Error executing tool" in str(exc)
                             else FailureClass.NONE)
        self.metrics.failure_classes[cls] = \
            self.metrics.failure_classes.get(cls, 0) + 1

    def _apply_repair(self, name: str, repair_fn, sql: str, *args,
                      revalidate: bool = True):
        """Apply ONE deterministic repair under governance:
          - budget check (cap 2/query): beyond it, return original SQL and
            count a skip — validators still fire, LLM retry takes over;
          - post-apply structural revalidation (syntax / read-only / join
            semantics / column existence) so no repair output reaches
            execution without passing the structural layer."""
        if not self._repair_budget_left():
            self.metrics.repairs_skipped += 1
            logger.info(f"[repair:{name}] budget exhausted "
                        f"({self.metrics.repairs_applied}/"
                        f"{self.REPAIR_BUDGET}) - skipping")
            return sql, False
        new_sql = repair_fn(sql, *args)
        if new_sql == sql:
            return sql, False
        if revalidate:
            self.validate_sql(new_sql)
            try:
                validate_columns_exist(new_sql, *self._revalidate_args())
            except SemanticValidationError as e:
                logger.info(f"[repair:{name}] output failed structural "
                            f"revalidation: {e}")
                return sql, False
        self.metrics.repairs_applied += 1
        return new_sql, True

    def _revalidate_args(self):
        fks = getattr(self, "_last_foreign_keys", []) or []
        schemas = getattr(self, "_last_table_schemas", {}) or {}
        return (schemas, fks)

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

    # ---- Step 2: pick relevant tables + build a query plan (ONE LLM call) ----
    _PLAN_METRICS = {"SUM", "COUNT", "AVG", "MIN", "MAX", "NONE"}

    async def pick_relevant_tables(self, question: str, all_tables: list) -> tuple:
        """One cheap LLM call that does BOTH jobs: picks the table NAMES
        (schema not fetched yet — same cost as the old name-only picker)
        and extracts a structured query plan (metric / entity / ranking /
        grouping) that later stages validate the generated SQL against.

        Returns (tables, plan). plan is a dict whose keys may be absent —
        absent means "unknown, don't validate against it" (fail-open); it
        is None outright when the ≤3-tables skip fires (no LLM call at
        all), in which case lexical validators still apply."""
        if len(all_tables) <= 3:
            return all_tables, None  # not worth an LLM call for a tiny schema

        prompt = (
            f"Database tables: {all_tables}\n\n"
            f"Question: {question}\n\n"
            "Produce a query plan as ONE JSON object with exactly these keys:\n"
            '- "tables": array of table names needed to answer this — include '
            "any table needed to JOIN to the answer even if not named directly.\n"
            '- "metric": "SUM", "COUNT", "AVG", "MIN", "MAX" or "NONE" — how the '
            'question aggregates ("how many"/"number of" -> COUNT; "total '
            'revenue"/"spent"/"sales" -> SUM over the amount column; a plain '
            "listing -> NONE).\n"
            '- "metric_column": the column the metric applies to, or null.\n'
            '- "entity": the main table the question is about, or null.\n'
            '- "ranking": {"enabled": <bool>, "direction": "DESC"|"ASC", '
            '"limit": <int or null>} — enabled only for top/bottom/best/first-N '
            "questions.\n"
            '- "grouping": column to group by for per-X breakdowns ("per '
            'customer"), or null.\n\n'
            "Respond with ONLY the JSON object, nothing else."
        )
        text = await self._call_llm(prompt, "pick_tables")
        raw = extract_json_object(text)

        picked = [t for t in raw.get("tables", []) or [] if t in all_tables]
        if not picked:
            logger.warning(
                "[pick_tables] Could not parse a usable table list from the model's "
                "response — falling back to ALL tables (safe, but defeats schema "
                "filtering for this call; plan checks degrade accordingly). "
                "Response was: " + text[:150]
            )
            picked = list(all_tables)  # fail open, not closed

        plan = {}
        metric = str(raw.get("metric") or "").upper()
        if metric in self._PLAN_METRICS:
            # Corroboration gate: an aggregation metric is only trusted when
            # the question shows matching language (money terms, or how-many
            # phrasing). Observed live: a plain listing question came back
            # with metric=SUM — obeying it would have rejected perfectly
            # good non-aggregated SQL.
            if metric != "NONE" and not (
                _MONEY_TERM_RE.search(question) or _COUNT_INTENT_RE.search(question)
            ):
                logger.info(f"[pick_tables] plan claimed metric={metric} but the "
                            f"question shows no aggregation language - ignoring")
            else:
                plan["metric"] = metric
        ranking = raw.get("ranking")
        if isinstance(ranking, dict) and ranking.get("enabled"):
            if not _RANK_HINT_RE.search(question):
                logger.info("[pick_tables] plan claimed ranking but the question "
                            "shows no ranking language - ignoring")
            else:
                plan["ranking"] = {
                    "enabled": True,
                    "direction": corroborate_ranking_direction(
                        question, ranking.get("direction")),
                    "limit": ranking.get("limit") if isinstance(ranking.get("limit"), int) else None,
                }
        else:
            # No ranking claimed, but an aggregate over an entity implies a
            # business-default ordering: best/highest first. Injects DESC
            # so per-entity listings stop coming back ascending (live Q4).
            if metric in ("SUM", "COUNT", "AVG"):
                plan["ranking"] = {"enabled": True, "direction": "DESC",
                                   "limit": None, "soft": True}
        entity = raw.get("entity")
        if entity in all_tables:
            plan["entity"] = entity
        if raw.get("metric_column"):
            plan["metric_column"] = str(raw["metric_column"])
        if raw.get("grouping"):
            plan["grouping"] = str(raw["grouping"])
        return picked, plan

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
        answer = await self._call_llm(prompt, "format_answer")

        # Truncation guard: qwen3 emits <think> reasoning even when told not
        # to, and under a tight num_predict cap it can spend the ENTIRE
        # budget on that reasoning and get cut off before producing any
        # answer text (_strip_thinking leaves empty string). One corrective
        # re-invoke with reasoning explicitly forbidden; failing that is a
        # loud error, never a silently-blank answer panel.
        if not answer:
            logger.warning("[format_answer] model produced only reasoning and no "
                           "answer (likely truncated mid-think) — retrying with "
                           "reasoning explicitly forbidden")
            answer = await self._call_llm(
                prompt + "\n\nIMPORTANT: Do NOT think out loud or emit any "
                "reasoning. Output ONLY the final answer text itself.",
                "format_answer",
            )
        if not answer:
            raise RuntimeError(
                "Model produced no answer text (reasoning consumed the entire "
                "token budget twice). Re-run with a higher --max-tokens."
            )
        return answer

    def _build_schema_context(self, table_schemas: dict, foreign_keys: list, relevant: list) -> str:
        """Schema filtering: only the picked tables' columns, only FKs that
        connect two picked tables — this is what keeps the prompt small.
        `table_schemas` here only ever contains entries for `relevant`
        tables in the first place (see run()), so this is filtering FKs,
        not re-filtering columns.

        Adds auto-derived fact/dimension hints (from the FK graph's
        direction, nothing hardcoded): tables that REFERENCE others are
        transactional/event tables whose numeric columns are already
        aggregated per row; referenced tables are lookups. This targets an
        observed live failure where the model joined a fact table to its
        granular line-items table and summed the fact column through the
        fan-out — seeing 'already aggregated per event' reduces that."""
        fk_out = {fk["table"] for fk in foreign_keys}
        fk_in = {fk["references_table"] for fk in foreign_keys}

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
        for t in relevant:
            if t in fk_out and t not in fk_in:
                lines.append(
                    f"# NOTE: {t} is a transactional/event table (one row per "
                    f"recorded event) — its numeric columns are already "
                    f"aggregated per event. When aggregating THIS table's own "
                    f"columns, do NOT join its granular child tables."
                )
            elif t in fk_in and t not in fk_out:
                lines.append(f"# NOTE: {t} is a lookup/dimension table "
                             f"(referenced by other tables via FK).")
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

            # Step 2: pick relevant tables + extract the query plan (names
            # only — no schema fetched yet). plan is None on the ≤3-tables
            # skip; plan-vs-SQL checks degrade to the lexical validators.
            seed_tables, query_plan = await self.pick_relevant_tables(question, tables)

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

            # Statistical profile (cached per DB, v4): row counts, column
            # min/max/avg/ndistinct, confirmed-1:N edges. Powers execution
            # validation and graph-first grain resolution.
            profile = None
            if self.cache is not None:
                profile = self.cache.get_db_meta(self.db_url, "profile")
                if profile is None and len(table_schemas) <= 20:
                    try:
                        profile = build_profile(
                            table_schemas, foreign_keys, self.execute_sql,
                            log=lambda m: logger.info(f"[profile] {m}"))
                        self.cache.set_db_meta(self.db_url, "profile", profile)
                        logger.info("[profile] statistical schema profile built")
                    except Exception as exc:
                        logger.info(f"[profile] build failed: {exc}")

            # Graph-first measure/grain expectation (v3): consumed by the
            # count-grain validator/repair below. Falls back to lexical
            # matching when the plan carries no usable signal.
            measure_exp = resolve_measure_source(
                question, query_plan or {}, table_schemas, foreign_keys,
                profile or {})
            if measure_exp.get("weak"):
                logger.info("[measure] weak plan signal - lexical fallback")

            error_context = ""
            sql = ""
            result = None
            last_error = None
            prev_attempt_sql = None

            # Step 6: explicit, bounded retry loop
            for attempt in range(self.max_retries + 1):
                self.metrics.attempts += 1
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
                # Deterministic dialect repair — T-SQL '+' on text columns
                # silently becomes numeric addition elsewhere (names turn
                # into 0s and the answer layer starts hallucinating to fill
                # the void). Rewritten to '||' where both sides are text.
                # Alias/qualifier style normalization BEFORE anything else:
                # mixed full-name + alias qualifiers break sqlite/postgres.
                sql = normalize_table_qualifiers(sql,
                                                 _sqlglot_dialect(self.dialect))
                sql = repair_string_concat(sql, table_schemas, _sqlglot_dialect(self.dialect))
                # Semantic measure optimizer — safe-shape only (ungrouped
                # scalar SUM over learned-equivalent detail arithmetic).
                equiv = self.cache.get_db_meta(self.db_url, "measure_equiv") \
                    if self.cache else None
                sql = apply_measure_optimization(sql, equiv, table_schemas,
                                                 foreign_keys, self.metrics,
                                                 _sqlglot_dialect(self.dialect))
                # Injected-ranking ordering: grouped per-entity listings get
                # ORDER BY <metric> DESC deterministically.
                sql = repair_missing_order(sql, query_plan,
                                           _sqlglot_dialect(self.dialect))
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
                        self._tag_failure(e)
                        self.metrics.semantic_rejections += 1
                        if not self._repair_budget_left():
                            self.metrics.repairs_skipped += 1
                            raise  # budget exhausted: LLM retry instead
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
                        self.metrics.repairs_applied += 1
                        self.validate_sql(sql)
                        validate_join_semantics(sql, foreign_keys, _sqlglot_dialect(self.dialect))
                    # Column-existence check now runs AFTER join repair:
                    # repaired shapes legitimately change which columns are
                    # referenced, so hallucinated-column rejection must not
                    # preempt deterministic join reconstruction (live Q1
                    # failure: IL.AlbumId died before Track could be added).
                    try:
                        new_sql_rel, relocated = relocate_qualifiers(
                            sql, table_schemas, _sqlglot_dialect(self.dialect))
                        if relocated and self._repair_budget_left():
                            sql = new_sql_rel
                            self.metrics.repairs_applied += 1
                            logger.info("[repair:relocation] remapped "
                                        "misqualified column(s)")
                        elif relocated:
                            self.metrics.repairs_skipped += 1
                        validate_columns_exist(sql, table_schemas,
                                               _sqlglot_dialect(self.dialect))
                    except SemanticValidationError as e:
                        self._tag_failure(e)
                        self.metrics.semantic_rejections += 1
                        raise
                    # Fan-out check — catches a REAL FK join to a child table
                    # that multiplies the aggregated table's rows (e.g.
                    # SUM(Invoice.Total) JOIN InvoiceLine ~9x inflates).
                    # Structurally valid, semantically wrong. When the
                    # offending child join is provably load-bearing-free it
                    # is dropped deterministically (repair_fanout_join);
                    # otherwise a targeted LLM retry with the explanation.
                    try:
                        validate_aggregation_fanout(sql, foreign_keys, table_schemas,
                                                    _sqlglot_dialect(self.dialect))
                    except SemanticValidationError as e:
                        self._tag_failure(e)
                        self.metrics.semantic_rejections += 1
                        if not self._repair_budget_left():
                            self.metrics.repairs_skipped += 1
                            raise
                        repaired = repair_fanout_join(sql, foreign_keys, table_schemas,
                                                      _sqlglot_dialect(self.dialect))
                        if repaired == sql:
                            raise  # nothing we could deterministically fix
                        sql = repaired
                        self.validate_sql(sql)
                        self.metrics.repairs_applied += 1
                        validate_aggregation_fanout(sql, foreign_keys, table_schemas,
                                                    _sqlglot_dialect(self.dialect))
                    # Metric-intent check — question says money ("spent",
                    # "revenue") but query only COUNTs, or asks how-many but
                    # query SUMs; plus the integer-measure guard (money
                    # answered with SUM over INTEGER quantity columns).
                    try:
                        validate_metric_intent(question, sql,
                                               _sqlglot_dialect(self.dialect),
                                               table_schemas=table_schemas)
                    except SemanticValidationError as e:
                        self._tag_failure(e)
                        self.metrics.semantic_rejections += 1
                        raise
                    # Plan-vs-SQL consistency — when the Step-2 plan call
                    # produced a structured plan, the generated SQL must
                    # actually implement it: right metric family, right
                    # entity, and for top-N questions an ORDER BY in the
                    # planned direction with a sane LIMIT. This is what
                    # structurally prevents COUNT-instead-of-SUM style
                    # measure swaps regardless of question keywords.
                    try:
                        validate_plan_matches_sql(query_plan, sql,
                                                  _sqlglot_dialect(self.dialect),
                                                  table_schemas=table_schemas,
                                                  question=question)
                    except SemanticValidationError as e:
                        self._tag_failure(e)
                        self.metrics.semantic_rejections += 1
                        if not self._repair_budget_left():
                            self.metrics.repairs_skipped += 1
                            raise
                        repaired = repair_metric_column(sql, query_plan,
                                                        table_schemas,
                                                        _sqlglot_dialect(self.dialect))
                        if repaired == sql:
                            raise  # nothing we could deterministically fix
                        sql = repaired
                        self.validate_sql(sql)
                        self.metrics.repairs_applied += 1
                        validate_plan_matches_sql(query_plan, sql,
                                                  _sqlglot_dialect(self.dialect),
                                                  table_schemas=table_schemas,
                                                  question=question)
                    # Count-grain check — 'purchases'/'invoices'-style
                    # questions must be counted at the document grain, not a
                    # detail-line grain. Deterministic retarget first.
                    try:
                        validate_count_grain(question, query_plan, sql,
                                             table_schemas, foreign_keys,
                                             _sqlglot_dialect(self.dialect),
                                             profile=profile)
                    except SemanticValidationError as e:
                        self._tag_failure(e)
                        self.metrics.semantic_rejections += 1
                        if not self._repair_budget_left():
                            self.metrics.repairs_skipped += 1
                            raise
                        repaired = repair_count_grain(sql, question, query_plan,
                                                      table_schemas,
                                                      foreign_keys,
                                                      _sqlglot_dialect(self.dialect),
                                                      profile=profile)
                        if repaired == sql:
                            raise  # expected table not joined: LLM retry
                        sql = repaired
                        self.metrics.repairs_applied += 1
                        self.validate_sql(sql)
                        validate_count_grain(question, query_plan, sql,
                                             table_schemas, foreign_keys,
                                             _sqlglot_dialect(self.dialect),
                                             profile=profile)
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
                    # Execution-based validation (v3): statistically
                    # impossible results are rejected BEFORE answer
                    # formatting, feeding the bounded retry loop.
                    if profile:
                        validate_execution_result(
                            result, sql, profile, table_schemas,
                            _sqlglot_dialect(self.dialect), sqlglot)
                    last_error = None
                    break
                except Exception as e:
                    last_error = str(e)
                    self._tag_failure(e)
                    logger.warning(f"[attempt {attempt + 1}/{self.max_retries + 1} failed] {last_error}")
                    error_context = f"Query: {sql}\nError: {last_error}"
                    prev_attempt_sql = sql
                    if attempt < self.max_retries:
                        self.metrics.retries += 1

            if last_error:
                raise RuntimeError(
                    f"Failed after {self.max_retries + 1} attempt(s). Last error: {last_error}"
                )

            # ---- Semantic layer post-processing (deterministic) ----
            # 1) attempt-to-attempt semantic diff (observability)
            if prev_attempt_sql is not None:
                diff_info = semantic_diff(prev_attempt_sql, sql,
                                          _sqlglot_dialect(self.dialect))
                self.metrics.attempt_diffs.append(diff_info)

            # 2) measure-equivalence learning from retry pairs / on demand.
            #    Shape: old attempt summed multi-column arithmetic over a
            #    detail table D; new attempt sums a single numeric column
            #    over D's parent H. One paired whole-table probe decides;
            #    agreement persists H.C == SUM(expr-on-D) to the cache.
            learn_now = self.cache is not None and (
                self.learn_measures
                or any("AGGREGATE_CHANGE" in d.get("tags", [])
                       for d in self.metrics.attempt_diffs))
            if learn_now and sql:
                try:
                    _dg = _sqlglot_dialect(self.dialect)
                    grains_ = classify_grains(table_schemas, foreign_keys)
                    adj_, chm_ = build_fk_maps(foreign_keys)
                    old_ast = sqlglot.parse_one(prev_attempt_sql or sql,
                                                read=_dg or None)
                    new_ast = sqlglot.parse_one(sql, read=_dg or None)
                    aliases_o = _resolve_table_aliases(old_ast)
                    aliases_n = _resolve_table_aliases(new_ast)
                    _arith = (exp.Add, exp.Mul, exp.Sub, exp.Div)
                    old_sum = next((a for a in old_ast.find_all(*_AGG_TYPES)
                                    if isinstance(a, exp.Sum)
                                    and isinstance(a.this, _arith)), None)
                    new_col = None
                    for a in new_ast.find_all(*_AGG_TYPES):
                        if isinstance(a, exp.Sum) and isinstance(a.this, exp.Column):
                            new_col = a.this
                            break
                    if old_sum and new_col:
                        d_tables = {aliases_o.get(c.table, c.table)
                                    for c in old_sum.this.find_all(exp.Column)}
                        p_table = aliases_n.get(new_col.table, new_col.table)
                        if len(d_tables) == 1 and p_table in adj_.get(
                                next(iter(d_tables)), set()):
                            d_name = next(iter(d_tables))
                            # Qualifier-free rendering of the detail-side
                            # expression: the probe runs unaliased.
                            import copy as _copymod
                            probe_expr = _copymod.deepcopy(old_sum.this)
                            for c in probe_expr.find_all(exp.Column):
                                c.set("table", None)
                            probe_old = ("SELECT SUM("
                                         + probe_expr.sql(dialect=_dg)
                                         + ") FROM " + d_name)
                            probe_new = (f"SELECT SUM({new_col.name}) "
                                         f"FROM {p_table}")
                            r_old = await self.execute_sql(probe_old)
                            r_new = await self.execute_sql(probe_new)
                            v_old = r_old.get("rows", [[None]])[0][0]
                            v_new = r_new.get("rows", [[None]])[0][0]
                            import sys as _s3
                            print(f"DBG3 v_old={v_old!r} v_new={v_new!r}", file=_s3.stderr)
                            if v_old is not None and v_new is not None:
                                tol = 0.005 * max(abs(float(v_old)),
                                                  abs(float(v_new)), 1.0)
                                if abs(float(v_old) - float(v_new)) <= tol:
                                    eq = self.cache.get_db_meta(
                                        self.db_url, "measure_equiv") or []
                                    expr_txt = old_sum.this.sql(dialect=_dg)
                                    if not any(
                                            _norm_expr(x.get("expr_sql")) == _norm_expr(expr_txt)
                                            and x.get("detail_table") == d_name
                                            for x in eq):
                                        eq.append({"detail_table": d_name,
                                                   "expr_sql": probe_expr.sql(dialect=_dg),
                                                   "parent_table": p_table,
                                                   "parent_col": new_col.name})
                                        self.cache.set_db_meta(
                                            self.db_url, "measure_equiv", eq)
                                        note = (f"learned equivalence: SUM("
                                                f"{expr_txt} on {d_name}) == "
                                                f"{p_table}.{new_col.name}")
                                        if note not in self.metrics.optimization_notes:
                                            self.metrics.optimization_notes.append(note)
                except Exception as exc:  # probes are best-effort
                    logger.info(f"[learn] equivalence probe skipped: {exc}")

            # 3) post-SQL cross-check against a learned pre-aggregated total
            cross_warning = None
            equiv_list = (self.cache.get_db_meta(self.db_url, "measure_equiv")
                          if self.cache else None)
            if equiv_list and result and result.get("row_count") == 1:
                try:
                    _dg = _sqlglot_dialect(self.dialect)
                    cur = sqlglot.parse_one(sql, read=_dg or None)
                    for e_ in equiv_list:
                        def _bare_norm(node):
                            import copy as _cm
                            n = _cm.deepcopy(node)
                            for c in n.find_all(exp.Column):
                                c.set("table", None)
                            return _norm_expr(n.sql(dialect=_dg))
                        hit = any(
                            isinstance(a, exp.Sum)
                            and _bare_norm(a.this) == _norm_expr(e_["expr_sql"])
                            for a in cur.find_all(*_AGG_TYPES))
                        if not hit:
                            continue
                        pr = await self.execute_sql(
                            f'SELECT SUM({e_["parent_col"]}) '
                            f'FROM {e_["parent_table"]}')
                        pv = pr.get("rows", [[None]])[0][0]
                        rv = result["rows"][0][0]
                        if pv is not None and rv is not None:
                            tol = 0.01 * max(abs(float(pv)), 1.0)
                            if abs(float(rv) - float(pv)) > tol:
                                cross_warning = (
                                    f"\n\n⚠️ Cross-check: this figure diverges "
                                    f"from the stored measure "
                                    f"({e_['parent_table']}.{e_['parent_col']} "
                                    f"= {pv}).")
                                self.metrics.optimization_notes.append(
                                    "post-SQL cross-check divergence")
                        break
                except Exception as exc:
                    logger.info(f"[cross-check] skipped: {exc}")

            answer = await self.format_answer(question, sql, result)
            if cross_warning:
                answer += cross_warning

            # Step 8 (new): answer verification. A syntactically valid,
            # semantically valid, correctly executed query can still get a
            # bad final answer if the LLM editorializes when writing it up
            # — an observed real failure was the model appending a
            # fabricated "(likely a typo)" note instead of trusting the
            # DB's actual value. Check that every number in the answer
            # traces back to the real result set; if not, give the model
            # one corrective retry with the mismatch called out explicitly.
            unverified = verify_answer(answer, result)
            if unverified:
                self._tag_failure(
                    SemanticValidationError("could not be fully verified"))
            if unverified:
                self._tag_failure(
                    SemanticValidationError("could not be fully verified"))
            self.metrics.answer_verified = not unverified
            if unverified:
                pretty = ", ".join(f"{n:g}" for n in unverified)
                logger.warning(f"[verify] Answer contains number(s) not found in the "
                                f"query results: {pretty} — retrying format_answer once.")
                self.metrics.verification_retries += 1
                correction_note = (
                    "\n\nIMPORTANT: Your previous answer included the number(s) "
                    f"{pretty}, which do NOT appear anywhere in the result rows "
                    "above. Use ONLY the exact values shown in the rows — do not "
                    "alter, round unexpectedly, invent, or 'correct' any figure."
                )
                answer = await self.format_answer(question, sql, result, correction_note)
                still_unverified = verify_answer(answer, result)
                self.metrics.answer_verified = not still_unverified
                if still_unverified:
                    logger.warning(f"[verify] Still unverified after retry: "
                                   f"{', '.join(f'{n:g}' for n in still_unverified)}")
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
        help="Request the model's extended 'thinking' mode (Ollama only). "
             "Model-dependent: honored by some models (e.g. Qwen3), ignored "
             "by others (e.g. Llama 3.2). Off by default — thinking adds "
             "significant latency for little benefit on straightforward SQL "
             "generation.",
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
    parser.add_argument("--learn-measures", action="store_true",
                        help="Probe for pre-aggregated measure equivalences "
                             "(e.g. Invoice.Total == SUM(UnitPrice*Quantity)) "
                             "and persist them to the schema cache")
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
            learn_measures=args.learn_measures,
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