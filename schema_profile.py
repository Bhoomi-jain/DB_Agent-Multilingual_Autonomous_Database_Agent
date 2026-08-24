"""
schema_profile.py — Statistical model of the database, built from the data
itself (column stats + FK-graph evidence), replacing keyword heuristics for
grain/measure reasoning.

One probe per table gathers everything: row count plus MIN/MAX/AVG/
COUNT(DISTINCT) for every numeric column. The profile is persisted in the
schema cache (db-meta key "profile") so each database pays this cost once.

Derived signals — all computed from numbers, no name matching:
  identifier   : ndistinct ≈ row_count and min ≈ 1  → sequential key
  measure      : numeric but not an identifier      → aggregable quantity
  edge ratio   : child_rows / parent_rows on each FK edge; ratio ≥ 1.5
                 confirms a real 1:N relationship ("confirmed_1N")
  event_depth  : longest confirmed-1:N chain above each table; roots at 0

The profile is intentionally conservative: tables/columns with missing or
ambiguous statistics degrade to "unknown" and downstream validators skip
their checks rather than guessing.
"""

import re

import sqlglot
from sqlglot import exp
from dataclasses import dataclass, field

_INT_PREFIXES = ("INT", "SMALLINT", "BIGINT", "TINYINT")
_NUMERIC_PREFIXES = ("NUMERIC", "DECIMAL", "DEC", "REAL", "FLOA",
                     "DOUBLE", "MONEY") + _INT_PREFIXES


def is_numeric_type(col_type: str) -> bool:
    return str(col_type).upper().startswith(_NUMERIC_PREFIXES)


def is_integer_type(col_type: str) -> bool:
    return str(col_type).upper().startswith(_INT_PREFIXES)


@dataclass
class TableProfile:
    row_count: int = 0
    columns: dict = field(default_factory=dict)
    # columns[col] = {"min": float|None, "max": ..., "avg": ..., "ndistinct": int}


def _numeric_columns(table_schemas: dict, table: str):
    return [c["name"] for c in table_schemas.get(table, [])
            if is_numeric_type(c.get("type", ""))]


def build_profile(table_schemas: dict, foreign_keys: list,
                  execute_fn, max_tables: int = 20,
                  log=None) -> dict:
    """Probe the database (one aggregate SELECT per table) and derive the
    statistical profile.

    execute_fn(sql) -> {"rows": [[...]], ...}   (the agent's MCP executor)
    Returns the profile dict ready for cache persistence.
    """
    tables = list(table_schemas.keys())[:max_tables]
    profile = {
        "tables": {},
        "columns": {},
        "identifiers": {},
        "measures": {},
        "edges": {},
        "event_depth": {},
    }

    def _log(msg):
        if log:
            log(msg)

    for t in tables:
        num_cols = _numeric_columns(table_schemas, t)
        selects = ["COUNT(*)"]
        for c in num_cols:
            selects += [f"MIN({c})", f"MAX({c})", f"AVG({c})",
                        f"COUNT(DISTINCT {c})"]
        sql = f"SELECT {', '.join(selects)} FROM {t}"
        try:
            res = execute_fn(sql)
            row = res.get("rows", [[]])[0]
        except Exception as exc:
            _log(f"[profile] {t}: skipped ({type(exc).__name__})")
            continue

        row_count = int(row[0])
        cols = {}
        idx = 1
        for c in num_cols:
            mn, mx, avg, nd = row[idx], row[idx + 1], row[idx + 2], row[idx + 3]
            idx += 4
            cols[c] = {
                "min": float(mn) if mn is not None else None,
                "max": float(mx) if mx is not None else None,
                "avg": float(avg) if avg is not None else None,
                "ndistinct": int(nd) if nd is not None else 0,
            }
        profile["tables"][t] = {"row_count": row_count}
        profile["columns"][t] = cols
        profile.setdefault("coltypes", {})[t] = {
            c["name"]: str(c.get("type", "")) for c in table_schemas.get(t, [])}

        identifiers, measures = [], []
        for c, st in cols.items():
            if st["min"] is None:
                continue
            if (st["ndistinct"] >= max(1, int(row_count * 0.95))
                    and abs(st["min"]) <= 1
                    and st["max"] <= max(row_count * 1.5, 10)):
                identifiers.append(c)
            else:
                measures.append(c)
        # PKs from the schema are identifiers by definition.
        for c in table_schemas.get(t, []):
            if c.get("primary_key") and c["name"] not in identifiers \
                    and any(c["name"] == k for k in cols):
                identifiers.append(c["name"])
        profile["identifiers"][t] = sorted(set(identifiers))
        profile["measures"][t] = [m for m in measures
                                  if m not in identifiers]

    # Edge multiplicity evidence from row-count ratios.
    children_of = {}
    for fk in foreign_keys:
        c, p = fk.get("table"), fk.get("references_table")
        if c in profile["tables"] and p in profile["tables"]:
            children_of.setdefault(p, set()).add(c)

    for parent, kids in children_of.items():
        prow = profile["tables"][parent]["row_count"]
        for child in kids:
            crow = profile["tables"][child]["row_count"]
            ratio = (crow / prow) if prow else 0.0
            profile["edges"].setdefault(child, {})[parent] = {
                "ratio": round(ratio, 3),
                "confirmed_1N": ratio >= 1.5,
            }
    return profile


# ---------------------------------------------------------------------------
# Derived queries over a stored profile
# ---------------------------------------------------------------------------

def confirmed_children(profile: dict, table: str):
    """Direct children of `table` whose edge is CONFIRMED 1:N."""
    out = []
    for child, edges in profile.get("edges", {}).items():
        e = edges.get(table)
        if e and e.get("confirmed_1N"):
            out.append(child)
    return out


def event_depths(profile: dict, foreign_keys: list) -> dict:
    """Longest chain of confirmed-1:N detail levels above each table.
    Roots (no confirmed children) sit at 0; their direct confirmed children
    at 1; grand-children at 2 ..."""
    depths = {t: 0 for t in profile.get("tables", {})}
    changed = True
    while changed:
        changed = False
        for child, edges in profile.get("edges", {}).items():
            best_parent = -1
            for parent, e in edges.items():
                if e.get("confirmed_1N") and parent in depths:
                    best_parent = max(best_parent, depths[parent])
            if best_parent >= 0 and depths.get(child, 0) != best_parent + 1 \
                    and best_parent + 1 > depths.get(child, 0):
                depths[child] = best_parent + 1
                changed = True
    return depths


def measures_in_subtree(profile: dict, foreign_keys: list, root: str):
    """Numeric measure columns available within `root`'s subtree (root plus
    its confirmed descendants)."""
    ch = {}
    for fk in foreign_keys:
        c, p = fk.get("table"), fk.get("references_table")
        ch.setdefault(p, set()).add(c)
    seen, frontier = {root}, [root]
    while frontier:
        cur = frontier.pop()
        for k in ch.get(cur, ()):
            if k not in seen:
                seen.add(k)
                frontier.append(k)
    cols = []
    for t in sorted(seen):
        cols += [(t, m) for m in profile.get("measures", {}).get(t, [])]
    return cols


def base_row_count(profile: dict, table: str):
    return profile.get("tables", {}).get(table, {}).get("row_count")


# ---------------------------------------------------------------------------
# Execution-based validation (uses the profile)
# ---------------------------------------------------------------------------

class ExecutionSanityError(ValueError):
    """Raised when an executed result violates statistically-impossible
    bounds derived from the schema profile."""


def validate_execution_result(result: dict, sql: str, profile: dict,
                              table_schemas: dict, dialect: str = "",
                              sqlglot_mod=None) -> None:
    """Statistical sanity gate between execution and answer formatting.
    Raises ExecutionSanityError on impossible results; silent when the SQL
    or profile can't support a check (fail open)."""
    if not profile or not result:
        return
    rows = result.get("rows", [])
    if len(rows) != 1 or len(rows[0]) != 1:
        return  # multi-row/multi-col results: per-row checks live elsewhere
    value = rows[0][0]
    if value is None:
        return
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return

    if sqlglot_mod is None:
        return
    try:
        ast = sqlglot_mod.parse_one(sql, read=dialect or None)
    except Exception:
        return
    if ast is None or not isinstance(ast, exp.Select) or ast.args.get("group"):
        return

    sums = []
    avgs = []
    counts = []
    for a in ast.find_all(exp.Sum):
        sums.append(a)
    for a in ast.find_all(exp.Avg):
        avgs.append(a)
    for a in ast.find_all(exp.Count):
        counts.append(a)
    if sum(len(x) for x in (sums, avgs, counts)) == 0:
        return

    aliases = {}
    for t in ast.find_all(exp.Table):
        aliases[t.alias or t.name] = t.name

    def _col_stats(node):
        cols = list(node.find_all(exp.Column))
        if not cols:
            return None
        c = cols[0]
        t = aliases.get(c.table, c.table)
        st = profile.get("columns", {}).get(t, {}).get(c.name)
        return st

    def _resolve_first_col(node):
        cols = list(node.find_all(exp.Column))
        if not cols:
            return None
        c = cols[0]
        t = aliases.get(c.table, c.table)
        st = profile.get("columns", {}).get(t, {}).get(c.name)
        return t, st

    # SUM over a non-negative measure cannot be negative.
    if sums:
        node = sums[0].this
        if node is not None and not isinstance(node, exp.Star):
            info = _resolve_first_col(node)
            if info:
                st = info[1]
                if st and st["min"] is not None and st["min"] >= 0 \
                        and value_f < 0:
                    raise ExecutionSanityError(
                        f"Impossible result: SUM over non-negative column "
                        f"(profile min={st['min']}) returned {value_f}.")

    # AVG must lie within the column's profiled [min, max].
    if avgs:
        node = avgs[0].this
        if node is not None and not isinstance(node, exp.Star):
            info = _resolve_first_col(node)
            if info:
                st = info[1]
                if st and st["min"] is not None and st["max"] is not None \
                        and not (st["min"] - 1e-9 <= value_f <= st["max"] + 1e-9):
                    raise ExecutionSanityError(
                        f"Impossible AVG: {value_f} outside profiled "
                        f"column range [{st['min']}, {st['max']}].")

    # COUNT can never exceed the database's total row count, nor be
    # larger than the biggest single table when it targets one table.
    if counts:
        total_rows = sum(v.get("row_count", 0)
                         for v in profile.get("tables", {}).values())
        if value_f < 0 or (total_rows and value_f > total_rows):
            raise ExecutionSanityError(
                f"Impossible COUNT: {value_f} exceeds every table's row "
                f"count ({total_rows}).")


def is_money_capable(profile: dict, table: str, column: str) -> bool:
    """DECIMAL-family (non-integer) columns can carry currency."""
    t = profile.get("coltypes", {}).get(table, {}).get(column, "")
    return str(t).upper().startswith(("NUMERIC", "DECIMAL", "DEC", "REAL",
                                      "FLOA", "DOUBLE", "MONEY"))


def money_columns_in_subtree(profile: dict, foreign_keys: list, root: str):
    """ALL decimal-family columns under root's subtree — including ones the
    identifier heuristic claimed (Invoice.Total is near-unique yet is
    exactly the stored revenue measure we care about)."""
    ch = {}
    for fk in foreign_keys:
        c_, p_ = fk.get("table"), fk.get("references_table")
        ch.setdefault(p_, set()).add(c_)
    seen, frontier = {root}, [root]
    while frontier:
        cur = frontier.pop()
        for k in ch.get(cur, ()):
            if k not in seen:
                seen.add(k)
                frontier.append(k)
    out = []
    for t in sorted(seen):
        for cname, ctype in profile.get("coltypes", {}).get(t, {}).items():
            if is_money_capable(profile, t, cname):
                out.append((t, cname))
    return out


def range_note_for_answer(value_f: float, profile: dict, table: str,
                          column: str):
    """Optional annotation helper: returns a human-readable profile-range
    string for a figure, or None when unknown. Used by answer verification
    diagnostics."""
    st = profile.get("columns", {}).get(table, {}).get(column)
    if not st or st["min"] is None:
        return None
    return (f"profile range for {table}.{column}: "
            f"[{st['min']}, {st['max']}] over "
            f"{base_row_count(profile, table)} rows")
