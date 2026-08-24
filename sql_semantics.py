"""
sql_semantics.py — The semantic layer beneath syntax: table-grain inference,
question-to-grain mapping, and attempt-to-attempt semantic diffing.

Everything here is DETERMINISTIC and LLM-free. It answers three questions
the syntactic pipeline cannot:

1. What does one row of each table MEAN (its grain)? Derived purely from FK
   topology: tables that reference lookups but are referenced by no other
   event-family table are event roots (Invoice = one per purchase document);
   leaf tables referenced by events are detail lines (InvoiceLine); pure-
   referenced tables are dimensions (Customer); FK-less tables standalone.

2. What grain does the QUESTION ask for, and what grain does the SQL count?
   "Purchases" about a Customer means invoice-level events — counting
   InvoiceLine rows is the live failure this layer exists to kill.

3. What actually CHANGED between two SQL attempts? A structural diff with
   human-readable tags (MEASURE/AGGREGATE_CHANGE, JOIN_DROPPED, ...) so
   retries become observable instead of opaque string swaps.

Also home to root_and_detail_side(): the single shared definition of "which
side of the join am I on", consumed by both the fan-out detector (core_agent)
and the count-grain validator here.
"""

import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

AGG_TYPES = (exp.Sum, exp.Avg, exp.Count, exp.Min, exp.Max)


def _stem(token: str) -> str:
    """Symmetric plural stripper: drop ONE trailing 's' when it leaves a
    non-trivial stem. Applied to BOTH table names and question tokens, so
    'invoices'/'invoice' collapse to one key without a morphology engine."""
    t = token.lower()
    if t.endswith("s") and len(t) > 3:
        return t[:-1]
    return t


@dataclass
class GrainInfo:
    role: str            # event_root | detail | dimension | standalone
    depth: int           # 0 = root event; +1 per detail level below it
    pk_cols: list = field(default_factory=list)
    numeric_measures: list = field(default_factory=list)

    @property
    def is_eventish(self) -> bool:
        return self.role in ("event_root", "detail")


def classify_grains(table_schemas: dict, foreign_keys: list) -> dict:
    """Role + depth per table, derived from FK topology alone.

    KNOWN LIMITATION (documented, deliberate): global dimension-typing is
    undecidable from FK shape alone — Chinook's Customer references Employee,
    which makes it indistinguishable topologically from an event. Roles here
    are therefore heuristic; the DECISION-CRITICAL paths (count-grain checks)
    use descendants()/plan_entity reachability instead, which IS sound.

    event_root : references others AND is referenced by others (mid-spine or
                 party-hub — includes parties that reference lookups)
    detail     : referenced below some other event/detail node, references
                 others — line grain
    dimension  : pure lookup — referenced by others, references nothing
    standalone : no declared relationships at all
    depth      : longest event-edge chain above the node (roots sit at 0 of
                 their own subtree); used for "shallower document" ordering
    """
    tables = set(table_schemas)
    parents_of = {t: set() for t in tables}
    children_of = {t: set() for t in tables}
    for fk in foreign_keys:
        c, p = fk.get("table"), fk.get("references_table")
        if c in tables and p in tables and c != p:
            parents_of[c].add(p)
            children_of[p].add(c)

    info = {}
    for t in tables:
        cols = table_schemas[t]
        pk = [c["name"] for c in cols if c.get("primary_key")]
        measures = [c["name"] for c in cols
                    if str(c.get("type", "")).upper().startswith(
                        ("NUMERIC", "DECIMAL", "DEC", "REAL", "FLOA",
                         "DOUBLE", "MONEY"))
                    and not c.get("primary_key")]
        if not parents_of[t] and not children_of[t]:
            role = "standalone"
        elif not parents_of[t]:
            role = "dimension"
        elif not children_of[t]:
            role = "detail"
        else:
            role = "event_root"
        info[t] = GrainInfo(role=role, depth=0, pk_cols=pk,
                            numeric_measures=measures)

    # Depth fixpoint along event edges (child deeper than its parents).
    changed = True
    while changed:
        changed = False
        for child in tables:
            if not info[child].is_eventish:
                continue
            for parent in parents_of[child]:
                if not info[parent].is_eventish:
                    continue
                candidate = info[parent].depth + 1
                if candidate > info[child].depth:
                    info[child].depth = candidate
                    changed = True
    return info


def _flat(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def descendants(table: str, fk_children: dict) -> set:
    """All tables transitively BELOW `table` given {parent: set(children)}."""
    seen = set()
    frontier = [table]
    while frontier:
        cur = frontier.pop()
        for ch in fk_children.get(cur, ()):
            if ch not in seen:
                seen.add(ch)
                frontier.append(ch)
    return seen


def build_fk_maps(foreign_keys: list):
    """(fk_adjacency, fk_children): child->parents and parent->children."""
    adj, ch = {}, {}
    for fk in foreign_keys:
        c, p = fk.get("table"), fk.get("references_table")
        adj.setdefault(c, set()).add(p)
        ch.setdefault(p, set()).add(c)
    return adj, ch


# Words that mark the question as being ABOUT ACTIVITY on top of the entity
# (as opposed to merely listing/counting the entities themselves). Tiny by
# design; the fallback ordering below degrades gracefully either way.
_ACTIVITY_RE = re.compile(
    r"\b(?:made|placed|bought|purchased|ordered|spent|paid|sold|shipped)\b"
    r"|\bhow many\b|\btotal\b|\bsum\b|\bper\b",
    re.IGNORECASE,
)


def infer_question_grain(question: str, plan_entity, grains: dict,
                         fk_adjacency=None):
    """Expected grain table for COUNT/list-style questions.

    Returns (table_name, GrainInfo) or None when there's no signal.
    Resolution order:
      0. Entity named AND no activity language ('List the customers')
         -> the entity itself is the unit.
      1. A DIFFERENT eventish table named explicitly ('count invoices')
         -> that table is the unit.
      2. Activity language + dimension-ish/hub entity -> its document
         stream: shallowest FK-child that has dependencies of its own
         (Customer -> Invoice), excluding pure detail leaves.
      3. Dimension-name direct hit, else None.
    fk_adjacency: {child_table: set(parent_tables)} — required for rule 2.
    """
    tokens = {_stem(t) for t in re.findall(r"[A-Za-z]+", question.lower())}
    fk_children = invert(fk_adjacency) if fk_children_available(fk_adjacency) else {}

    def _name_tokens(t):
        return {_stem(x) for x in re.findall(r"[A-Za-z]+", t.lower())}

    activity = bool(_ACTIVITY_RE.search(question))
    entity_stem_hit = bool(plan_entity and _stem(plan_entity) in tokens)

    # Underscore-insensitive normalization: flatten BOTH sides so a question
    # saying "order_lines" contains the same character run as table
    # "order_lines" ("orderlines"), while plain-plural tables ("orders")
    # never accidentally match that compound run. Tier A (flattened
    # substring) outranks Tier B (underscore-fragment token overlap).
    # Flattened per-token set: underscores glue a compound question token
    # into one unit ('order_lines' -> 'orderline') that whole-name tier A
    # can match EXACTLY, so plain-plural 'orders' (whose flat form 'order'
    # is merely a PREFIX of 'orderline') can never steal the match.
    qtok_flat = {_stem(re.sub(r"[^a-z0-9]", "", t))
                 for t in re.findall(r"[a-z0-9_]+", question.lower())}

    def _match_score(table_name):
        """Lexical strength for one table against the question.
        (matched_part_count, exact_compound_hit, -specificity):
          - matched PART count leads — a question naming BOTH words of
            'order_items' targets that table even when a shorter table
            ('orders') merely contains the single word 'order';
          - an exact flattened-compound hit breaks ties;
          - among equal evidence the MORE specific name wins;"""
        flat = _stem(re.sub(r"[^a-z0-9]", "", table_name.lower()))
        substr = bool(flat) and flat in qtok_flat
        parts = {_stem(x) for x in re.findall(r"[A-Za-z]+", table_name.lower())}
        matched = parts & tokens
        return len(matched), (1 if substr else 0), -len(parts)

    verb_activity = bool(re.search(
        r"\b(?:made|placed|bought|purchased|ordered|spent|paid|sold|"
        r"shipped|received)\b", question, re.IGNORECASE))

    event_direct = []
    dim_direct = []
    entity_stem = _stem(str(plan_entity)) if plan_entity else None
    for t, gi in grains.items():
        score = _match_score(t)
        if score <= (0, 0, 0):
            continue
        matched_parts = score[0]
        # Rule 1 guard: a table matching ONLY the entity's own stem is the
        # entity-as-unit case (rule 0 / fallback), never an activity target.
        if plan_entity and matched_parts == 1 and entity_stem in                 {_stem(x) for x in re.findall(r"[A-Za-z]+", t.lower())} \
                and t == plan_entity:
            continue
        if gi.is_eventish and t != plan_entity:
            event_direct.append((score, t, gi))
        elif gi.role == "dimension":
            dim_direct.append((score, t, gi))

    if event_direct:
        top = max(score for score, _, _ in event_direct)
        best = [(t, gi) for score, t, gi in event_direct if score == top]
        return min(best, key=lambda tg: tg[1].depth)

    # Rule 2: an explicit activity VERB over a hub entity -> its document
    # stream ("made the most purchases" about Customer -> Invoice), never a
    # bare detail leaf (a count of documents is what such questions mean).
    if plan_entity and fk_children and verb_activity:
        docs = [t for t in fk_children
                if plan_entity in fk_adjacency.get(t, ())
                and grains.get(t, GrainInfo("x", 0)).is_eventish
                and fk_children.get(t)]
        if docs:
            best = min(docs, key=lambda t: (grains[t].depth, t))
            return best, grains[best]

    # Entity named as its own unit without any other table signal.
    if plan_entity and fk_children and entity_stem_hit:
        return plan_entity, grains[plan_entity]

    if dim_direct:
        top = max(score for score, _, _ in dim_direct)
        best = [(t, gi) for score, t, gi in dim_direct if score == top]
        return min(best, key=lambda tg: tg[1].depth)
    return None


def fk_children_available(fk_adjacency):
    return isinstance(fk_adjacency, dict)


def invert(adj):
    """{child: parents} -> {parent: set(children)}"""
    inv = {}
    for child, ps in adj.items():
        for p in ps:
            inv.setdefault(p, set()).add(child)
    return inv


# ---------------------------------------------------------------------------
# FROM-root / detail-side classification (shared with fan-out detection)
# ---------------------------------------------------------------------------

def root_and_detail_side(ast: exp.Expression, foreign_keys: list,
                         known: set):
    """Returns (base, lineage, below_root) for the query's FROM structure:
    base = FROM-root table name (or None); lineage = base + every FK-ancestor
    reachable within `known`; below_root = queried tables outside that
    lineage (the detail side, whose rows scale measures by line count)."""
    query_tables = {t.name for t in ast.find_all(exp.Table) if t.name in known}
    parents_of = {}
    for fk in foreign_keys:
        c, p = fk.get("table"), fk.get("references_table")
        if c in known and p in known:
            parents_of.setdefault(c, set()).add(p)

    from_node = ast.args.get("from_") or ast.args.get("from")
    base = None
    if from_node:
        first = next(iter(from_node.find_all(exp.Table)), None)
        base = first.name if first and first.name in known else None

    lineage = set()
    if base:
        lineage.add(base)
        frontier = [base]
        while frontier:
            cur = frontier.pop()
            for p in parents_of.get(cur, ()):
                if p not in lineage:
                    lineage.add(p)
                    frontier.append(p)
    return base, lineage, query_tables - lineage


# ---------------------------------------------------------------------------
# Graph-first measure/grain resolution (v3)
# ---------------------------------------------------------------------------

def resolve_measure_source(question, plan, table_schemas, foreign_keys,
                           profile):
    """Graph+stats resolution of WHERE a question's measure lives.

    Priority (no keyword matching on the primary path):
      1. plan.metric_column anchors the source: its owning table among the
         fetched schema (must be one of the plan's picked tables when the
         plan named them).
      2. plan.entity anchors the document stream: shallowest-depth member
         of (plan.tables ∩ descendants(entity)) for SUM/AVG; for COUNT the
         SHALLOWEST such stream is the document unit ('purchases' ->
         Invoice), deeper members are lines.
      3. Nothing usable -> {'weak': True}; caller falls back to the legacy
         lexical validators (demoted last resort).

    Returns {'source': table|None, 'op': str|None, 'money': bool,
             'weak': bool}.
    """
    plan = plan or {}
    known_tables = set(profile.get("tables", {}))
    coltypes = profile.get("coltypes", {})
    tables_c = [t for t in (plan.get("tables") or []) if t in known_tables]
    entity = plan.get("entity")
    if entity not in known_tables:
        entity = None
    metric = (plan.get("metric") or "").upper() or None

    # Anchor 1: planned metric_column's unique owner among known tables.
    mc = plan.get("metric_column")
    if mc:
        mc_l = str(mc).lower()
        owners = [t for t in known_tables
                  if any(c.lower() == mc_l for c in coltypes.get(t, {}))]
        owners_in_plan = [t for t in owners if t in tables_c] or owners
        if len(owners_in_plan) == 1:
            src = owners_in_plan[0]
            money = is_money_type(coltypes, src, str(mc))
            return {"source": src, "op": metric, "money": money,
                    "weak": False}

    if not entity and not tables_c:
        return {"source": None, "op": metric, "money": False, "weak": True}

    adj, chm = build_fk_maps(foreign_keys)
    depths = event_depths_from_children(chm, known_tables)

    pool = [t for t in tables_c]
    if entity:
        desc = descendants(entity, chm)
        scoped = [t for t in pool if t == entity or t in desc]
        pool = scoped or pool

    if not pool:
        return {"source": None, "op": metric, "money": False, "weak": True}

    def _depth(t):
        return depths.get(t, 0)

    def _eventish(t):
        edges = profile.get("edges", {})
        for _p, e in edges.get(t, {}).items():
            if e.get("confirmed_1N"):
                return True
        for _c, es in edges.items():
            e = es.get(t)
            if e and e.get("confirmed_1N"):
                return True
        return False

    if metric == "COUNT":
        # Document unit = shallowest event-family member of the pool
        # (event-family = endpoint of a CONFIRMED 1:N edge).
        events = [t for t in pool if _eventish(t)]
        src = min(events or pool, key=_depth)
    elif metric in ("SUM", "AVG", "MIN", "MAX"):
        money_src = [t for t in pool
                     if any(is_money_type(coltypes, t, c)
                            for c in coltypes.get(t, {}))]
        src = max(money_src or pool, key=_depth)
    else:
        src = min(pool, key=_depth)

    money = metric in ("SUM", "AVG") and any(
        is_money_type(coltypes, t, c)
        for t in pool for c in coltypes.get(t, {}))

    # 'sold' requires the SALES lineage (Invoice/InvoiceLine family) to be
    # part of the data path — counting catalog rows instead of sales rows
    # was a live miss ('tracks sold' answered from Album->Track alone).
    sold_signal = bool(re.search(r"\bsold\b|\bsales\b", question,
                                 re.IGNORECASE))
    sales_parent = next((t for t in list(chm.keys())
                         if _flat(t) in ("invoice", "invoices")),
                        None)
    # When the question says SOLD/SALES, the measure source MUST be the
    # sales lineage's deepest fact (InvoiceLine), overriding any noun-
    # derived shallower source (e.g. 'tracks' -> Track would count catalog
    # rows instead of sold units — the live miss).
    if sold_signal and sales_parent:
        deepest = [t for t in chm.get(sales_parent, set())]
        deepest.append(sales_parent)
        src = max(deepest, key=lambda t: depths.get(t, 0)) if depths else src
        pool = [src]
    return {"source": src, "op": metric, "money": money, "weak": False,
            "requires_sales_lineage": sold_signal and sales_parent is not None,
            "sales_lineage_hint": ([sales_parent]
                                   + sorted(chm.get(sales_parent, set()))
                                   ) if sales_parent else []}


def is_money_type(coltypes, table, column):
    t = str(coltypes.get(table, {}).get(column, "")).upper()
    return t.startswith(("NUMERIC", "DECIMAL", "DEC", "REAL", "FLOA",
                         "DOUBLE", "MONEY"))


def event_depths_from_children(children_map, known_tables, max_rounds=12):
    """Longest-parent-chain depth. Self-referencing FKs (e.g.
    Employee.ReportsTo -> Employee) are ignored and a hard round-cap
    guarantees termination on cyclic graphs."""
    depths = {t: 0 for t in known_tables}
    for _ in range(max_rounds):
        changed = False
        for parent, kids in children_map.items():
            if parent not in depths:
                continue
            pd = depths[parent]
            for k in kids:
                if k == parent or k not in depths:
                    continue  # ignore self-references
                if depths[k] < pd + 1:
                    depths[k] = pd + 1
                    changed = True
        if not changed:
            break
    return depths


# ---------------------------------------------------------------------------
# Semantic diff
# ---------------------------------------------------------------------------

def _extract_shape(ast: exp.Expression, dialect: str) -> dict:
    joins = sorted({j.this.name for j in ast.find_all(exp.Join)
                    if isinstance(j.this, exp.Table)})
    aggregates = sorted(
        f"{type(a).__name__.upper()}({a.sql(dialect=dialect)})"
        for a in ast.find_all(*AGG_TYPES)
    )
    where = ast.args.get("where")
    group = ast.args.get("group")
    limit = ast.args.get("limit")
    order = ast.args.get("order")
    return {
        "joins": joins,
        "aggregates": aggregates,
        "where": where.sql(dialect=dialect) if where else "",
        "group": group.sql(dialect=dialect) if group else "",
        "limit": limit.sql(dialect=dialect) if limit else "",
        "order": order.sql(dialect=dialect) if order else "",
    }


def semantic_diff(sql_a: str, sql_b: str, dialect: str) -> dict:
    """Structural comparison of two SQL attempts.

    Returns {'changed': bool, 'tags': [...], 'details': {...}} with tags:
    AGGREGATE_CHANGE, JOIN_ADDED / JOIN_DROPPED, FILTER_CHANGED,
    GROUPING_CHANGED, LIMIT_CHANGED, ORDER_CHANGED, SYNTAX_ONLY (renders
    identical after parse), RENDER_ONLY (same shape, cosmetic differences),
    UNPARSEABLE."""
    try:
        ast_a = sqlglot.parse_one(sql_a, read=dialect or None)
        ast_b = sqlglot.parse_one(sql_b, read=dialect or None)
    except Exception:
        return {"changed": True, "tags": ["UNPARSEABLE"], "details": {}}

    sa, sb = _extract_shape(ast_a, dialect), _extract_shape(ast_b, dialect)
    tags, details = [], {}

    ja, jb = set(sa["joins"]), set(sb["joins"])
    if ja != jb:
        if jb - ja:
            tags.append("JOIN_ADDED")
        if ja - jb:
            tags.append("JOIN_DROPPED")
        details["joins"] = {"added": sorted(jb - ja),
                            "dropped": sorted(ja - jb)}
    if sa["aggregates"] != sb["aggregates"]:
        tags.append("AGGREGATE_CHANGE")
        details["aggregates"] = {"before": sa["aggregates"],
                                 "after": sb["aggregates"]}
    for key, tag in (("where", "FILTER_CHANGED"),
                     ("group", "GROUPING_CHANGED"),
                     ("limit", "LIMIT_CHANGED"),
                     ("order", "ORDER_CHANGED")):
        if sa[key] != sb[key]:
            tags.append(tag)
            details[key] = {"before": sa[key], "after": sb[key]}

    if tags:
        return {"changed": True, "tags": tags, "details": details}

    ra = ast_a.sql(dialect=dialect or None)
    rb = ast_b.sql(dialect=dialect or None)
    if ra == rb:
        return {"changed": False, "tags": ["SYNTAX_ONLY"], "details": {}}
    return {"changed": True, "tags": ["RENDER_ONLY"],
            "details": {"before": ra, "after": rb}}


def summarize_diff(diff: dict) -> str:
    if not diff.get("changed"):
        return "no change"
    return "+".join(diff.get("tags", []))
