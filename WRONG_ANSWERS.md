# WRONG_ANSWERS.md — Live-answer quality log

Every live agent run gets recorded here **after each test**: one row per
question in the ledger (§1), and a detailed entry (§2) for any answer that
is not fully correct — wrong figure, wrong shape, hallucination, silent
drift, crash, or retry storm. Correct-but-notable observations go in §3.
This file is append-only history; fixes get documented in
[PROJECT_HANDOFF.md](PROJECT_HANDOFF.md) §3 when they land, and the
detailed entry here gains a "FIXED" pointer then.

Test invocation convention:

```bash
python core_agent.py --db-url sqlite:///chinook.db --no-cache --metrics "<question>"
```

---

## 1. Ledger (every tested question, newest batch last)

| # | Date | Question | Agent answer | Gold | Verdict | Class | Retries |
|---|------|----------|--------------|------|---------|-------|---------|
| T-01 | 2026-08-25 | How many customers are there? | 59 | 59 | CORRECT | — | 0 |
| T-02 | 2026-08-25 | How many invoices exist? | 412 | 412 | CORRECT | — | 0 |
| T-03 | 2026-08-25 | How many tracks are there? | 3503 | 3503 | CORRECT | — | 0 |
| T-04 | 2026-08-25 | How many artists are there? | 275 | 275 | CORRECT | — | 0 |
| T-05 | 2026-08-25 | How many customers are from Brazil? | 5 | 5 | CORRECT | — | 0 |
| T-06 | 2026-08-25 | How many invoices were issued in 2010? | 0 | 0 | CORRECT | — | 0 |
| T-07 | 2026-08-25 | List customers from Germany | "There are 4 customers from Germany." | 4 rows (names below) | **WRONG SHAPE** | *(uncaught — no class assigned)* | 0 |
| T-08 | 2026-08-25 | What is the average track price? | $1.0395535714285713 | 1.0508050242649156 | **WRONG FIGURE** | GRAIN_ERROR-family, *uncaught* | 1 |
| T-09 | 2026-08-25 | What is the maximum track duration? | 5,286,953 ms | 5286953 ms | CORRECT (after 1 format-retry) | VERIFICATION_ERROR caught & self-healed | 0 |
| T-10 | 2026-08-25 | What is the minimum invoice total? | $0.99 | 0.99 | CORRECT | — | 0 |
| T-11 | 2026-08-25 | List customer names with their invoice totals | top-20-by-total listing (~61 customers), verified=False banner | 412 (name, total) pairs | **WRONG SCOPE** + verifier false-positives | ROW_ATTRIBUTION_ERROR ×15 (+false positives) | 0 |
| T-12 | 2026-08-25 | Show all tracks with their album names | meta-confusion text ("There is no question to answer…") | 3503 (track, album) rows | **WRONG — unusable answer** | *(uncaught — verified=True vacuously)* | 0 |
| T-13 | 2026-08-25 | Which artists have albums? | "204 artists have albums." | 204 distinct artists WITH albums (but question asked WHICH) | **WRONG SHAPE** (W-001 recurrence) | *(uncaught)* | 0 |
| T-14 | 2026-08-25 | Which artist has the most tracks? | Iron Maiden, 213 tracks | Iron Maiden, 213 | CORRECT | — | 1 |
| T-15 | 2026-08-25 | Which artist generated the most revenue? | "Iron Maiden generated **$48,900.60**" | Iron Maiden ✓, revenue **$138.60** | **WRONG FIGURE (~353× inflated)** | JOIN/fan-out via disconnected component, *uncaught* | 1 |
| T-16 | 2026-08-25 | Which genre has the most tracks? | Rock, 1297 tracks | Rock, 1297 | CORRECT | — | 0 |
| T-17 | 2026-08-25 | Which genre generated the most revenue? | **HARD FAILURE after 3 attempts** (exit 1) | Rock, $826.65 | **FALSE-REJECTION LOOP** | plan-table claim killed 3 good attempts | 3/3 |
| T-18 | 2026-08-25 | Total revenue from all invoices | $20,848.62 | $2,328.60 (`SUM(Invoice.Total)`) | **WRONG FIGURE (~8.95× inflated)** | fan-out via parent×child arithmetic, *uncaught* | 0 |
| T-19 | 2026-08-25 | Total revenue from all invoice lines | 2328.6 | 2328.6 | CORRECT (1 retry forced by W-007-class rejection) | — | 1 |

## 1a. Post-fix full sweep (2026-08-25, after all ten fixes landed)

Same 19 questions re-run end-to-end against chinook.db / llama3.2:

| # | Question | Result | vs before |
|---|----------|--------|-----------|
| R-01–05 | counts (customers/invoices/tracks/artists/Brazil) | 59/412/3503/275/5 ✓ | unchanged correct |
| R-06 | invoices in 2010 | 0 ✓ | unchanged correct |
| R-07 | List customers from Germany | **hard fail ×3** — attempt 1 COUNT correctly rejected by intent_error; attempts 2–3 died on genuine model hallucinations (ambiguous CustomerId → invented BillingAddress table). All rejections CORRECT; llama3.2 capability-bound | was silent-wrong COUNT shipped verified |
| R-08 | average track price | **1.0508050242649156 via `AVG(UnitPrice) FROM Track`** ✓ | was 1.0396 weighted |
| R-09 | max track duration | 5286953 ms via deterministic preview ✓ (formatter's ms→minutes conversion habit now trips the sanity gate → honest raw-value fallback) | was hallucinated figures then raw value |
| R-10 | min invoice total | $0.99 ✓ | unchanged |
| R-11 | customer names + invoice totals | **412 rows — FULL listing**, preview, verified ✓ | was silent top-20 drift |
| R-12 | tracks + album names | 500-row MCP-capped honest preview ✓ | was meta-garbage |
| R-13 | which artists have albums | 204-row listing ✓ (IDs not names — cosmetic) | was scalar COUNT |
| R-14 | artist with most tracks | Iron Maiden, 213 ✓ (1 retry) | unchanged |
| R-15 | artist revenue | **Iron Maiden $138.60 ✓** (1 retry; connectivity repair auto-appends missing FK edge) | was $48,900.60 (~353×) |
| R-16 | genre with most tracks | Rock, 1297 ✓ | unchanged |
| R-17 | genre revenue | **Rock $826.65 ✓** | was exit-1 hard failure |
| R-18 | total invoice revenue | **$2,328.60 via `SUM(Total)`** ✓ | was $20,848.62 (~9×) |
| R-19 | invoice-lines revenue | $2,328.60 ✓ final run (an earlier variant answered SUM(Quantity)=2240 — generator nondeterminism; money/integer guard did not fire that path, see OPEN item below) | was correct |

**Sweep score: 17/19 fully correct; R-07 capability-bound (all rejections
legitimate); R-19 flaky across runs.**

OPEN items logged for next session:
1. **Magnitude gate** (needs profile data): one structurally-valid SQL
   variant produced $485k artist revenue during fix tuning and escaped all
   validators — result-scale bound vs profiled column max × row count is
   the missing generic net.
2. **Money-vocabulary integer guard gap**: `SUM(Quantity)` survived
   "...revenue..." phrasing once; validate_metric_intent's integer-measure
   guard needs to key off question money-terms regardless of plan metric.

---

Gold cross-checks were taken from direct read-only SQL against the same
DB file in the same session (`sqlite3.connect('file:chinook.db?mode=ro')`),
not from memory.

---

## 2. Detailed entries (wrong / problematic answers)

### W-001 · "List customers from Germany" answered with a COUNT

| Field | Value |
|---|---|
| Date | 2026-08-25 14:33 |
| DB | `chinook.db` (SQLite) |
| Model | llama3.2 (default), Ollama, default budget, `--no-cache` |
| Question | List customers from Germany |
| SQL used | `SELECT COUNT(*) FROM Customer WHERE Country = 'Germany'` |
| Agent answer | "There are 4 customers from Germany." |
| Ground truth | Leonie Köhler, Hannah Schneider, Niklas Schröder, Fynn Zimmermann — COUNT(*) = 4 |
| Verdict | Wrong answer SHAPE: asked to enumerate entities, returned a scalar aggregate |
| Failure class reported by pipeline | NONE (no layer flagged it) |
| Metrics | 3 LLM calls, 4 tool calls, 0 retries, 0 semantic rejections, verified=True, 3.54 s |

**What happened:** the generator produced an aggregate where enumeration
was requested. The number itself was right, so flat figure verification
passed ("4" traces to the real result row), and nothing else in the
pipeline compares the *shape* of the request to the *shape* of the SQL.

**Why every layer missed it (gap analysis):**

- `validate_grouping_intent` catches the inverse error ("average X per
  customer" without GROUP BY) — there is no counterpart requiring entity
  rows when the question asks to LIST.
- Plan layer carries `{tables, metric, metric_column, entity, ranking,
  grouping}` — no `result_shape` field, so plan-vs-SQL checks had nothing
  to enforce.
- Answer verification checks figures and entity↔figure binding (§6.26);
  a bare count with no entity claims is trivially self-consistent.

**Impact severity:** low here (count happened to be right); high in
general — same drift on a filtered/ordered listing would silently return
a scalar and the user would never see the rows.

**Candidate countermeasure (NOT implemented yet):**
(a) add `result_shape: list|scalar` to the Step-2 merged plan call;
(b) when plan says `list` and top-level SELECT contains an aggregate over
zero GROUP BY columns, reject as actionable semantic error (retry path);
(c) cheap lexical fallback independent of plan noise: question verb
list/enumerate/name/show-all → expect ≥1 non-aggregate select item.
Risk to manage: false positives on legitimate "how many" phrasings that
mention "list" collaterally — gate on the noun being plural/entity-typed.

**Status:** FIXED — fix #7 intent_error / result_shape + lexical fallback (HANDOFF §6.29).

### W-002 · "Average track price" answered with a sales-weighted average

| Field | Value |
|---|---|
| Date | 2026-08-25 14:40 |
| DB | `chinook.db` (SQLite) |
| Model | llama3.2 (default), Ollama, `--no-cache` |
| Question | What is the average track price? |
| SQL used (final) | `SELECT AVG(T.UnitPrice) FROM InvoiceLine IL JOIN Track T ON IL.TrackId = T.TrackId` |
| Agent answer | $1.0395535714285713 |
| Ground truth | `SELECT AVG(UnitPrice) FROM Track` = **1.0508050242649156** |
| Verdict | WRONG FIGURE: answered the average *line-item* price (each sale counted), not the average price of the 3503 tracks |
| Failure class reported by pipeline | NONE on final attempt (attempt 1 was rejected, but for a different reason) |
| Metrics | 4 LLM calls, 6 tool calls, 1 semantic rejection, 1 retry, verified=True, 16.09 s |

**The number in detail:** AVG over the 2240 InvoiceLine rows joined to
Track = 1.03955… exactly equals the agent's answer; per-track average =
1.05080…. The join changed nothing about grain — one row per invoice line
survives, so popular tracks are weighted by how often they were sold.
A sales-frequency-weighted average is a *different statistic* than the
question asked for.

**Journey irony (important):** the plan-vs-SQL table check DID fire on
attempt 1 — *"plan identifies 'Track' as the main table … never appears in
FROM/JOIN (query touches: ['InvoiceLine'])"* — but the actionable message
pushed toward ADDING Track to the query, and the model's retry added it as
a JOIN while keeping InvoiceLine as FROM-root. The correct repair was the
opposite shape: `AVG(UnitPrice) FROM Track` alone. The validator's signal
was right; its suggested fix led away from the right answer. Attempt diff:
JOIN_ADDED+AGGREGATE_CHANGE.

**Why every layer missed it (gap analysis):**

- `validate_aggregation_fanout` targets INFLATION (SUM/COUNT multiplied by
  child rows). AVG is immune to inflation — it stays bounded and plausible
  — so the fan-out detector is structurally blind to this bias.
- Execution validation checks statistical impossibility (negative sums,
  out-of-range averages); 1.0396 is inside Track.UnitPrice's profiled range
  [0.99, 1.99], so nothing trips.
- Answer verification passed because every figure traced to real result
  rows — tracing ≠ correctness (same lesson as §6.26).
- The plan's main-table claim was consumed only as "planned table must
  APPEAR somewhere", not as "aggregation should be computed OVER the
  planned table's rows".

**Impact severity:** medium-high — silently biased, perfectly plausible,
fully verified wrong number on any AVG/MAX/MIN-over-dimension question
phrased against an entity ("average track X", "typical customer Y").

**Candidate countermeasure (NOT implemented yet):**
(a) sharpen the plan-table message: when the plan names ONE main entity
table and the question is a scalar aggregate over that entity's attribute,
reject aggregates whose FROM-root is NOT the planned table (the current
message already half-says this); (b) profile-based cross-check:
`schema_profile.py` v4 already stores per-column avg — a post-execute gate
could flag single-column AVG results deviating >X% from the profiled column
avg when the queried column is unambiguous (risky: legitimate filtered
averages deviate too — needs a filter-awareness guard or wide tolerance);
(c) cheapest: when plan.entity == a dimension table and no GROUP BY/ranking
exists, prefer prompting retry with "aggregate directly over <Table>
without joining transactional tables".

**Status:** FIXED (grain anchor) — fix #8 wrong_grain (HANDOFF §6.29); profiled-avg cross-check idea remains open.

### W-003 · Unrequested TOP-20 injected into a "list everything" question (+ verifier false positives)

| Field | Value |
|---|---|
| Date | 2026-08-25 14:45 |
| DB | `chinook.db` (SQLite) |
| Question | List customer names with their invoice totals |
| SQL used (final) | `…RANK() OVER (ORDER BY Total DESC) AS __rnk … WHERE __rnk <= 20 ORDER BY __rnk` (tie-aware rewrite of a model-generated `LIMIT 20`) |
| Agent answer | Numbered top-20 customers by their LARGEST invoice total + "multiple entities tied at 13.86" group (~42 names); ends with the honest unverified banner |
| Ground truth | **412** (name, total) pairs — every customer's every invoice (e.g. Leonie Köhler: 1.98, 1.98, 3.96, 5.94, 8.91, 13.86) |
| Verdict | WRONG SCOPE: silently became "top 20 invoices by amount" — no ranking or limit was requested |
| Failure classes | ROW_ATTRIBUTION_ERROR ×15; verified=False after corrective retry |
| Metrics | 4 LLM calls, 5 tool calls, 0 retries, 0 semantic rejections, 27.59 s |

**Failure stack (three distinct problems in one answer):**

1. **Silent scope drift (the big one).** Plan carried NO ranking, yet the
   generator emitted `LIMIT 20` and the tie-aware RANK() rewrite
   legitimized it. No validator enforces "plan has no ranking → SQL may
   not add one" (existing checks only enforce direction/LIMIT when the
   plan DOES rank). The RANK() rewrite then dressed an arbitrary cutoff as
   principled tie-handling.
2. **Attribution checker FALSE POSITIVES.** Of the 15 flagged bindings,
   most are list INDICES read as figures: "2 is cited for 'Helena Holý'"
   — the `2.` of the next list line fell into Helena's positional segment
   and was parsed as her claiming the figure "2". §6.15 taught flat
   verification to strip list markers; `verify_row_attribution`'s segment
   ownership apparently does NOT strip them before figure extraction.
3. **One REAL binding miss:** 'Johannes Van der Berg' cited as 'Johannes
   Van' (model truncated his name) → partial-name match failed → flagged
   as entity-not-in-rows. His actual figures were correct.

**What worked:** Helena's 25.86, Johannes' 13.86 etc. are genuine values;
the final banner honestly reported unverifiability instead of shipping
silently.

**Candidate countermeasures (NOT implemented):**
(a) reject LIMIT/top-N when plan.ranking is absent AND question lacks any
N-vocabulary ("top", "first", N) — actionable retry message;
(b) strip leading list markers (`^\s*\d+[.)]\s`) per segment before figure
extraction in segment ownership — mirrors the §6.15 fix;
(c) prefix/partial entity matching with uniqueness check ('Johannes Van'
uniquely prefixes exactly one result entity → bind, else flag);
(d) large-listing policy: when row count exceeds a threshold, either page
or say "showing first K of N rows" explicitly — never reframe as ranked.

**Status:** PARTIALLY FIXED — (b) list markers = fix #1 verifier_noise; (a) unsolicited LIMIT = fix #6 scope_drift; (c) prefix binding = fix #9 entity_binding; (d) large-listing policy = fix #5 format_error (HANDOFF §6.29).

### W-004 · Formatter collapse on a 3503-row listing

| Field | Value |
|---|---|
| Date | 2026-08-25 14:45 |
| DB | `chinook.db` (SQLite) |
| Question | Show all tracks with their album names |
| SQL used | `SELECT T.Name, A.Title FROM Track T INNER JOIN Album A ON T.AlbumId = A.AlbumId` (CORRECT, 3503 rows) |
| Agent answer | *"There is no question to answer directly and concisely with key numbers. The provided text appears to be a large list of song titles…"* |
| Verdict | WRONG: execution fine, formatting stage produced meta-confusion instead of the listing |
| Failure class | NONE reported — **answer verified=True vacuously** (no numeric claims → nothing to trace) |
| Metrics | 3 LLM calls, 5 tool calls, 7.73 s |

**Root cause:** `format_answer` was fed all 3503 rows and asked for a
concise answer with key numbers; llama3.2 drowned and replied to the
*prompt's framing* ("be concise") rather than the user's request. No
chunking/truncation/paging strategy exists for large listings.

**Why verification passed:** the figure-tracing checker only rejects
numbers that DON'T trace; an answer containing zero numbers can't fail it.
A vacuous pass — same tracing-≠-correctness lesson as W-002.

**Candidate countermeasure (NOT implemented):** size-aware formatting:
when row count > threshold (e.g. 50), skip the summarizing LLM call and
render a deterministic table preview + explicit total-count sentence
("3503 tracks across 347 albums — showing first 50"), optionally offer
paging/filtering. Also add a formatter sanity gate: if the formatted text
contains no token from ANY result cell AND mentions none of the requested
columns, treat as format failure and retry once deterministically.

**Status:** FIXED — fix #5 format_error size-aware preview + sanity gate (HANDOFF §6.29).

### W-005 · "Which artists have albums?" answered with a COUNT (W-001 recurrence)

| Field | Value |
|---|---|
| Date | 2026-08-25 14:46 |
| Question | Which artists have albums? |
| SQL used | `SELECT COUNT(DISTINCT ArtistId) FROM Album` |
| Agent answer | "204 artists have albums." |
| Gold | 204 distinct artists DO have albums (of 275), but the question asks WHICH — i.e. the names |
| Verdict | WRONG SHAPE — second occurrence of the W-001 class |

**Significance:** confirms W-001 is systematic, not a one-off: both
occurrences are "which/list X" → scalar COUNT, both fully verified, zero
rejections. Strengthens candidate countermeasure (a) from W-001: a
`result_shape` expectation on interrogative-listing questions ("which",
"list", "show me", "what are the names").

**Status:** FIXED — folds into W-001 fix (#7) (HANDOFF §6.29).

### W-006 · Revenue inflated ~353× by a DISCONNECTED join component (verified "True")

| Field | Value |
|---|---|
| Date | 2026-08-25 14:50 |
| DB | `chinook.db` (SQLite) |
| Question | Which artist generated the most revenue? |
| SQL used (final) | `…FROM Artist A JOIN Album ON A.ArtistId=Album.ArtistId JOIN InvoiceLine IL ON IL.InvoiceId=Invoice.InvoiceId JOIN Track T ON IL.TrackId=T.TrackId JOIN Invoice ON IL.InvoiceId=Invoice.InvoiceId GROUP BY A.Name` |
| Agent answer | "Iron Maiden generated $48,900.60 in revenue." |
| Ground truth | Iron Maiden ✓ (artist right), revenue **$138.60** (`SUM(IL.UnitPrice*IL.Quantity)` via Album⋈Track) |
| Verdict | WRONG FIGURE: ~353× inflation; **every join predicate is a real FK column, but the query graph has TWO components** — {Artist, Album} never connects to {InvoiceLine, Invoice, Track} because `T.AlbumId = Album.AlbumId` is missing → full cross-product between the halves |
| Failure class reported | EXECUTION_ERROR + JOIN_ERROR on attempt 1 (alias bug); final attempt NONE |
| Metrics | 4 LLM calls, 10 tool calls, 1 retry, 1 semantic rejection, verified=True, 8.41 s |

**Why every layer missed it (gap analysis):**

- `validate_join_semantics` (§6.2) was built for HALLUCINATED predicates
  (fake FK pairs). Here all four ON-conditions are genuine FK columns —
  what's missing is an EDGE, leaving two internally-consistent
  components. No connected-components check exists.
- `repair_missing_joins` (§6.5) adds chains for tables in NO join at all;
  every table here participates in some join.
- Fan-out detector v2 missed it twice over: the measure
  `SUM(IL.Quantity * T.UnitPrice)` is exactly the detail-side shape §6.23c
  declared immune, and the broken path may defeat lineage reasoning that
  assumes declared-FK connectivity.
- Answer verification passed trivially: $48,900.60 IS the returned value.
  Tracing ≠ correctness (third occurrence of this lesson).

**Impact severity:** HIGH — plausible-looking entity with an absurd but
unflagged figure; same question family as T-17 which failed for the
OPPOSITE reason (over-validation), showing the validator set is
simultaneously too strict and too blind.

**Candidate countermeasures (NOT implemented):**
(a) connected-components check over the query's join graph using fetched
FKs: >1 component among touched tables → reject with actionable message
naming candidate missing edges (here: Album.AlbumId=T.AlbumId);
(b) magnitude sanity gate: profile layer (§3.2) already stores per-column
avg/max — flag SUM results exceeding row_count × column max by orders of
magnitude;
(c) cheap heuristic: an aggregate over N tables requires ≥ N−1 DISTINCT
join edges connecting them.

**Status:** FIXED — fix #3 missing_join connected-components check (HANDOFF §6.29); magnitude gate idea remains open.

### W-007 · Plan-table claim false-rejection loop kills "genre revenue" 3× (exit 1)

| Field | Value |
|---|---|
| Date | 2026-08-25 14:50 |
| DB | `chinook.db` (SQLite) |
| Question | Which genre generated the most revenue? |
| SQL used | attempts all of form `SELECT G.Name, SUM(IL.Quantity*T.UnitPrice) … FROM Genre G JOIN Track T … JOIN InvoiceLine IL … GROUP BY G.Name ORDER BY … DESC` — **structurally correct and complete** |
| Agent answer | none — `Failed after 3 attempt(s)` hard failure, exit code 1 |
| Ground truth | Rock, $826.65 |
| Verdict | PIPELINE FAILURE: correct SQL rejected 3× on *"plan identifies 'Invoice' as the main table … never appears in FROM/JOIN"* |
| Failure classes | plan-table rejection ×3 |
| Metrics | 3 retries consumed, no answer |

**Root cause:** money vocabulary drove Step-2 planning to name Invoice as
the main entity table. The plan-table enforcement (a §6.7-era rule) then
demanded Invoice appear in FROM/JOIN. But revenue for genres lives in
InvoiceLine lines — joining Invoice would be either useless or fan-out-
prone. The model correctly refused to add it and died on the same message
three times. This is the §6.24a class ("hallucinated plan claims kill good
attempts") surviving on a DIFFERENT plan field: metric claims got
corroboration gates after §6.24a/§6.28 — the MAIN-TABLE claim never did.

**The damning contrast:** T-15 (same question about ARTISTS) succeeded —
its plan happened not to demand Invoice. Identical semantics, opposite
outcomes, decided by plan noise.

**Candidate countermeasure (NOT implemented):** apply the established
corroboration doctrine to plan.tables[0]: enforce main-table presence only
when the claim is corroborated (question lexically references that table's
domain, or the table carries the measure column the question needs);
otherwise demote to warning. Alternatively: when rejection repeats ≥2×
with the SAME message and the attempted SQL already aggregates the
planned-measure family, auto-downgrade to pass-with-warning (bounded
stubbornness guard).

**Status:** FIXED — fix #2 overvalidation corroboration + stubbornness guard (HANDOFF §6.29).

### W-008 · "Total revenue from all invoices" = SUM(Total × Quantity) → 8.95× inflation

| Field | Value |
|---|---|
| Date | 2026-08-25 14:54 |
| DB | `chinook.db` (SQLite) |
| Question | Total revenue from all invoices |
| SQL used | `SELECT SUM(INVOICE.Total * I.Quantity) FROM InvoiceLine I JOIN Invoice INVOICE ON I.InvoiceId=INVOICE.InvoiceId JOIN Track T ON I.TrackId=T.TrackId` |
| Agent answer | $20,848.62 — verified=True, 0 retries |
| Ground truth | **$2,328.60** (`SELECT SUM(Total) FROM Invoice`) |
| Verdict | WRONG FIGURE: each invoice's Total counted once per line item AND multiplied by quantity (~2240 line rows × qty weights vs 412 invoices → 8.95× inflation) |

**Mechanism:** the fan-out detector v2 (§6.20/§6.23c) catches a BARE
ancestor-column aggregate (`SUM(Invoice.Total)` with child joined — its
exact founding case). Here wrapping it in arithmetic — `Total * Quantity`,
mixing parent column with child measure — disguised it as detail-side
line-grain arithmetic, the exact shape §6.23c declared immune. The
immunity rule was drawn along syntax lines (expression vs bare column)
and the model walked straight through that seam. The Track join is dead
weight but harmless.

**Notable:** this is the §6.20 founding failure returning through a
loophole in its own countermeasure; fourth instance of tracing-≠-
correctness.

**Candidate countermeasure (NOT implemented):** lineage-aware expression
analysis: inside a SUM expression, ANY reference to a column from a table
that is an FK-ANCESTOR of the FROM-root's grain (here Invoice is the
parent of FROM-root InvoiceLine) should disqualify the detail-side
immunity and trigger grouped-grain/DISTINCT scrutiny regardless of
expression complexity. I.e., immunity must be per-COLUMN-LINEAGE, not
per-expression-shape.

**Status:** FIXED — double net: fix #4 fanout_seam lineage immunity + fix #10 semantic_error measure-expression guard (HANDOFF §6.29).

---

## 3. Correct answers with notable behavior

- **T-19 (total revenue from invoice lines = 2328.6, correct):** attempt 1
  was hit by the W-007-class false rejection ("plan identifies 'Invoice'
  as the main table…") — the model escaped by adding a redundant but
  benign `JOIN Invoice` (child→parent, no fan-out). W-007's validator bug
  is therefore confirmed RECURRING; T-17 only died because there adding
  Invoice was wrong/awkward. Also: optimizer detected the
  SUM(UnitPrice×Quantity) recompute pattern but has no learned equivalent
  yet (`--learn-measures` was off) — with it on, this query is exactly the
  §3.1 rewrite target.
- **T-09 (max track duration):** first `format_answer` hallucinated "5" and
  "28" (likely a botched ms→minutes conversion attempt) — figure
  verification CAUGHT it ("numbers not found in the query results: 5, 28"),
  corrective retry produced the clean raw-milliseconds answer. VERIFICATION_ERROR
  logged; final answer exact (5,286,953 ms ≈ 88.1 min, track
  'Occupation / Precipice'). The §6.6 countermeasure working as designed.
- **T-06 (invoices in 2010 = 0):** genuinely zero — this chinook.db copy's
  InvoiceDate range is **2021-01-01 → 2025-12-22**, NOT the classic Chinook
  2009–2013. Any gold set copied from upstream Chinook docs will be wrong
  for date questions on this fixture. Flagged for benchmark/gold maintenance.
- **T-01–T-05, T-07 counts:** happy path throughout — 3 LLM calls + 4 tool
  calls, zero retries, first-call cold-start latency ~12 s then ~3.3–3.9 s.

---

## Maintenance rules

1. After EVERY live test run, append/update rows in §1 (next free T-nn).
2. Any non-correct verdict gets a full §2 entry (W-nnn) the moment it is
   observed — before any fix discussion, so the observation stays honest.
3. When a fix lands, mark the §2 entry `Status: FIXED (see HANDOFF §x.yy)`
   and leave the original text untouched.
4. Never edit old entries except to add the FIXED pointer; corrections to
   gold values are new footnotes, not rewrites.
