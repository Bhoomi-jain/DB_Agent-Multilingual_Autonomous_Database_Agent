# PROJECT_MEMORY_LOG.md — Project memory

The running memory of this project's answer-quality work: every bug we
discovered, why it happened, what tools/methods were used, how it was
solved (or why not yet), how the fix was verified, how the system behaved
under test, and exactly what changed in the codebase as a result.

Companion files, each with a distinct job:
- [WRONG_ANSWERS.md](WRONG_ANSWERS.md) — per-test ledger + per-bug evidence
  (what happened; append-only)
- [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md) — architecture rationale + the
  §3 compendium where landed fixes are documented for posterity
- [failure_taxonomy.py](failure_taxonomy.py) — slug registry joining bugs ↔
  fixes ↔ runtime failure classes

**Update rules:** one new ML-nnn entry per discovered bug or shipped change,
written at the time of the event. Fields in every entry: DISCOVERED · WHY ·
USED · SOLVED/HOW · VERIFIED · SYSTEM BEHAVIOR · CHANGES. Never rewrite old
entries; corrections are appended footnotes.

---

## Session 2026-08-25 — live answer-quality audit (llama3.2 via Ollama, chinook.db)

Method for the whole session: run questions through `python core_agent.py
--db-url sqlite:///chinook.db --no-cache --metrics "<q>"`, then take golds
from direct read-only SQL (`sqlite3.connect('file:chinook.db?mode=ro')`)
in the same session — never from memory. 19 questions tested (T-01…T-19),
8 wrong/problematic answers (W-001…W-008), full evidence in WRONG_ANSWERS.md.

---

### ML-001 · Listing question answered with COUNT (intent drift)

- **DISCOVERED:** T-07 "List customers from Germany" → "There are 4
  customers from Germany." Names never returned. Recurred as T-13 "Which
  artists have albums?" → "204 artists have albums." (2/2 systematic).
- **WHY:** plan schema has no `result_shape` field; validators enforce
  aggregation-when-required but nothing requires *entity rows when listing
  is asked*; figure verification passes trivially on a bare count.
- **USED:** agent CLI runs w/ metrics; direct sqlite3 cross-check of names;
  code reading of validate_grouping_intent / plan-vs-SQL checks.
- **SOLVED:** OPEN → planned Fix #7 (`result_shape` in Step-2 plan +
  lexical fallback requiring ≥1 non-aggregate select item).
- **VERIFIED:** gold = 4 German customers / 204 artists-with-albums via
  direct SQL; counts right, shape wrong both times.
- **SYSTEM BEHAVIOR:** clean happy path both runs — 0 retries, 0 semantic
  rejections, verified=True; pipeline fully satisfied with the wrong shape.
- **CHANGES:** none yet (documented only).

### ML-002 · AVG computed over joined-detail grain (sales-weighted bias)

- **DISCOVERED:** T-08 "What is the average track price?" → $1.03955…
  (per-invoice-line average) instead of $1.05080… (per-track average).
- **WHY:** fan-out detector targets inflation (SUM/COUNT); AVG stays
  bounded so no layer flags frequency-weighted averages; plan-table check
  fired on attempt 1 but its message pushed toward ADDING Track as a join
  rather than dropping InvoiceLine.
- **USED:** CLI runs; direct SQL comparison of all three candidate
  statistics (per-line / join / per-track); attempt-diff telemetry.
- **SOLVED:** OPEN → planned Fix #8 (dimension-grain rule: scalar aggregate
  over planned dimension ⇒ FROM-root must be that table).
- **VERIFIED:** agent's number equals `AVG(IL.UnitPrice)` over 2240 lines
  to full float precision; truth equals `AVG(UnitPrice)` over 3503 tracks.
- **SYSTEM BEHAVIOR:** 1 semantic rejection + 1 retry that moved AWAY from
  the correct query; final attempt verified=True, zero flags.
- **CHANGES:** none yet.

### ML-003 · Unrequested TOP-20 injection + verifier false positives + name-binding miss

- **DISCOVERED:** T-11 "List customer names with their invoice totals"
  returned top-20-by-total (~61 customers) instead of 412 pairs; verifier
  flagged 15 bindings, most bogus.
- **WHY:** three stacked causes — (a) generator added LIMIT 20 with no
  ranking in plan and no validator forbidding unsolicited LIMIT; tie-aware
  RANK() rewrite legitimized it; (b) attribution segment ownership parses
  list indices ("2.", "3.") as cited figures — §6.15's marker stripping
  was never ported into verify_row_attribution; (c) model truncated
  'Johannes Van der Berg' to 'Johannes Van'; matcher lacks unique-prefix
  binding.
- **USED:** CLI run w/ metrics (27.6 s, ROW_ATTRIBUTION_ERROR ×15);
  direct SQL checks of Helena's/Johannes' real totals and rank≤20 size.
- **SOLVED:** OPEN → Fixes #1 (verifier_noise), #6 (scope_drift),
  #9 (entity_binding).
- **VERIFIED:** banner honestly reported unverifiability (good); figures
  spot-checked genuine (25.86, 13.86 correct).
- **SYSTEM BEHAVIOR:** 0 semantic rejections on the scope drift itself;
  verification storm hit AFTER the answer was already mis-scoped.
- **CHANGES:** none yet.

### ML-004 · Formatter collapse on 3503-row listing

- **DISCOVERED:** T-12 "Show all tracks with their album names" — SQL
  correct (3503 rows), format_answer produced meta-confusion ("There is no
  question to answer directly…").
- **WHY:** formatter fed all rows with a "be concise" prompt; llama3.2
  answered the prompt's framing, not the user. No size-aware strategy.
- **USED:** CLI run; metrics showed execution fine, formatting stage
  consumed the time.
- **SOLVED:** OPEN → planned Fix #5 (rows > K → deterministic preview +
  "first K of N"; sanity gate retry).
- **VERIFIED:** row count via direct SQL (3503); answer contained zero
  result tokens → vacuously verified=True (tracing ≠ correctness, again).
- **SYSTEM BEHAVIOR:** fully green pipeline shipping an unusable answer.
- **CHANGES:** none yet.

### ML-005 · Disconnected join components → 353× revenue inflation

- **DISCOVERED:** T-15 "Which artist generated the most revenue?" → Iron
  Maiden "$48,900.60" vs true $138.60.
- **WHY:** every ON-predicate used real FK columns but the graph had TWO
  components ({Artist,Album} ⋈̸ {InvoiceLine,Invoice,Track} — missing
  T.AlbumId=Album.AlbumId) → cross-product. No connected-components check
  exists; repair_missing_joins only handles fully-unjoined tables; measure
  looked like the protected detail-side shape.
- **USED:** CLI run; direct-SQL revenue ranking; manual join-graph audit.
- **SOLVED:** OPEN → planned Fix #3 (validate_join_connectivity union-find
  over touched tables vs FK graph, actionable edge candidates).
- **VERIFIED:** ground truth from `SUM(IL.UnitPrice*IL.Quantity)` grouped
  via Album⋈Track = $138.60; ratio ≈353× explained by component sizes.
- **SYSTEM BEHAVIOR:** attempt-1 EXECUTION_ERROR (alias bug) masked the
  deeper issue; final attempt verified=True on the inflated value.
- **CHANGES:** none yet.

### ML-006 · Plan-table claim kills correct SQL 3× (exit 1)

- **DISCOVERED:** T-17 "Which genre generated the most revenue?" — three
  structurally correct attempts all rejected: "plan identifies 'Invoice'
  as the main table … never appears in FROM/JOIN". Hard failure.
- **WHY:** money vocabulary drove planning to Invoice; main-table claim
  has NO corroboration gate (unlike metric claims since §6.24a/§6.28).
  Recurred benignly in T-19 where adding Invoice was harmless.
- **USED:** CLI run capturing identical rejection ×3; contrast with T-15's
  success on the same semantics.
- **SOLVED:** OPEN → planned Fix #2 (corroborate main-table claim before
  enforcing; same-message-≥2× stubbornness guard downgrades to warning).
- **VERIFIED:** correct gold (Rock, $826.65) computable by hand; rejected
  SQL was complete without Invoice.
- **SYSTEM BEHAVIOR:** bounded retry burned all attempts on ONE message;
  exit code 1, no answer.
- **CHANGES:** none yet.

### ML-007 · Double aggregation through the fanout-immunity seam

- **DISCOVERED:** T-18 "Total revenue from all invoices" → SUM(Total×
  Quantity) over invoice lines = $20,848.62 vs true $2,328.60 (8.95×).
- **WHY:** §6.20 detector catches BARE ancestor columns; wrapping in
  arithmetic (`Total * Quantity`) disguised it as the §6.23c-protected
  detail-arithmetic shape — immunity drawn along expression-syntax lines.
- **USED:** CLI run; direct SQL both forms; history review of §6.20/§6.23c.
- **SOLVED:** OPEN → Fixes #4 (lineage-based immunity: any ancestor-column
  operand disqualifies protection) + #10 (standalone double-aggregation
  guard grounded in measure-layer knowledge — defense in depth).
- **VERIFIED:** both statistics reproduced exactly via direct SQL.
- **SYSTEM BEHAVIOR:** zero retries, zero rejections, verified=True.
- **CHANGES:** none yet.

### ML-008 · Fixture date-range trap (not an agent bug)

- **DISCOVERED:** T-06 "Invoices in 2010?" → 0, correct here but any
  upstream-Chinook gold would say otherwise.
- **WHY:** bundled chinook.db has InvoiceDate 2021-01-01 → 2025-12-22.
- **VERIFIED:** MIN/MAX(InvoiceDate) via direct SQL.
- **CHANGES:** flagged for benchmark/gold maintenance; no agent change.

### ML-009 · What worked (keep doing this)

- **Figure verification save (T-09):** formatter hallucinated "5"/"28"
  (botched ms→minutes conversion); verify_answer caught untraceable
  numbers, corrective retry shipped exact 5,286,953 ms. §6.6 machinery
  working as designed.
- **Happy-path economics:** simple counts run 3 LLM calls + 4 tool calls,
  0 retries, ~3.3–3.9 s warm (12 s cold Ollama start).

### ML-010 · Audit → taxonomy → fix program (process entry)

- **DISCOVERED:** after 19 tests, pattern-level finding: the validator set
  is simultaneously too strict (ML-006) and too blind (ML-005/007), and
  figure tracing ≠ correctness appeared 4× (ML-002/004/005/007).
- **WHY / WHAT WE USED:** full read-through of WRONG_ANSWERS.md; pipeline
  anchor mapping (validate_* call sites in SQLAgent.run retry loop);
  severity triage C/A/H/M/X.
- **HOW SOLVED:** agreed 10-fix program, red-pin-first, executed in order
  1→2→3→4→10→5→6→7→8→9, no commits, plus user-mandated additions:
  failure-category registry and this memory log.
- **CHANGES SO FAR (Fix step 0, DONE):**
  - NEW `failure_taxonomy.py`: FAILURES registry (10 slugs: missing_join,
    wrong_grain, format_error, intent_error, semantic_error, overvalidation,
    fanout_seam, scope_drift, verifier_noise, entity_binding) binding each
    to bug label, fix numbers, runtime FailureClass, refs.
  - core_agent.py: FailureClass += PLAN_TABLE_REJECTION, RANKING_ERROR;
    classify_error resolves "[slug]" tags authoritatively before needle
    matching; SemanticValidationError grew `.category` (auto-derived from
    message tag; legacy raise sites unaffected).
- **VERIFIED:** smoke assertions (slug parse round-trip, class routing for
  all 10 slugs, legacy needle classification intact, category derivation)
  passed; FULL SUITE 22/22 after wiring.

---

## Next entries

ML-011 will record Fix #1 (verifier_noise) — discovery pin, change, suite
result. Thereafter one entry per fix as it lands.

---

### ML-011 · Fix #1 landed — verifier_noise: list indices no longer cited as figures

- **DISCOVERED:** T-11 live run flagged 15 entity↔figure bindings; most
  were bogus ("2 is cited for 'Helena Holý'"). Reproduced 1:1 with a
  synthetic 5-row result before touching any code.
- **WHY (true mechanism, subtler than first logged):**
  `_SENTENCE_SPLIT_RE` treats each list marker's dot as a sentence ender,
  so every entity's "sentence" ENDS with the next item's bare index
  (`"…25.86\n2."`); `_strip_list_markers` required `\s+` after the dot and
  missed it. The §6.15 stripping existed but its trailing-whitespace
  requirement was wrong for this shape.
- **USED:** empirical reproduction via `verify_row_attribution` on a
  synthetic result set (pure function — no FakeLLM needed); regex trace of
  split → strip → figure-extraction path.
- **HOW SOLVED:** one-character-class change — `_strip_list_markers` now
  accepts end-of-string after the marker: `(?m)^\s*\d+[.)](?:\s+|$)`.
- **VERIFIED:** new red-pin-first tests in `test_live_regressions.py`
  (RED confirmed pre-fix: indices 2/3/4 falsely cited; GREEN post-fix);
  negative control proves real misbindings (99.99) still flagged; FULL
  SUITE 23/23 (22 prior + new regression file).
- **SYSTEM BEHAVIOR:** attribution checker now silent on numbered listings;
  'Johannes Van' partial-name flag remains visible in reproduction — that
  is BUG-M3 / fix #9 territory, deliberately not touched here.
- **CHANGES:** `core_agent.py::_strip_list_markers` (+docstring); NEW
  `test_live_regressions.py` with two pins under slug `verifier_noise`.
  Status in WRONG_ANSWERS.md: W-003 partially addressed (sub-fix (b));
  (a)/(c)/(d) remain open under scope_drift/entity_binding/format_error.

## Next entries

ML-013 will record Fix #3 (missing_join). One entry per fix thereafter.

---

### ML-012 · Fix #2 landed — overvalidation: plan-table claim corroborated before it can kill

- **DISCOVERED:** T-17 "Which genre generated the most revenue?" died after
  3 IDENTICAL rejections ("plan identifies 'Invoice' as the main table…"),
  exit 1, on structurally correct SQL; benign recurrence in T-19.
- **WHY:** entity-presence rule was unconditional; metric claims gained a
  corroboration gate in §6.24a/§6.28 but the MAIN-TABLE claim never did —
  money vocabulary alone put 'Invoice' in the plan and three good attempts
  died on it.
- **USED:** pure-function red pin against validate_plan_matches_sql;
  existing-suite contract discovery (test_query_plan.py revealed the
  no-question default); retry-loop reading for guard wiring.
- **HOW SOLVED:** two independent mechanisms — (a) corroboration gate:
  enforce presence ONLY when the question lexically references the claimed
  table (singular/plural token match) OR when no question text exists
  (historical strictness preserved — caught by the legacy pin mid-fix);
  otherwise log-and-skip; rejection message now carries "[overvalidation]"
  + category; (b) bounded-stubbornness guard in run(): an IDENTICAL
  overvalidation rejection repeating across attempts is downgraded to a
  warning and the attempt accepted instead of burning every retry.
- **VERIFIED:** pins RED→GREEN (T-17 shape now passes untouched; positive
  control — corroborated claim with genuinely absent table — still
  rejected, classify_error routes to PLAN_TABLE_REJECTION); FULL SUITE
  23/23.
- **SYSTEM BEHAVIOR:** genre-revenue-class questions now survive planner
  noise on attempt 1; even a corroborated-but-impossible demand dies at
  worst once before the guard accepts.
- **CHANGES:** core_agent.py — validate_plan_matches_sql entity block
  (gate + tag), NEW _table_lexemes/_main_table_claim_corroborated,
  retry-loop except-branch guard + per-question `self._last_overvalidation`
  reset; FailureClass.PLAN_TABLE_REJECTION consumed via taxonomy.

## Next entries

ML-013 will record Fix #3 (missing_join). One entry per fix thereafter.

---

### ML-013 · Fix #3 landed — missing_join: disconnected join components rejected

- **DISCOVERED:** W-006/T-15 — 353× revenue inflation; every ON-predicate
  was a real FK pair but the query graph had two components ({Artist,
  Album} vs {InvoiceLine, Invoice, Track}; missing Album.AlbumId =
  Track.AlbumId) → cross-product, verified "True".
- **WHY:** validate_join_semantics checks each predicate in isolation;
  repair_missing_joins only rescues fully-unjoined tables; nothing checked
  GLOBAL connectivity.
- **USED:** sqlglot API probes (FROM lives under `from_` in this version —
  the §6.25/§6.27 trap, third sighting); union-find over outer-scope table
  nodes with FK-matched equi-predicates as edges.
- **HOW SOLVED:** NEW `validate_join_connectivity(sql, foreign_keys,
  dialect)` — nodes = real tables in outer FROM/JOINs (derived-table
  internals excluded); edges = ON *or WHERE* column=column predicates that
  match declared FKs either direction (WHERE scan kills the old-style
  comma-join false-positive class); >1 component → "[missing_join]"
  rejection naming candidate reconnecting FK edges from the schema.
  Wired at BOTH pipeline sites right after validate_join_semantics
  (initial + post-repair revalidation).
- **VERIFIED:** red pin rejects the exact W-006 SQL with an Album↔Track
  hint and routes to JOIN_ERROR via classify_error; negative controls:
  proper 4-hop path passes, single table passes, comma-join tolerated by
  design; FULL SUITE 23/23. One test-fixture bug caught during pinning
  (missing InvoiceLine→Track FK edge in my own fixture — the checker was
  right, the fixture was wrong).
- **SYSTEM BEHAVIOR:** cross-product shapes now die pre-execution with an
  actionable message instead of shipping inflated verified answers.
- **CHANGES:** core_agent.py — NEW validate_join_connectivity (~120
  lines); run() wiring ×2. test_live_regressions.py — missing_join pins.

## Next entries

ML-015 will record Fix #10 (semantic_error). One entry per fix thereafter.

---

### ML-014 · Fix #4 landed — fanout_seam: immunity now per-column-lineage

- **DISCOVERED (mechanism refined while pinning):** TWO shields had to
  fall for W-008: (1) the blanket detail-shape exemption (`any operand
  below root ⇒ immune`) hid parent×child mixes like SUM(Total×Quantity);
  (2) the "nothing joined below root" early-out ALSO shielded the minimal
  shape SUM(Parent.Total × Child.qty) where the FROM-root itself is the
  duplicating child — no third table needed.
- **WHY:** v2 immunity was decided per-AGGREGATE by expression shape;
  inflation is decided per-COLUMN by lineage.
- **USED:** empirical red pins BEFORE code (caught the second shield —
  first patch passed the live-shaped pin but not the minimal-form pin);
  sql_semantics.root_and_detail_side reading (lineage INCLUDES base).
- **HOW SOLVED:** `_fanout_findings` v3 — detail-side immunity requires
  ALL operands below_root (`not arg_tables & lineage`); removed the
  redundant `not below_root` early-out entirely (per-candidate
  `multiplying` FK-child check + `_grouped_at_grain` carry its protective
  weight exactly).
- **VERIFIED:** mixed-arithmetic pin RED→GREEN; controls green pre- AND
  post-fix (pure detail arithmetic immune, §6.20 founding bare-column
  case still flagged); FULL SUITE 23/23.
- **SYSTEM BEHAVIOR:** false-positive audit held: grouped-at-grain,
  MIN/MAX, DISTINCT, grouped-COUNT exemptions untouched.
- **CHANGES:** core_agent.py `_fanout_findings` (immunity condition +
  early-out removal + docstring v3); test_live_regressions.py —
  fanout_seam pins (+ file rebuilt cleanly after a mangled heredoc append).


---

### ML-015 · Fix #10 landed — semantic_error: standalone double-aggregation guard

- **DISCOVERED:** W-008/T-18 ($2,328.60 → $20,848.62 via SUM(Total×
  Quantity)); user explicitly required this as its OWN fix rather than a
  fanout-detector tweak — "SEMANTIC VALIDATION LAYER… Add as Fix #10".
- **WHY:** fix #4 repairs one detector's heuristic; a bug this expensive
  deserves an independent net grounded in measure semantics that fires
  even when join topology looks innocent.
- **USED:** same red-pin protocol; `_column_type` reuse for numeric/PK
  classification of the parent-side column.
- **HOW SOLVED:** NEW `validate_measure_expression(sql, table_schemas,
  foreign_keys, dialect)` — for every SUM/AVG expression containing
  columns from TWO tables in direct FK child↔parent relation, if the
  PARENT-side column is numeric non-PK → "[semantic_error]" rejection
  naming both correct forms (child-level revenue vs parent-level total).
  Wired in run() right after the fan-out block; reject+retry only, no
  deterministic repair (deliberate, documented choice).
- **VERIFIED:** pins RED→GREEN (W-008 shape flagged w/ correct category +
  AGGREGATION_ERROR routing; pure line arithmetic and bare SUM(Total)
  stay silent); FULL SUITE 23/23.
- **SYSTEM BEHAVIOR:** either #4 or #10 alone now stops the W-008 class;
  both together give defense-in-depth with distinct failure messages.
- **CHANGES:** core_agent.py — NEW validate_measure_expression; run()
  wiring after fanout block; test_live_regressions.py — semantic_error
  pins ×2. (One self-inflicted signature clip during editing repaired
  immediately.)

---

### ML-016 · Fix #5 landed — format_error: size-aware formatting + sanity gate

- **DISCOVERED:** W-004/T-12 — correct 3503-row listing, formatter replied
  with meta-confusion; verification passed vacuously (no numbers cited).
- **WHY:** no size strategy in format_answer; nothing checked whether the
  formatted answer references the result data at all.
- **USED:** pins with a forbidden-LLM stub (proves determinism) and a
  scripted-LLM stub (proves gate -> retry -> fallback ordering); live-suite
  reruns to catch fixture coupling.
- **HOW SOLVED:** rows > FORMAT_PREVIEW_ROW_LIMIT(50) never reach the LLM —
  deterministic preview ("N rows returned / Showing the first 20 / ...and M
  more"); small path gains _answer_touches_result sanity gate: an answer
  citing NO column/string-cell/numeric-cell gets ONE corrective re-invoke,
  then the deterministic preview ships instead of meta-garbage.
- **VERIFIED:** 3 new pins RED->GREEN; FULL SUITE journey exposed THREE
  latent issues, all resolved: (1) my token regex kept trailing punctuation
  ("$300." -> "300." != "300"); (2) SS6.16 again - PG NUMERIC arrives as
  STRING "300.00"; helper now number-parses string cells; (3) TWO pre-
  existing test fixtures shipped vacuously-detached answers the old
  pipeline silently blessed: metric_mismatch's scripted answer said Alice
  while its own gold row is Bob (49.99); grouping_intent's answer cited
  zero data. Both updated to reference real output; original assertions
  untouched and passing.
- **SYSTEM BEHAVIOR:** large listings are hallucination-proof by
  construction; detached small answers self-correct or degrade to an
  honest data table instead of shipping meta-text as verified=True.
- **CHANGES:** core_agent.py - FORMAT_PREVIEW_* constants,
  _render_result_preview, _answer_touches_result (+punctuation strip +
  string-number parsing), format_answer tail rewrite; test_live_
  regressions.py format_error pins x3; fixture answers fixed in
  test_metric_mismatch.py (Alice->Bob x6) & test_grouping_intent.py.

---

### ML-017 · Fix #6 landed — scope_drift: unsolicited LIMIT rejected

- **DISCOVERED:** W-003#1/T-11 — "List customers..." silently became
  top-20-by-total; every ranking check only fires WHEN the plan ranks,
  so an ADDED limit had no countermeasure at all.
- **WHY:** asymmetric enforcement — planned ranking enforced, unplanned
  ranking ignored; the RANK() tie-rewrite then legitimized the cutoff.
- **USED:** red pin on validate_unsolicited_limit; two suite collisions
  used as contract discovery for exemption vocabulary.
- **HOW SOLVED:** NEW validate_unsolicited_limit — rejects when plan
  ranking disabled AND top-level LIMIT present AND question lacks N-
  vocabulary AND lacks ANY superlative (most/highest/cheapest/... may
  legitimately end in LIMIT 1 even when the plan missed the ranking
  claim) AND the query is not an ungrouped scalar aggregate. Wired post
  plan-vs-SQL; tagged [scope_drift]; reject+retry only.
- **VERIFIED:** pins RED->GREEN (T-11 shape rejected; "Top 5 customers"
  passes); suite collisions fixed at ROOT cause: scenario F needed
  cheapest/expensive in the exemption regex (its LIMIT 1 legitimate);
  answer_verification's SPEND_SQL carried an unsolicited LIMIT 20 that
  contradicted its own "each customer" intent — removed from fixture.
  FULL SUITE 23/23.
- **SYSTEM BEHAVIOR:** full listings stay full; genuine top-N paths
  untouched; zero false positives after root-cause fixes rather than
  rule loosening.
- **CHANGES:** core_agent.py — _N_VOCAB_RE/_SUPERLATIVE_RE +
  validate_unsolicited_limit + run() wiring; fixture:
  test_answer_verification.py SPEND_SQL LIMIT removed.

## Next entries

ML-018 will record Fix #7 (intent_error). One entry per fix thereafter.

---

### ML-018 · Fix #7 landed — intent_error: listing questions can't be answered by COUNT

- **DISCOVERED:** W-001/T-07 + W-005/T-13 (2/2 systematic): "List customers
  from Germany" / "Which artists have albums?" answered with scalar COUNT,
  fully verified both times.
- **WHY:** plan had no result_shape field; every validator enforced
  aggregation-when-required but nothing required entity ROWS when
  enumeration was asked.
- **HOW SOLVED:** three coordinated pieces — (a) Step-2 prompt gains
  "result_shape": list|scalar|null; (b) parser stores it only when the
  question lexically corroborates it (§6.24a discipline), else records the
  lexical fallback; (c) NEW infer_result_shape(question) drives an
  enforcement block in validate_plan_matches_sql that runs BEFORE the
  empty-plan guard (question-driven, works even with {} / None plans):
  list-shape + aggregates + no GROUP BY + zero non-aggregate select items
  -> "[intent_error]" rejection. Exemptions: any aggregation-demand vocab
  (how many/most/total/average/extremes) flips expectation to scalar, so
  every count/superlative question stays untouched; grouped listings pass.
- **VERIFIED:** pins RED->GREEN covering W-005 and W-001 shapes plus
  controls (how-many phrasing passes; 'most tracks' LIMIT-1 passes;
  grouped spend listing passes); FULL SUITE 23/23. Two placement mistakes
  during wiring (block landed before ast parse; guard ordering broke
  {}-plan pins) caught by pins within seconds — §6.25 lesson re-earned.
- **SYSTEM BEHAVIOR:** Germany/artists-with-albums class now gets an
  actionable retry demanding entity rows instead of shipping a count.
- **CHANGES:** core_agent.py — _LIST_INTENT_RE/_AGG_DEMAND_RE,
  infer_result_shape, plan-prompt line, parse-side corroboration,
  enforcement block in validate_plan_matches_sql; test_live_regressions.py
  intent_error pins ×2.

## Next entries

ML-019 will record Fix #8 (wrong_grain). One entry per fix thereafter.

---

### ML-019 · Fix #8 landed — wrong_grain: dimension aggregates anchored to their entity table

- **DISCOVERED:** W-002/T-08 — AVG(UnitPrice) over InvoiceLine⋈Track =
  sales-frequency-weighted 1.0396 instead of per-track 1.0508; fan-out
  detector structurally blind because AVG biases without inflating.
- **WHY:** plan.entity was consumed only as "must APPEAR somewhere"
  (W-007's check), never as "compute OVER this entity's rows".
- **HOW SOLVED:** NEW validate_aggregation_grain — fires when plan.entity
  exists ∧ schema-known ∧ question corroborates it lexically ∧
  infer_result_shape==scalar ∧ no GROUP BY ∧ SUM/AVG/COUNT aggregate
  present ∧ FROM-root != entity ∧ an aggregated operand column resolves
  on the entity. Message instructs: aggregate directly over <Entity>.
  MIN/MAX exempt (duplication-idempotent); grouped shapes untouched.
  Wired post metric-intent; tagged [wrong_grain]; reject+retry only.
- **VERIFIED:** pins RED->GREEN (W-002 join shape rejected with Track
  named in message; direct AVG(UnitPrice) FROM Track passes;
  entity-less plan passes); FULL SUITE 23/23.
- **SYSTEM BEHAVIOR:** average/sum-over-dimension questions now get
  pushed toward the per-entity statistic on retry one instead of
  shipping a plausible weighted bias verified=True.
- **CHANGES:** core_agent.py — validate_aggregation_grain + run() wiring;
  test_live_regressions.py wrong_grain pins.

## Next entries

ML-020 will record Fix #9 (entity_binding). One entry per fix thereafter.

---

### ML-020 · Fix #9 landed — entity_binding: unique-prefix citations bind

- **DISCOVERED:** W-003#3/T-11 — model cited 'Johannes Van' for
  'Johannes Van der Berg'; figures were correct but the attribution layer
  flagged the entity as fabricated (full-cell matching only).
- **HOW SOLVED:** _cell_prefix helper (leading two-token window of 3+
  token names, length>=6 guard) wired into THREE matching surfaces:
  all_cells_norm index (fabricated-entity check), _sentence_entities
  (row resolution for tie-pick check), _entity_mentions (segment-
  ownership spans — prefix span maps back through idx_map so positional
  ownership still cuts correctly). Multi-row ambiguity guard preserved by
  existing set-of-rows semantics; pin proves shared prefixes refuse to
  bind while unique ones do.
- **VERIFIED:** pin RED->GREEN; verifier_noise + misbinding controls stay
  green; FULL SUITE 23/23.
- **SYSTEM BEHAVIOR:** truncated-but-unambiguous name citations now bind
  silently; genuinely ambiguous or fabricated names still flagged.
- **CHANGES:** core_agent.py — _cell_prefix + three match-site extensions;
  test_live_regressions.py entity_binding pins.

---

## Session close-out — ALL TEN FIXES SHIPPED

| # | slug | status |
|---|------|--------|
| 1 | verifier_noise | LANDED |
| 2 | overvalidation | LANDED |
| 3 | missing_join | LANDED |
| 4 | fanout_seam | LANDED |
| 10 | semantic_error | LANDED |
| 5 | format_error | LANDED |
| 6 | scope_drift | LANDED |
| 7 | intent_error | LANDED |
| 8 | wrong_grain | LANDED |
| 9 | entity_binding | LANDED |

Suite: 23/23 (22 original + test_live_regressions.py). No commits made,
per instruction.

### Victory-lap rerun (same questions that originally failed)

| Question | Before fixes | After fixes |
|---|---|---|
| List customers from Germany | silent wrong COUNT, verified | attempts now correctly REJECTED when wrong (intent_error fired on COUNT; later attempts died on genuine model hallucinations — ambiguous CustomerId, invented BillingAddress table). No silent wrongness; llama3.2 capability-bound |
| Which artists have albums? | scalar COUNT, verified | **204-row listing** (shape fixed; selects IDs not names — cosmetic) |
| What is the average track price? | 1.0396 weighted, verified | **1.0508050242649156 via `AVG(UnitPrice) FROM Track`** — exact per-entity statistic |
| Which artist generated the most revenue? | $48,900.60 (~353x), verified | **Iron Maiden, $138.60** — exact (connectivity check rejected broken SQL, repair auto-appended Album.AlbumId=Track.AlbumId, semantic guard then steered to line-grain SUM) |
| Total revenue from all invoices | $20,848.62 (~9x), verified | **$2,328.60** — exact |
| Show all tracks with their album names | meta-garbage, verified vacuously | deterministic preview: honest row count + first 20 + "...and more" |

Post-victory-lap refinements (caught live, pinned after):
- #4 mixed-expression immunity refined AGAIN: lineage operand disqualifies
  only when it is a STORED MEASURE name (_STORED_MEASURE_NAMES via
  _is_stored_measure_name) — raw attributes (UnitPrice x Quantity) are the
  canonical correct revenue shape and stay immune;
- validate_measure_expression shares the same stored-measure name test
  (its first cut flagged Track.UnitPrice — raw attribute — as a measure);
- validate_aggregation_grain gains lexical entity inference when the plan
  omits entity (unique schema-table lexeme match), which is what anchors
  AVG onto Track despite planner silence.

Final suite state: 23/23. Live behavior: every previously-silent wrong
answer is now either exact, honestly degraded, or loudly rejected.
