# Project roadmap

This is the ordered work plan for taking DB-Agent from a tested beta to a
reliable production product. Work should proceed top-to-bottom unless a
security or operational incident requires otherwise.

## Current baseline

- Explicit SQL control loop is implemented in `core_agent.py`.
- Read-only AST-validated MCP database access is implemented in
  `db_mcp_server.py`.
- Multilingual and hybrid SQL/RAG entry points are implemented.
- FastAPI API and browser UI are implemented in `api.py` and `static/index.html`.
- Docker deployment with local Ollama is implemented.
- The regression suite currently passes 23/23 tests.
- The current production path is read-only and single-database per process.
- The free local LLM path uses Ollama with `llama3.2:latest`.

## Priority 0 — production safety gate

These items must be complete before exposing the service to real users or
real customer data.

### P0.1 Create a production deployment contract

- [x] Document supported deployment targets: local Docker, single VM, and
  managed container platform.
- [x] Define supported database engines and minimum versions.
- [x] Define CPU, RAM, disk, and optional GPU requirements for Ollama.
- [x] Define expected latency, concurrency, timeout, and request-size limits.
- [ ] Define the incident owner and rollback procedure.

**Acceptance criteria**

- `DEPLOYMENT.md` contains one tested command path for the selected target.
- A new operator can deploy without guessing a database URL, port, model, or
  secret.
- The documented rollback procedure restores the previous application image.

### P0.2 Enforce database least privilege

- [ ] Create a dedicated database user with SELECT-only permissions.
- [ ] Restrict the database network to the application host or private network.
- [ ] Verify that DML and DDL fail at both the MCP AST layer and database role
  layer.
- [ ] Document credential rotation.
- [ ] Ensure no client request can provide or override `DATABASE_URL`.

**Acceptance criteria**

- A production database credential cannot INSERT, UPDATE, DELETE, DROP, ALTER,
  or CREATE.
- A request payload contains only a question and cannot select another tenant
  or database.
- A credential rotation can be performed without changing source code.

### P0.3 Put the service behind HTTPS and an identity boundary

- [x] Put Nginx, Caddy, or a managed HTTPS load balancer in front of FastAPI.
- [x] Disable direct public access to Uvicorn and Ollama in the HTTPS Compose mode.
- [ ] Replace the bootstrap bearer token with an identity-aware proxy or
  integrate a real authentication provider.
- [ ] Add rate limiting per user or identity.
- [x] Add request correlation IDs.

**Acceptance criteria**

- Public traffic uses HTTPS only.
- Ollama is reachable only by the private application network.
- Unauthenticated and over-limit requests are rejected before reaching the LLM.
- Every request can be traced from access log to agent log.

### P0.4 Make readiness meaningful

- [x] Update `/health/ready` to verify the configured LLM provider is reachable.
- [x] Verify that the configured Ollama model exists.
- [x] Verify database connectivity and a harmless metadata query.
- [x] Return readiness failure when any required dependency is unavailable.
- [x] Keep `/health/live` independent of external dependencies.

**Acceptance criteria**

- The service is not reported ready when Ollama is down.
- The service is not reported ready when the database is unavailable.
- Load balancers can safely use `/health/live` and `/health/ready`.

## Priority 1 — continuous integration and release protection

### P1.1 Add GitHub Actions CI

- [ ] Create `.github/workflows/ci.yml`.
- [ ] Run on pushes and pull requests.
- [ ] Use Python 3.11, matching the supported runtime.
- [ ] Install from `uv.lock`.
- [ ] Run syntax compilation.
- [ ] Run the full `python run_tests.py` suite.
- [ ] Upload `test_report.txt` as a workflow artifact.
- [ ] Build the production Docker image.
- [ ] Validate `docker-compose.production.yml`.

**Acceptance criteria**

- A pull request cannot pass when the regression suite fails.
- CI starts the required database services automatically.
- CI seeds fixtures before running database tests.
- CI reports PostgreSQL, MySQL/MariaDB, and SQLite results separately.
- The Docker image build is part of the required check.

### P1.2 Add database-matrix jobs

- [ ] PostgreSQL service job.
- [ ] MySQL or MariaDB service job.
- [ ] SQLite job using the committed fixture-generation path.
- [ ] Export all DSNs through environment variables.
- [ ] Avoid hardcoded passwords outside ephemeral CI service configuration.
- [ ] Preserve setup failures separately from application failures.

**Acceptance criteria**

- The same commands documented in `SETUP_TESTS.md` work in CI.
- A database service failure is clearly distinguished from a test failure.
- The CI matrix catches dialect-specific SQL and MCP behavior regressions.

### P1.3 Add release checks

- [ ] Add a versioned application image tag.
- [ ] Add a smoke test that starts the image and checks `/health/live`.
- [ ] Add a protected release branch or required status checks.
- [ ] Publish an image only after CI passes.
- [ ] Keep a previous image available for rollback.

**Acceptance criteria**

- Every deployable image is tied to a commit.
- A failed health check prevents promotion.
- Rollback does not require rebuilding the application.

## Priority 2 — model-quality correctness

These items address the remaining live-model weaknesses documented in
`WRONG_ANSWERS.md` and `PROJECT_HANDOFF.md`.

### P2.1 Add a magnitude sanity gate

- [ ] Use `schema_profile.py` statistics to estimate plausible result bounds.
- [ ] Detect scalar and grouped values that exceed safe bounds derived from
  profiled column maxima, row counts, and aggregate type.
- [ ] Make the gate lineage-aware so legitimate detail-side arithmetic is not
  rejected.
- [ ] Reject and retry with an actionable message.
- [ ] Add regression tests for inflated joins and valid high-magnitude totals.

**Acceptance criteria**

- The known artist-revenue inflation case is rejected or repaired.
- Valid revenue and detail-line totals remain accepted.
- The gate never silently changes the query or answer.

### P2.2 Add filter-aware tolerance

- [ ] Track whether a query is filtered, grouped, ranked, or scalar.
- [ ] Avoid comparing filtered aggregates directly to whole-column profile
  averages.
- [ ] Use profile checks only when the query shape supports a valid comparison.
- [ ] Add tests for unfiltered, filtered, grouped, and empty-result queries.

**Acceptance criteria**

- The average-track-price case is validated against the correct grain.
- A legitimate filtered average is not rejected because it differs from the
  whole-table average.
- A wrong-grain average remains detectable.

### P2.3 Close the money-vocabulary integer guard gap

- [ ] Treat revenue, sales, spending, cost, price, and monetary synonyms as
  money intent regardless of noisy or missing plan metadata.
- [ ] Reject integer-only measures such as `SUM(Quantity)` when the question
  asks for money and no valid money measure is present.
- [ ] Preserve fail-open behavior when the schema has no corroborating money
  column.
- [ ] Add scripted and live regression tests.

**Acceptance criteria**

- “What is the revenue?” cannot silently become `SUM(Quantity)`.
- Correct quantity questions remain valid.
- The retry message identifies the expected monetary measure family.

### P2.4 Improve multilingual numeric integrity

- [ ] Preserve numeric digits in translated answers where possible.
- [ ] Accept localized decimal and thousands separators during verification.
- [ ] Avoid appending an English warning when the translated number is
  semantically equivalent but formatted differently.
- [ ] Add tests for Spanish, French, German, Hindi, and Arabic numeric output.

**Acceptance criteria**

- A translated answer containing “eight” or a localized numeric word is not
  incorrectly treated as a missing figure when the meaning is verified.
- Decimal values survive translation without silent changes.
- The English fallback appears only when numeric integrity cannot be established.

### P2.5 Expand the live benchmark

- [ ] Increase the Chinook gold set beyond the current 14 questions.
- [ ] Add listing, filtered aggregate, grouped aggregate, ranking, tie, empty
  result, multilingual, and semantic-search cases.
- [ ] Run the benchmark against `llama3.2`, `qwen3:4b`, and any selected
  production model.
- [ ] Store before/after results and failure classes.
- [ ] Define minimum release thresholds for execution accuracy,
  hallucination rate, retry rate, and verification failures.

**Acceptance criteria**

- A model change cannot ship without benchmark comparison.
- Every known failure class has at least one gold regression case.
- Threshold failures block a release or require an explicit waiver.

## Priority 3 — observability and operations

### P3.1 Structured logging

- [ ] Emit JSON logs in production mode.
- [ ] Include request ID, route, database identifier, model, latency, retries,
  tool calls, LLM calls, and verification status.
- [ ] Never log passwords, bearer tokens, full database URLs, or raw sensitive
  result rows.
- [ ] Log failure taxonomy categories and semantic diff tags.

**Acceptance criteria**

- Operators can identify slow, failed, rejected, and unverifiable requests.
- Logs are safe to send to a centralized logging system.

### P3.2 Metrics and tracing

- [ ] Add request count, success count, timeout count, and error count.
- [ ] Add latency histograms for LLM, MCP, database, and total request time.
- [ ] Add counters for retries, semantic rejections, and unverified answers.
- [ ] Export OpenTelemetry traces or an equivalent standard.
- [ ] Add a dashboard and alert thresholds.

### P3.3 Cache and concurrency hardening

- [ ] Replace process-local schema cache files with tenant-scoped persistent
  storage where multiple replicas are used.
- [ ] Define cache invalidation when schema changes.
- [ ] Add per-user and global concurrency limits.
- [ ] Add cancellation handling for timed-out LLM calls.
- [ ] Test multiple simultaneous requests.

**Acceptance criteria**

- Multiple replicas do not corrupt or unpredictably diverge in cache state.
- A slow model cannot exhaust all worker capacity.
- Timed-out requests release MCP sessions and semaphore slots.

## Priority 4 — product UX

### P4.1 Multi-turn conversation memory

- [ ] Add conversation and message identifiers.
- [ ] Preserve prior user questions and verified answers.
- [ ] Resolve follow-ups such as “and Germany?” against the prior context.
- [ ] Keep SQL generation context bounded.
- [ ] Define retention and deletion behavior.

**Acceptance criteria**

- Follow-up questions work without resending the entire conversation manually.
- A conversation cannot leak context across users or tenants.
- Users can start a new conversation explicitly.

### P4.2 Improve the browser UI

- [ ] Add a visible connection/readiness indicator.
- [ ] Add request history for the current session.
- [ ] Add copy buttons for answer and SQL.
- [ ] Show verification status and warnings clearly.
- [ ] Add a cancel button for long-running requests.
- [ ] Improve mobile layout and accessibility.
- [ ] Add an optional result table for structured rows.

**Acceptance criteria**

- A user can distinguish loading, success, timeout, authentication failure,
  and unverified answer states.
- SQL and answer can be copied without selecting manually.
- The UI does not expose credentials or database URLs.

### P4.3 Guarded write-mode decision

- [ ] Decide whether writes are in scope for the product.
- [ ] If writes are approved, create a separate write-capable service and
  database role; do not weaken the read-only service.
- [ ] Require explicit confirmation showing the generated mutation and scope.
- [ ] Use transaction boundaries, audit logs, idempotency keys, and rollback.
- [ ] Add policy checks and approval workflows.

**Acceptance criteria**

- Read-only mode remains read-only.
- No write can occur from an ambiguous natural-language request.
- Every approved write is auditable and reversible where possible.

## Priority 5 — test tooling and maintainability

### P5.1 Migrate toward pytest

- [ ] Preserve standalone scripts for compatibility.
- [ ] Add pytest wrappers or convert scripts with collection guards.
- [ ] Create shared fixtures for database targets and fake LLMs.
- [ ] Add markers for unit, SQLite, PostgreSQL, MySQL, live-model, and Docker
  tests.
- [ ] Make IDE test discovery work without executing tests at import time.

**Acceptance criteria**

- `pytest` discovers the complete suite.
- Tests can be selected by database and category.
- Existing `python run_tests.py` behavior remains available during migration.

### P5.2 Add API and UI automated tests

- [ ] Test health endpoints.
- [ ] Test authentication and malformed requests.
- [ ] Test successful query response shape.
- [ ] Test timeout and upstream failure mapping.
- [ ] Test that the static UI is served.
- [ ] Add browser-level tests only if the project adopts a browser test
  dependency.

### P5.3 Dependency and supply-chain maintenance

- [ ] Review dependency pins and transitive vulnerabilities.
- [ ] Remove unused legacy dependencies where safe.
- [ ] Keep `uv.lock` synchronized with `pyproject.toml`.
- [ ] Add automated dependency update review.
- [ ] Build from a minimal, reproducible Docker context.

## Priority 6 — scale and tenancy

Do not start this work until the single-tenant beta is stable.

- [ ] Define tenant identity and database mapping.
- [ ] Store secrets per tenant.
- [ ] Isolate schema caches, vector indexes, logs, and conversations.
- [ ] Add per-tenant quotas and rate limits.
- [ ] Prevent cross-tenant result, prompt, and cache leakage.
- [ ] Load-test the API and Ollama/database bottlenecks.
- [ ] Decide whether to use one model service per tenant or a shared private
  model service.

**Acceptance criteria**

- A tenant cannot select another tenant's database.
- A tenant cannot observe another tenant's SQL, rows, cache, or conversation.
- Capacity limits and failure behavior are documented and tested.

## Definition of production ready

The project should not be labeled production-ready until all of the following
are true:

- [ ] CI runs the real database matrix and blocks regressions.
- [ ] Deployment is reproducible from documented commands.
- [ ] Database access uses least privilege.
- [ ] Public access is protected by HTTPS and authentication.
- [ ] Readiness checks cover LLM and database dependencies.
- [ ] Logs and metrics support incident diagnosis.
- [ ] Model-quality thresholds are defined and met.
- [ ] Known magnitude, tolerance, money-intent, and translation gaps are
  closed or explicitly accepted by the product owner.
- [ ] Rollback has been tested.
- [ ] A controlled beta has produced representative live benchmark data.
