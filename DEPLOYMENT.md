# Deployment and operating guide

This guide describes three separate ways to run the project:

1. **API only** — run the FastAPI service and call `POST /v1/query`.
2. **Browser UI** — run the same FastAPI service and open `/`.
3. **Docker deployment** — run the API and Ollama together with Docker Compose.

The application is read-only. It accepts natural-language questions, generates
and validates a read-only SQL query, executes it through the AST-validated MCP
server, and returns an answer plus the SQL and run metrics.

## 1. Prerequisites

- Linux, macOS, or WSL2
- Python 3.11 or newer
- Git
- Ollama for the local free-LLM path
- Docker and Docker Compose for the container path
- A database URL for a real deployment, or the bundled `chinook.db` for a demo

The free local model path does not require an Anthropic or other paid API key.
The default model is `llama3.2:latest`.

## 2. Install the project locally

```bash
cd "/home/bhooms/DB_Agent—Multilingual_Autonomous_Database_Agent"
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

If `uv` is unavailable:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The repository includes the API dependencies (`fastapi` and `uvicorn`) in
`pyproject.toml`. After changing dependencies, refresh and install the lock:

```bash
uv lock
uv sync
```

## 3. Start Ollama locally

Install Ollama if the `ollama` command does not exist:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start the Ollama server in its own terminal and leave it running:

```bash
ollama serve
```

In another terminal, download the model:

```bash
ollama pull llama3.2
```

Check that the server and model are available:

```bash
ollama --version
ollama list
curl -sS http://127.0.0.1:11434/api/tags
```

The model list should contain `llama3.2:latest`. If the model is missing, the
API will start but query requests will fail.

## 4. API-only mode with the bundled SQLite database

This is the simplest local run. It uses the repository's ignored demo database
and exposes only the JSON API.

### Terminal 1: Ollama

```bash
ollama serve
```

### Terminal 2: API

Use a fixed local token so it is easy to copy:

```bash
cd "/home/bhooms/DB_Agent—Multilingual_Autonomous_Database_Agent"

DATABASE_URL="sqlite:////home/bhooms/DB_Agent—Multilingual_Autonomous_Database_Agent/chinook.db" \
API_TOKEN="local-test-token" \
.venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000
```

Keep this terminal open. The expected startup line is:

```text
Uvicorn running on http://127.0.0.1:8000
```

Check liveness and readiness:

```bash
curl -sS http://127.0.0.1:8000/health/live
curl -sS http://127.0.0.1:8000/health/ready
```

Expected responses:

```json
{"status":"ok"}
{"status":"ready"}
```

### Terminal 3: API request

Every protected request must include the exact bearer token:

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/query" \
  -H "Authorization: Bearer local-test-token" \
  -H "Content-Type: application/json" \
  -d '{"question":"How many customers are there?"}'
```

A missing or mismatched header returns `401 Unauthorized`:

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"How many customers are there?"}'
```

For a generated token instead of a fixed token:

```bash
export API_TOKEN="$(openssl rand -hex 32)"
echo "$API_TOKEN"
```

Start the API in the same shell with that exported variable, then send:

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/query" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"question":"Which artist has the most tracks?"}'
```

Do not put the literal placeholder `your read-only database URL` in
`DATABASE_URL`. It is documentation text, not a usable connection string.

## 5. Browser UI mode

The UI is served by the same FastAPI process; no frontend build or Node.js
installation is required.

Start the API as in the previous section, then open:

```text
http://127.0.0.1:8000/
```

The OpenAPI console is also available at:

```text
http://127.0.0.1:8000/docs
```

In the browser UI:

1. Enter `local-test-token` in the **API token** field.
2. Enter a question.
3. Click **Ask database**.
4. Expand **SQL used** and **Run metrics** to inspect the execution.

The token is kept in `sessionStorage` for the current browser tab only. The UI
uses a same-origin request, so no CORS setup is required.

Useful test questions:

```text
How many customers are there?
How many customers are from Brazil?
Which artist has the most tracks?
Which genre has the most tracks?
What is the total revenue from all invoices?
What is the average track price?
¿Cuántos clientes hay de Canadá?
```

## 6. Production database URLs

For a real deployment, use a database account created specifically for this
application with read-only permissions. Do not use an administrator account.

PostgreSQL:

```bash
export DATABASE_URL="postgresql+psycopg2://readonly_user:PASSWORD@db.example.com:5432/appdb"
```

MySQL or MariaDB:

```bash
export DATABASE_URL="mysql+pymysql://readonly_user:PASSWORD@db.example.com:3306/appdb"
```

SQLite:

```bash
export DATABASE_URL="sqlite:////absolute/path/to/database.db"
```

The API never accepts a database URL from the browser request. The server
process owns the configured database connection.

Before production, verify the read-only role using the database provider's
client. For PostgreSQL, a typical setup is:

```sql
CREATE USER agent_readonly WITH PASSWORD 'use-a-secret-from-your-secret-manager';
GRANT CONNECT ON DATABASE appdb TO agent_readonly;
GRANT USAGE ON SCHEMA public TO agent_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO agent_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT ON TABLES TO agent_readonly;
```

Run those statements as a database administrator, never as part of the
application startup.

## 7. Docker Compose deployment

The Compose file starts:

- `agent` — FastAPI API and browser UI
- `ollama` — local model server
- `ollama-model-init` — downloads the configured model once

The API container reaches Ollama at `http://ollama:11434`; do not use
`127.0.0.1:11434` inside the API container.

### 7.1 Docker with a real PostgreSQL/MySQL database

```bash
cd "/home/bhooms/DB_Agent—Multilingual_Autonomous_Database_Agent"

export DATABASE_URL="postgresql+psycopg2://readonly_user:PASSWORD@db.example.com:5432/appdb"
export API_TOKEN="$(openssl rand -hex 32)"
export LLM_MODEL="llama3.2:latest"

docker compose -f docker-compose.production.yml up -d --build
```

Follow startup logs:

```bash
docker compose -f docker-compose.production.yml logs -f agent
docker compose -f docker-compose.production.yml logs -f ollama
```

Check the running services:

```bash
docker compose -f docker-compose.production.yml ps
curl -sS http://127.0.0.1:8000/health/live
curl -sS http://127.0.0.1:8000/health/ready
```

Open the UI:

```text
http://127.0.0.1:8000/
```

The model download can take several minutes the first time. The model is
stored in the named `ollama_data` volume and is reused after restarts.

Stop the deployment without deleting the model:

```bash
docker compose -f docker-compose.production.yml down
```

Stop it and delete the downloaded model volume:

```bash
docker compose -f docker-compose.production.yml down -v
```

Do not use `down -v` unless you intentionally want to download the model again.

### 7.2 Docker with the bundled SQLite demo

The SQLite file must be mounted into the API container read-only. Add this
under the `agent` service in `docker-compose.production.yml`:

```yaml
    volumes:
      - ./chinook.db:/app/chinook.db:ro
```

Then run:

```bash
export DATABASE_URL="sqlite:////app/chinook.db"
export API_TOKEN="$(openssl rand -hex 32)"
docker compose -f docker-compose.production.yml up -d --build
```

Open:

```text
http://127.0.0.1:8000/
```

The direct Uvicorn method is simpler for the SQLite demo because it avoids the
container volume mapping.

### 7.3 Validate Compose before starting

```bash
docker compose -f docker-compose.production.yml config --quiet
docker build -t db-agent-local .
```

Warnings about unset `DATABASE_URL` or `API_TOKEN` mean the shell variables
were not exported. The configuration may parse, but the API will not start
without authentication and a database URL.

## 8. Running the full regression suite

The project test runner executes 23 scripts in subprocesses:

```bash
cd "/home/bhooms/DB_Agent—Multilingual_Autonomous_Database_Agent"
python seed_testdb.py --target all
python run_tests.py
```

The expected current result is:

```text
23/23 passed
```

The report is written to:

```text
test_report.txt
```

The suite is separate from the live Ollama API. It uses scripted fake LLM
responses and real database fixtures, so it does not prove that a particular
Ollama model will answer every natural-language question correctly.

## 9. Exact troubleshooting commands

### 9.1 Browser says “failed to load page”

Check whether Uvicorn is still running:

```bash
ps -ef | grep '[u]vicorn'
curl -sS -v http://127.0.0.1:8000/health/live
```

If nothing is listening, restart the API and leave that terminal open:

```bash
.venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000
```

If port 8000 is busy:

```bash
.venv/bin/uvicorn api:app --host 127.0.0.1 --port 8001
```

Then open `http://127.0.0.1:8001/`.

### 9.2 API returns 401

The request must contain:

```bash
-H "Authorization: Bearer local-test-token"
```

The token must exactly match the `API_TOKEN` used when Uvicorn started:

```bash
echo "$API_TOKEN"
```

If the server was started with a fixed token, use that fixed token in both
places. Restart Uvicorn after changing `API_TOKEN`; environment variables are
read at process startup.

### 9.3 API returns “Query failed”

The API deliberately returns a generic client error. Read the terminal running
Uvicorn for the full logged exception, then run these checks:

```bash
curl -sS http://127.0.0.1:11434/api/tags
ollama list
curl -sS http://127.0.0.1:8000/health/ready
```

If Ollama is unavailable:

```bash
ollama serve
ollama pull llama3.2
```

If the model is not listed, pull it again:

```bash
ollama pull llama3.2:latest
```

If the database URL is wrong, test that the file exists for SQLite:

```bash
test -f "/home/bhooms/DB_Agent—Multilingual_Autonomous_Database_Agent/chinook.db" \
  && echo "SQLite database exists" \
  || echo "SQLite database is missing"
```

For a Python-level SQLite check:

```bash
.venv/bin/python - <<'PY'
import sqlite3
conn = sqlite3.connect("chinook.db")
print(conn.execute("SELECT COUNT(*) FROM Customer").fetchone()[0])
print(conn.execute("SELECT COUNT(*) FROM Track").fetchone()[0])
PY
```

### 9.4 API returns a wrong or unexpected answer

Send a simple known question first:

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/query" \
  -H "Authorization: Bearer local-test-token" \
  -H "Content-Type: application/json" \
  -d '{"question":"How many customers are there?"}'
```

Cross-check expected Chinook values:

```bash
.venv/bin/python - <<'PY'
import sqlite3
conn = sqlite3.connect("chinook.db")
checks = [
    ("Brazilian customers",
     "SELECT COUNT(*) FROM Customer WHERE Country='Brazil'"),
    ("Canadian customers",
     "SELECT COUNT(*) FROM Customer WHERE Country='Canada'"),
    ("Average track price",
     "SELECT AVG(UnitPrice) FROM Track"),
    ("Invoice revenue",
     "SELECT SUM(Total) FROM Invoice"),
]
for label, query in checks:
    print(label, "=>", conn.execute(query).fetchone()[0])
PY
```

The model is probabilistic. Keep the returned SQL and metrics when reporting a
quality issue; they are needed to reproduce the failure.

### 9.5 API returns 422

The request body must be JSON with a non-empty `question`:

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/query" \
  -H "Authorization: Bearer local-test-token" \
  -H "Content-Type: application/json" \
  -d '{"question":"How many tracks are there?"}'
```

The configured maximum question length is controlled by:

```bash
export MAX_QUESTION_LENGTH=2000
```

### 9.6 Docker container exits

Inspect all services:

```bash
docker compose -f docker-compose.production.yml ps
docker compose -f docker-compose.production.yml logs --tail=200 agent
docker compose -f docker-compose.production.yml logs --tail=200 ollama
docker compose -f docker-compose.production.yml logs --tail=200 ollama-model-init
```

Confirm required variables are present without printing the secret value:

```bash
test -n "$DATABASE_URL" && echo "DATABASE_URL is set" || echo "DATABASE_URL is missing"
test -n "$API_TOKEN" && echo "API_TOKEN is set" || echo "API_TOKEN is missing"
```

Validate the Compose file:

```bash
docker compose -f docker-compose.production.yml config --quiet
```

Rebuild after code or dependency changes:

```bash
docker compose -f docker-compose.production.yml down
docker compose -f docker-compose.production.yml up -d --build
```

## 10. Validation commands used during this deployment work

These are the checks used to validate the production surface:

```bash
python -m py_compile api.py production_agent.py
uv lock
uv sync
python run_tests.py
git diff --check
```

Local API smoke checks:

```bash
DATABASE_URL="sqlite:////home/bhooms/DB_Agent—Multilingual_Autonomous_Database_Agent/chinook.db" \
API_TOKEN="local-test-token" \
.venv/bin/python - <<'PY'
from fastapi.testclient import TestClient
from api import app

with TestClient(app) as client:
    assert client.get("/").status_code == 200
    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/health/ready").status_code == 200
    assert client.post("/v1/query", json={"question": "x"}).status_code == 401
    assert client.post(
        "/v1/query",
        headers={"Authorization": "Bearer local-test-token"},
        json={"question": ""},
    ).status_code == 422
print("API/UI smoke checks passed")
PY
```

Live API smoke request:

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/query" \
  -H "Authorization: Bearer local-test-token" \
  -H "Content-Type: application/json" \
  -d '{"question":"How many customers are there?"}'
```

Container build check:

```bash
docker build -t db-agent-local .
docker compose -f docker-compose.production.yml config --quiet
```

## 11. Security and production limits

- Use HTTPS in front of the API.
- Store `API_TOKEN` and database credentials in a secret manager.
- Use a database role with SELECT-only privileges.
- Do not accept database URLs from end users.
- Do not enable `ALLOW_ANONYMOUS=true` on a public deployment.
- Do not expose Ollama's port publicly.
- Keep the API and database on private network paths where possible.
- Set `MAX_CONCURRENCY` and `REQUEST_TIMEOUT_SECONDS` for the machine capacity.
- Review the returned SQL and `answer_verified` metric during beta operation.
- Keep the read-only MCP validation boundary; do not replace it with direct
  unrestricted SQL execution.
