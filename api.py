"""HTTP API for the production read-only database agent."""

from contextlib import asynccontextmanager
import asyncio
import logging
import os
import uuid
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from core_agent import SQLAgent
from multilingual import MultilingualAgent, _dialect_for
from production_agent import build_llm


logger = logging.getLogger("db_agent.api")
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL must be set before starting the API")

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "2000"))
API_TOKEN = os.getenv("API_TOKEN")
ALLOW_ANONYMOUS = os.getenv("ALLOW_ANONYMOUS", "").lower() == "true"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
HEALTH_TIMEOUT_SECONDS = float(os.getenv("HEALTH_TIMEOUT_SECONDS", "3"))


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)


class QueryResponse(BaseModel):
    answer: str
    sql: str | None
    language: str
    metrics: dict[str, Any]


def require_api_token(authorization: str | None = Header(default=None)) -> None:
    if ALLOW_ANONYMOUS:
        return
    if not API_TOKEN:
        raise HTTPException(status_code=503, detail="API authentication is not configured")
    expected = f"Bearer {API_TOKEN}"
    if authorization != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_TOKEN and not ALLOW_ANONYMOUS:
        raise RuntimeError("API_TOKEN must be set, or ALLOW_ANONYMOUS=true for local development")
    provider = LLM_PROVIDER
    model = LLM_MODEL
    max_tokens = int(os.getenv("MAX_TOKENS", "512"))
    app.state.llm = build_llm(provider, model, max_tokens=max_tokens)
    app.state.semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    yield


app = FastAPI(
    title="Multilingual Autonomous Database Agent",
    version="0.3.0",
    lifespan=lifespan,
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled request failure", extra={"request_id": request_id})
        raise
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
async def ready() -> dict[str, str]:
    checks: dict[str, str] = {}
    if not hasattr(app.state, "llm"):
        checks["application"] = "not_initialized"
    else:
        checks["application"] = "ok"

    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        engine.dispose()
        checks["database"] = "ok"
    except Exception:
        logger.exception("Readiness database check failed")
        checks["database"] = "unavailable"

    if LLM_PROVIDER == "ollama":
        try:
            async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT_SECONDS) as client:
                response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
            checks["llm"] = (
                "ok" if any(item.get("name") == LLM_MODEL for item in models)
                else "model_missing"
            )
        except Exception:
            logger.exception("Readiness Ollama check failed")
            checks["llm"] = "unavailable"
    elif LLM_PROVIDER == "anthropic":
        checks["llm"] = "ok" if os.getenv("ANTHROPIC_API_KEY") else "api_key_missing"
    else:
        checks["llm"] = "unsupported_provider"

    if any(value != "ok" for value in checks.values()):
        raise HTTPException(status_code=503, detail={"status": "not_ready", "checks": checks})
    return {"status": "ready", "database": "ok", "llm": "ok"}


@app.post("/v1/query", response_model=QueryResponse, dependencies=[Depends(require_api_token)])
async def query(request: QueryRequest) -> QueryResponse:
    async with app.state.semaphore:
        inner = SQLAgent(
            DATABASE_URL,
            app.state.llm,
            _dialect_for(DATABASE_URL),
            max_retries=int(os.getenv("MAX_RETRIES", "2")),
            cache_ttl=int(os.getenv("CACHE_TTL_SECONDS", "300")),
            use_cache=True,
        )
        agent = MultilingualAgent(inner, app.state.llm)
        try:
            answer, sql, metrics, language = await asyncio.wait_for(
                agent.run(request.question),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail={"message": "Query timed out", "request_id": request.state.request_id},
            ) from exc
        except Exception as exc:
            logger.exception(
                "Database-agent query failed",
                extra={"request_id": request.state.request_id},
            )
            raise HTTPException(
                status_code=500,
                detail={"message": "Query failed", "request_id": request.state.request_id},
            ) from exc

        return QueryResponse(
            answer=answer,
            sql=sql,
            language=language,
            metrics={
                "llm_calls": metrics.llm_calls,
                "tool_calls": metrics.tool_calls,
                "retries": metrics.retries,
                "answer_verified": metrics.answer_verified,
                "summary": metrics.summary(),
            },
        )
