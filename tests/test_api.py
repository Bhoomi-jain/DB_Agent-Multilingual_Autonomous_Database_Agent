import asyncio
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "sqlite:///test_sqlite.db")
os.environ.setdefault("API_TOKEN", "test-token")
os.environ.setdefault("ALLOW_ANONYMOUS", "false")

sys.path.insert(0, str(__file__).rsplit("/tests/", 1)[0])
import httpx

import api


class FakeMetrics:
    llm_calls = 1
    tool_calls = 2
    retries = 0
    answer_verified = True

    def summary(self):
        return "fake metrics"


class FakeAgent:
    async def run(self, question):
        return "There are 3 customers.", "SELECT COUNT(*) FROM customers", FakeMetrics(), "en"


class FakeSQLAgent:
    def __init__(self, *args, **kwargs):
        pass


class FakeMultilingualAgent:
    def __init__(self, inner, llm):
        pass

    async def run(self, question):
        return await FakeAgent().run(question)


class FakeOllamaResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"models": [{"name": api.LLM_MODEL}]}


class FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def get(self, url):
        assert url.endswith("/api/tags")
        return FakeOllamaResponse()


async def request(method, path, **kwargs):
    transport = httpx.ASGITransport(app=api.app)
    async with REAL_ASYNC_CLIENT(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, **kwargs)


async def main():
    api.app.state.llm = object()
    api.app.state.semaphore = asyncio.Semaphore(2)
    original_sql_agent = api.SQLAgent
    original_multilingual_agent = api.MultilingualAgent
    original_http_client = api.httpx.AsyncClient
    global REAL_ASYNC_CLIENT
    REAL_ASYNC_CLIENT = original_http_client
    api.SQLAgent = FakeSQLAgent
    api.MultilingualAgent = FakeMultilingualAgent
    api.httpx.AsyncClient = FakeAsyncClient
    try:
        live = await request("GET", "/health/live", headers={"X-Request-ID": "live-test"})
        assert live.status_code == 200
        assert live.json() == {"status": "ok"}
        assert live.headers["X-Request-ID"] == "live-test"

        ready = await request("GET", "/health/ready")
        assert ready.status_code == 200
        assert ready.json() == {"status": "ready", "database": "ok", "llm": "ok"}

        missing_auth = await request(
            "POST", "/v1/query", json={"question": "How many customers?"}
        )
        assert missing_auth.status_code == 401

        query = await request(
            "POST",
            "/v1/query",
            headers={"Authorization": "Bearer test-token", "X-Request-ID": "query-test"},
            json={"question": "How many customers?"},
        )
        assert query.status_code == 200
        assert query.headers["X-Request-ID"] == "query-test"
        assert query.json()["answer"] == "There are 3 customers."
        assert query.json()["metrics"]["answer_verified"] is True
    finally:
        api.SQLAgent = original_sql_agent
        api.MultilingualAgent = original_multilingual_agent
        api.httpx.AsyncClient = original_http_client


asyncio.run(main())
