from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(name):
    with (ROOT / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_production_compose_defines_api_and_model_init():
    compose = load_yaml("docker-compose.production.yml")
    services = compose["services"]
    assert {"agent", "ollama", "ollama-model-init"} <= services.keys()
    assert services["agent"]["depends_on"]["ollama-model-init"]["condition"] == (
        "service_completed_successfully"
    )
    assert services["ollama-model-init"]["restart"] == "no"
    assert "ollama_data" in compose["volumes"]


def test_sqlite_override_mounts_read_only_database():
    compose = load_yaml("docker-compose.sqlite.yml")
    assert compose["services"]["agent"]["volumes"] == [
        "./chinook.db:/app/chinook.db:ro"
    ]


def test_https_override_hides_direct_api_port():
    compose = load_yaml("docker-compose.https.yml")
    agent = compose["services"]["agent"]
    caddy = compose["services"]["caddy"]
    assert agent["ports"] == []
    assert agent["expose"] == ["8000"]
    assert caddy["ports"] == ["80:80", "443:443"]
    assert caddy["depends_on"]["agent"]["condition"] == "service_started"


def test_ci_workflow_covers_locked_tests_and_image():
    workflow = load_yaml(".github/workflows/ci.yml")
    steps = workflow["jobs"]["test"]["steps"]
    commands = "\n".join(step["run"] for step in steps if "run" in step)
    assert "uv sync --locked" in commands
    assert "seed_testdb.py --target all" in commands
    assert "run_tests.py" in commands
    assert "docker build --tag db-agent-ci ." in commands
    assert any(step.get("uses") == "actions/upload-artifact@v4" for step in steps)


def test_caddy_proxies_to_internal_agent():
    assert "{$DOMAIN}" in (ROOT / "Caddyfile").read_text(encoding="utf-8")
    assert "reverse_proxy agent:8000" in (ROOT / "Caddyfile").read_text(encoding="utf-8")


test_production_compose_defines_api_and_model_init()
test_sqlite_override_mounts_read_only_database()
test_https_override_hides_direct_api_port()
test_ci_workflow_covers_locked_tests_and_image()
test_caddy_proxies_to_internal_agent()
print("deployment configuration tests passed")
