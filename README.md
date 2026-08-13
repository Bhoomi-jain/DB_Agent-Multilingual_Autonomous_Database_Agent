# DB-Agent — Multilingual Autonomous Database Agent

A natural language to SQL query agent powered by LangChain, a local Ollama model, and the **Model Context Protocol (MCP)**. Ask questions about your database in any language and get back accurate answers — the agent detects the question's language, reasons over the database in English, and replies back in the language you asked in.

## Features

- Natural language to SQL query conversion, runs entirely on a local model (no cloud API key required)
- **Genuinely multilingual**: detects the input language, translates internally to English for SQL generation, then translates the final answer back — not just relying on the model's raw multilingual training
- **MCP-based database access**: the agent doesn't talk to SQLite directly. It connects as an MCP client to the official [`mcp-server-sqlite`](https://pypi.org/project/mcp-server-sqlite/) server over stdio, and only uses its read-only tools
- Automatic query validation and error correction — the agent rewrites and retries failed queries
- Support for complex queries (JOINs, aggregations, subqueries)
- Guardrails against destructive statements: the MCP server does expose `write_query`/`create_table`, but the agent is only ever given the `read_query`, `list_tables`, and `describe_table` tools, so DML/DDL is not reachable
- Clean, readable CLI output via `rich`

## Demo Database

Uses the [Chinook database](https://github.com/lerocha/chinook-database) — a sample database representing a digital media store with tables for artists, albums, tracks, customers, invoices, and more.

## Architecture

```
 you ──► detect language ──► translate to English (if needed)
                                      │
                                      ▼
                          LangChain agent (Qwen3 via Ollama)
                                      │
                         MCP client (stdio) ──► mcp-server-sqlite ──► chinook.db
                                      │
                                      ▼
                          English answer ──► translate back to your language
```

The agent never opens the SQLite file itself — every schema lookup and query goes through the MCP server as a tool call, the same way Claude Desktop or any other MCP client would talk to it.

## Quick Start

### Prerequisites

- [Ollama](https://ollama.com/) installed and running
- A compatible local model pulled (`qwen3:4b` recommended for the default setup — run `ollama pull qwen3:4b`)
- Python 3.11 or higher

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Bhoomi-jain/DB_Agent—Multilingual_Autonomous_Database_Agent.git
cd DB_Agent—Multilingual_Autonomous_Database_Agent
```

2. Download the Chinook database:
```bash
curl -L -o chinook.db https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite
```

3. Create a virtual environment and install dependencies:
```bash
# Using uv (recommended)
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Or using standard pip
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs `mcp-server-sqlite` and puts its CLI on your `PATH` inside the virtual environment — the agent launches it automatically as a subprocess, you don't need to run it separately.

No API keys are required — the agent talks to your local Ollama instance by default.

## Usage

### Command Line Interface

Run the agent from the command line with a natural language question, in any language the model understands:

```bash
python agent.py "What are the top 5 best-selling artists?"
```

```bash
python agent.py "¿Cuáles son los 5 álbumes más vendidos?"
```

```bash
python agent.py "कनाडा से कितने ग्राहक हैं?"
```

Optionally point at a different database file:

```bash
python agent.py "How many employees are there?" --db-path /path/to/other.db
```

The CLI prints the question, its progress (language detection, MCP connection, query processing), and a formatted panel with the final answer in your original language — or a formatted error panel if something goes wrong.

## How It Works

1. **Detect** - Identifies the language of the incoming question
2. **Translate (in)** - If not English, translates the question to English
3. **Connect** - Launches `mcp-server-sqlite` as a subprocess and connects as an MCP client
4. **Discover** - Lists available tables in the database via the MCP `list_tables` tool
5. **Inspect** - Retrieves schema for relevant tables via `describe_table`
6. **Generate** - Creates a SQL query using the local Ollama model
7. **Validate** - Double-checks the query for syntax and safety
8. **Execute** - Runs the query via the MCP `read_query` tool (the only query tool exposed to the agent)
9. **Retry** - If errors occur, automatically rewrites and retries
10. **Translate (out)** - Translates the English answer back into the original question's language

## Configuration

Key configuration options in `agent.py`:

```python
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="qwen3:4b",
    temperature=0
)

# Which MCP tools the agent is allowed to use
READ_ONLY_TOOLS = {"read_query", "list_tables", "describe_table"}

# Default result limit (in system prompt)
system_prompt=SYSTEM_PROMPT.format(
    dialect="SQLite",
    top_k=5  # Adjust as needed
)
```

Swap in a different Ollama model by changing the `model` argument to `ChatOllama`, provided it's been pulled locally. You can also point `MCP_SQLITE_COMMAND` (env var) at a different `mcp-server-sqlite` binary if it's not on your `PATH`.

## Project Structure

```
DB_Agent—Multilingual_Autonomous_Database_Agent/
├── agent.py              # Core agent: MCP client, translation pipeline, CLI entry point
├── chinook.db             # Sample SQLite database (gitignored)
├── pyproject.toml         # Project configuration and dependencies
├── uv.lock                 # Locked dependency versions
├── .gitignore               # Git ignore rules
└── README.md                 # This file
```

## Requirements

All dependencies are specified in `pyproject.toml`:

- langchain >= 1.2.3
- langchain-community >= 0.3.0
- langchain-ollama
- langchain-mcp-adapters
- mcp-server-sqlite
- langgraph >= 1.0.6
- langdetect
- sqlalchemy >= 2.0.0
- python-dotenv >= 1.0.0
- rich >= 13.0.0

> Note: `mcp-server-sqlite` is Anthropic's reference SQLite MCP server (published under `modelcontextprotocol/servers`). It has since been archived upstream — it still installs and runs fine, but if you'd rather use an actively maintained alternative, `@berthojoris/mcp-sqlite-server` (npm) is a drop-in replacement; just update the `command`/`args` in `build_agent`.

## License

MIT

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/) and [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
- Uses the [Chinook Database](https://github.com/lerocha/chinook-database)
- Database access via [mcp-server-sqlite](https://pypi.org/project/mcp-server-sqlite/)
- Powered by [Ollama](https://ollama.com/) and Qwen3

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
