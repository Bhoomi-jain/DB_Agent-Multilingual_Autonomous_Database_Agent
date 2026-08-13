"""
production_agent.py — Production text-to-SQL agent for PostgreSQL and MySQL.

Connects to `db_mcp_server.py` (a self-contained, AST-validated read-only MCP
server — see that file's docstring for why it exists instead of a community
MCP server) and answers natural-language questions against a live database,
with FK-aware joins and analytics-query support.

The LLM backend is configurable: local Ollama (free, private) or the
Anthropic API (stronger reasoning, needs ANTHROPIC_API_KEY).
"""
import os
import sys
import asyncio
import argparse

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()

SYSTEM_PROMPT = """
You are a senior data analyst agent with read-only access to a {dialect}
database, via a set of tools backed by a real database connection.

Workflow — follow this every time:
1. Call list_tables() to see what's available. Do NOT skip this.
2. Call describe_table(...) on every table you plan to reference, to get
   exact column names and types. Never guess a column name.
3. Before writing ANY join, call list_foreign_keys() (or
   list_foreign_keys(table_name) for a single table) to find the correct
   join keys. Do not guess join columns — use what the tool tells you.
4. Write a single, syntactically correct {dialect} SELECT statement and run
   it with run_query(...). Use JOINs, GROUP BY, aggregate functions (SUM,
   COUNT, AVG, etc.), and window functions freely for analytics questions —
   these are all supported and encouraged when they answer the question
   accurately.
5. Unless the user asks for a specific number of rows, LIMIT results to a
   reasonable number (e.g. 20) and order by whatever column makes the
   result most useful (e.g. a computed metric, descending).
6. If run_query fails, read the error, fix the query, and retry. You have no
   tool capable of INSERT/UPDATE/DELETE/DDL — every tool you're given is
   read-only, so don't attempt to write or modify data even if asked.
7. Answer the user's question directly and concisely, backed by the numbers
   you found. Mention the key figures, not just a vague summary.
"""


def build_llm(provider: str, model: str | None):
    provider = provider.lower()
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model or "qwen3:4b", temperature=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set — required for --provider anthropic."
            )
        return ChatAnthropic(model=model or "claude-sonnet-4-5-20250929", temperature=0)
    raise ValueError(f"Unknown provider '{provider}'. Use 'ollama' or 'anthropic'.")


async def build_agent(db_url: str, provider: str, model: str | None):
    dialect = "PostgreSQL" if db_url.startswith("postgresql") else \
              "MySQL" if db_url.startswith("mysql") else "SQL"

    server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db_mcp_server.py")
    client = MultiServerMCPClient({
        "db": {
            "command": sys.executable,
            "args": [server_script, "--db-url", db_url],
            "transport": "stdio",
        }
    })
    tools = await client.get_tools()

    llm = build_llm(provider, model)
    agent = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT.format(dialect=dialect))
    return agent


async def answer_question(question: str, db_url: str, provider: str, model: str | None) -> str:
    console.print(f"[dim]Connecting to database via MCP ({provider})...[/dim]")
    agent = await build_agent(db_url, provider, model)

    console.print("[dim]Processing query...[/dim]")
    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


def main():
    parser = argparse.ArgumentParser(
        description="Production text-to-SQL agent for PostgreSQL/MySQL, via MCP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  export DATABASE_URL="postgresql+psycopg2://user:pass@localhost/mydb"
  python production_agent.py "What are the top 5 customers by revenue?"

  python production_agent.py "Revenue by country, completed orders only" \\
      --db-url "mysql+pymysql://user:pass@localhost/mydb" \\
      --provider anthropic
        """,
    )
    parser.add_argument("question", type=str, help="Natural language question")
    parser.add_argument(
        "--db-url",
        default=os.getenv("DATABASE_URL"),
        help="SQLAlchemy URL, e.g. postgresql+psycopg2://user:pass@host/db "
             "or mysql+pymysql://user:pass@host/db (or set DATABASE_URL)",
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("LLM_PROVIDER", "ollama"),
        choices=["ollama", "anthropic"],
        help="LLM backend to use (default: ollama, or $LLM_PROVIDER)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL"),
        help="Override the default model for the chosen provider",
    )
    args = parser.parse_args()

    if not args.db_url:
        console.print("[bold red]Error:[/bold red] --db-url or DATABASE_URL env var is required")
        sys.exit(1)

    console.print(Panel(f"[bold cyan]Question:[/bold cyan] {args.question}", border_style="cyan"))
    console.print()

    try:
        answer = asyncio.run(answer_question(args.question, args.db_url, args.provider, args.model))
        console.print(Panel(f"[bold green]Answer:[/bold green]\n\n{answer}", border_style="green"))
    except Exception as e:
        console.print(Panel(f"[bold red]Error:[/bold red]\n\n{str(e)}", border_style="red"))
        sys.exit(1)


if __name__ == "__main__":
    main()
