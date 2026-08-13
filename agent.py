"""
DB-Agent — Multilingual Autonomous Database Agent

Connects to a SQLite database through the official `mcp-server-sqlite`
Model Context Protocol server (rather than talking to the DB directly),
and answers natural-language questions in whatever language they're asked.

Ask in Spanish, get an answer in Spanish. Ask in Hindi, get an answer in
Hindi. Internally, everything is translated to/from English, since that's
what the SQL-generation model reasons best in.
"""
import os
import sys
import asyncio
import argparse

from dotenv import load_dotenv
from langdetect import detect, DetectorFactory
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from rich.console import Console
from rich.panel import Panel

# Make langdetect deterministic (it's seeded off wall-clock jitter by default)
DetectorFactory.seed = 0

load_dotenv()
console = Console()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

You only have access to read-only tools (list_tables, describe_table,
read_query). Do NOT attempt any INSERT, UPDATE, DELETE, DROP, or other
data-modifying statement — you have no tool that would allow it, and any
such SQL will fail.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.

Always respond in English — a separate step will translate your final answer
into the user's language, so do not attempt translation yourself.
"""

# mcp-server-sqlite exposes six tools; we only expose the read-only ones to
# the agent so the original "no DML" guarantee still holds even though the
# MCP server itself is capable of writes.
READ_ONLY_TOOLS = {"read_query", "list_tables", "describe_table"}

MCP_SERVER_COMMAND = os.getenv("MCP_SQLITE_COMMAND", "mcp-server-sqlite")

LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "zh-cn": "Chinese", "zh-tw": "Chinese (Traditional)", "ja": "Japanese",
    "ko": "Korean", "ar": "Arabic", "hi": "Hindi", "bn": "Bengali",
    "pa": "Punjabi", "ur": "Urdu", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "gu": "Gujarati", "tr": "Turkish", "vi": "Vietnamese",
    "th": "Thai", "id": "Indonesian", "pl": "Polish", "uk": "Ukrainian",
    "he": "Hebrew", "el": "Greek", "sv": "Swedish", "fi": "Finnish",
}


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------

async def build_agent(db_path: str, model: ChatOllama):
    """Connect to the SQLite MCP server and build a LangChain agent restricted
    to its read-only tools."""
    client = MultiServerMCPClient({
        "sqlite": {
            "command": MCP_SERVER_COMMAND,
            "args": ["--db-path", db_path],
            "transport": "stdio",
        }
    })

    all_tools = await client.get_tools()
    tools = [t for t in all_tools if t.name in READ_ONLY_TOOLS]

    if not tools:
        raise RuntimeError(
            "No read-only tools found on the MCP SQLite server — "
            "check that mcp-server-sqlite is installed and on PATH."
        )

    agent = create_agent(
        model,
        tools,
        system_prompt=SYSTEM_PROMPT.format(dialect="SQLite", top_k=5),
    )
    return agent


# ---------------------------------------------------------------------------
# Multilingual helpers
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """Best-effort language code detection. Falls back to English."""
    try:
        return detect(text)
    except Exception:
        return "en"


async def translate(model: ChatOllama, text: str, target_language: str) -> str:
    """Translate text into target_language using the local model."""
    prompt = (
        f"Translate the following text into {target_language}. "
        f"Output ONLY the translation, with no preamble, quotes, or notes.\n\n"
        f"{text}"
    )
    result = await model.ainvoke(prompt)
    return result.content.strip()


# ---------------------------------------------------------------------------
# Main query flow
# ---------------------------------------------------------------------------

async def answer_question(question: str, db_path: str) -> str:
    model = ChatOllama(model="qwen3:4b", temperature=0)

    lang_code = detect_language(question)
    lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)

    if lang_code != "en":
        console.print(f"[dim]Detected language: {lang_name} — translating to English...[/dim]")
        english_question = await translate(model, question, "English")
    else:
        english_question = question

    console.print("[dim]Connecting to SQLite MCP server...[/dim]")
    agent = await build_agent(db_path, model)

    console.print("[dim]Processing query...[/dim]")
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": english_question}]
    })
    english_answer = result["messages"][-1].content

    if lang_code != "en":
        console.print(f"[dim]Translating answer back to {lang_name}...[/dim]")
        return await translate(model, english_answer, lang_name)

    return english_answer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multilingual Text-to-SQL Agent powered by LangChain, "
                     "MCP (mcp-server-sqlite), and Qwen3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py "What are the top 5 best-selling artists?"
  python agent.py "¿Cuáles son los 5 álbumes más vendidos?"
  python agent.py "कनाडा से कितने ग्राहक हैं?"
        """,
    )
    parser.add_argument(
        "question",
        type=str,
        help="Natural language question, in any supported language, to answer "
             "using the connected database",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to the SQLite database (default: chinook.db next to this script)",
    )
    args = parser.parse_args()

    db_path = args.db_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "chinook.db"
    )

    console.print(Panel(
        f"[bold cyan]Question:[/bold cyan] {args.question}",
        border_style="cyan",
    ))
    console.print()

    try:
        answer = asyncio.run(answer_question(args.question, db_path))
        console.print(Panel(
            f"[bold green]Answer:[/bold green]\n\n{answer}",
            border_style="green",
        ))
    except Exception as e:
        console.print(Panel(
            f"[bold red]Error:[/bold red]\n\n{str(e)}",
            border_style="red",
        ))
        sys.exit(1)


if __name__ == "__main__":
    main()
