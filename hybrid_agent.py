"""
hybrid_agent.py — RAG + SQL hybrid: text data routes to vector search,
structured data routes to SQL, and some questions need both.

Three routes, decided per-question:
  - "sql"      — pure structured query (e.g. "how many customers are from
                 Canada"). Delegates straight to SQLAgent, unchanged.
  - "semantic" — pure meaning-based lookup over free text (e.g. "which
                 products are described as durable"). Vector search only,
                 no SQL needed.
  - "hybrid"   — needs both: find rows by MEANING, then compute a NUMBER
                 on them (e.g. "total revenue from products described as
                 eco-friendly"). Vector search narrows to matching row IDs
                 first, then those IDs constrain a normal SQLAgent run —
                 so all of SQLAgent's validation/repair machinery (FK-aware
                 joins, semantic checks, tie-aware ranking, etc.) still
                 applies to the SQL half.

Reuses SQLAgent from core_agent.py rather than duplicating its pipeline —
the hybrid case injects the vector-search result as an explicit constraint
in the question text handed to SQLAgent, rather than modifying SQLAgent
itself, keeping the two concerns cleanly decoupled.
"""
import os
import sys
import asyncio
import argparse

from dotenv import load_dotenv
from sqlalchemy import create_engine, text as sqltext
from rich.console import Console
from rich.panel import Panel

from core_agent import SQLAgent, Metrics
from production_agent import build_llm
from vector_store import VectorStore, build_embedding_function

load_dotenv()
console = Console()

DEFAULT_DISTANCE_THRESHOLD = 1.2  # matches worse than this are treated as "not actually relevant"


# ---------------------------------------------------------------------------
# Question routing
# ---------------------------------------------------------------------------

CLASSIFY_PROMPT = """Classify how this question should be answered against a database that has both structured columns and a free-text description column.

Question: {question}

Respond with EXACTLY one word:
- SQL — the question only needs structured data (counts, sums, filters on exact values, dates, IDs). No meaning-based text matching needed.
- SEMANTIC — the question is purely about finding text that matches a MEANING or description (e.g. "which products are described as durable"), with no number to compute afterward.
- HYBRID — the question needs BOTH: finding rows by what their description MEANS, AND THEN computing a number/aggregate on those rows (e.g. "total revenue from products described as eco-friendly").

Respond with only the single word SQL, SEMANTIC, or HYBRID — nothing else."""


async def classify_question(llm, question: str) -> str:
    result = await llm.ainvoke(CLASSIFY_PROMPT.format(question=question))
    text = (result.content if hasattr(result, "content") else str(result)).strip().upper()
    for route in ("HYBRID", "SEMANTIC", "SQL"):
        if route in text:
            return route.lower()
    return "sql"  # fail toward the most conservative, most-tested path


# ---------------------------------------------------------------------------
# The hybrid agent
# ---------------------------------------------------------------------------

class HybridAgent:
    def __init__(self, db_url: str, llm, dialect: str,
                 vector_table: str, vector_text_column: str, vector_id_column: str,
                 embedding_provider: str = "tfidf", embedding_kwargs: dict = None,
                 top_k: int = 5, distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
                 max_retries: int = 2, use_cache: bool = True):
        self.db_url = db_url
        self.llm = llm
        self.dialect = dialect
        self.vector_table = vector_table
        self.vector_id_column = vector_id_column
        self.top_k = top_k
        self.distance_threshold = distance_threshold
        self.max_retries = max_retries
        self.use_cache = use_cache

        engine = create_engine(db_url)
        with engine.connect() as conn:
            rows = [dict(r._mapping) for r in conn.execute(
                sqltext(f"SELECT * FROM {vector_table}")
            )]
        engine.dispose()

        if not rows:
            raise RuntimeError(f"'{vector_table}' has no rows to index for semantic search.")

        embedding_kwargs = embedding_kwargs or {}
        if embedding_provider == "tfidf":
            embedding_kwargs = {**embedding_kwargs, "corpus": [str(r[vector_text_column]) for r in rows]}
        embedder = build_embedding_function(embedding_provider, **embedding_kwargs)

        self.vector_store = VectorStore(f"hybrid_{vector_table}", embedder)
        metadata_fields = [k for k in rows[0].keys() if k not in (vector_id_column, vector_text_column)]
        self.vector_store.index_rows(rows, id_field=vector_id_column, text_field=vector_text_column,
                                      metadata_fields=metadata_fields)

    def semantic_search(self, query: str) -> list[dict]:
        matches = self.vector_store.search(query, top_k=self.top_k)
        return [m for m in matches if m["distance"] <= self.distance_threshold]

    async def run(self, question: str):
        route = await classify_question(self.llm, question)
        console.print(f"[dim]Routed as: {route.upper()}[/dim]")

        if route == "semantic":
            matches = self.semantic_search(question)
            if not matches:
                return (f"No matching {self.vector_table} found for that description.",
                        None, Metrics(), route, [])
            lines = [f"- {m.get('name', m['id'])}: {m['text']}" for m in matches]
            answer = "Matching results:\n" + "\n".join(lines)
            return answer, None, Metrics(), route, matches

        if route == "hybrid":
            matches = self.semantic_search(question)
            if not matches:
                return (f"No matching {self.vector_table} found for that description, "
                        f"so there's nothing to compute.", None, Metrics(), route, [])
            ids = [m["id"] for m in matches]
            names = [m.get("name", m["id"]) for m in matches]
            augmented_question = (
                f"{question}\n\n"
                f"(Semantic search already identified the relevant {self.vector_table} rows "
                f"for this description: {self.vector_id_column} IN ({', '.join(ids)}), "
                f"i.e. {', '.join(names)}. Restrict your query to exactly these rows via "
                f"the {self.vector_id_column} column — do not re-interpret the description "
                f"yourself or select different rows.)"
            )
            agent = SQLAgent(self.db_url, self.llm, self.dialect,
                              max_retries=self.max_retries, use_cache=self.use_cache)
            answer, sql, metrics = await agent.run(augmented_question)
            return answer, sql, metrics, route, matches

        # route == "sql"
        agent = SQLAgent(self.db_url, self.llm, self.dialect,
                          max_retries=self.max_retries, use_cache=self.use_cache)
        answer, sql, metrics = await agent.run(question)
        return answer, sql, metrics, route, []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG + SQL hybrid agent.")
    parser.add_argument("question", type=str)
    parser.add_argument("--db-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "ollama"), choices=["ollama", "anthropic"])
    parser.add_argument("--model", default=os.getenv("LLM_MODEL"))
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--vector-table", required=True, help="Table with the free-text column to search")
    parser.add_argument("--vector-text-column", required=True, help="Column containing the free text")
    parser.add_argument("--vector-id-column", required=True, help="Primary key column of --vector-table")
    parser.add_argument("--embedding-provider", default="tfidf", choices=["tfidf", "ollama"])
    parser.add_argument("--embedding-model", default="nomic-embed-text", help="Model name, if --embedding-provider ollama")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--distance-threshold", type=float, default=DEFAULT_DISTANCE_THRESHOLD)
    args = parser.parse_args()

    if not args.db_url:
        console.print("[bold red]Error:[/bold red] --db-url or DATABASE_URL is required")
        sys.exit(1)

    dialect = "PostgreSQL" if args.db_url.startswith("postgresql") else \
              "MySQL" if args.db_url.startswith("mysql") else \
              "SQLite" if args.db_url.startswith("sqlite") else "SQL"

    console.print(Panel(f"[bold cyan]Question:[/bold cyan] {args.question}", border_style="cyan"))

    try:
        llm = build_llm(args.provider, args.model, reasoning=args.think, max_tokens=args.max_tokens)
        embedding_kwargs = {"model": args.embedding_model} if args.embedding_provider == "ollama" else {}
        agent = HybridAgent(
            args.db_url, llm, dialect,
            vector_table=args.vector_table,
            vector_text_column=args.vector_text_column,
            vector_id_column=args.vector_id_column,
            embedding_provider=args.embedding_provider,
            embedding_kwargs=embedding_kwargs,
            top_k=args.top_k,
            distance_threshold=args.distance_threshold,
        )
        answer, sql, metrics, route, matches = asyncio.run(agent.run(args.question))

        console.print(Panel(f"[bold green]Answer:[/bold green]\n\n{answer}", border_style="green"))
        if sql:
            console.print(Panel(f"[dim]{sql}[/dim]", title="SQL used", border_style="dim"))
        if matches:
            match_lines = "\n".join(f"- {m.get('name', m['id'])} (distance={m['distance']:.3f})" for m in matches)
            console.print(Panel(match_lines, title="Semantic matches used", border_style="blue"))
    except Exception as e:
        console.print(Panel(f"[bold red]Error:[/bold red]\n\n{str(e)}", border_style="red"))
        sys.exit(1)


if __name__ == "__main__":
    main()