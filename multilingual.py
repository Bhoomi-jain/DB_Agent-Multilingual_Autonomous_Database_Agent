"""
multilingual.py — The project's namesake feature, ported onto the current
engine.

The original agent.py had language detection + translation wired directly
into its (now superseded) create_agent() loop. When core_agent.py replaced
that loop with the explicit control-flow pipeline, the multilingual layer
was left behind on the dead entry point. This module restores it as a thin,
agent-agnostic WRAPPER rather than baking it into core_agent.py again:

    question ──► langdetect ──► [LLM] translate to English (if needed)
                                        │
                                        ▼
                        inner agent (SQLAgent / anything with a
                        compatible run()) reasons entirely in English —
                        schema filtering, SQL generation, validation,
                        repair, retries, answer verification all apply
                        UNCHANGED to the translated question
                                        │
                                        ▼
             [LLM] translate answer back ──► figure-integrity check ──► you

Why a wrapper and not integration:
  - core_agent's whole design philosophy is an explicit, inspectable
    pipeline; bolting translation steps into run() would entangle two
    orthogonal concerns.
  - Any agent whose run(question) returns (answer, sql, metrics) — i.e.
    SQLAgent — works unmodified. HybridAgent's 5-tuple also works; only its
    first three elements are surfaced.

Reliability detail carried over from the observed-failure log: LLM
translation can silently mangle figures (e.g. "59.88" -> "59,88" under a
locale convention, or dropping a number entirely). After translating the
answer BACK, every numeric figure present in the verified English answer is
checked against the translation; any that vanished are called out explicitly
instead of being lost quietly — same "diagnose loudly, never silently"
stance as the rest of this codebase.
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from langdetect import DetectorFactory, detect
from rich.console import Console
from rich.panel import Panel

from core_agent import SQLAgent, Metrics, _answer_figures, _strip_thinking
from production_agent import build_llm
from db_targets import PG_URL

# Make langdetect deterministic (it seeds off wall-clock jitter by default).
DetectorFactory.seed = 0

load_dotenv()
console = Console()

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

TRANSLATE_PROMPT = (
    "Translate the following text into {target}. "
    "Output ONLY the translation, with no preamble, quotes, or notes.\n\n"
    "{text}"
)


def detect_language(text: str) -> str:
    """Best-effort language-code detection; falls back to English on the
    many inputs langdetect chokes on (very short text, heavy punctuation,
    code-mixed questions). A wrong 'en' guess costs nothing: the question is
    already in English, so no translation happens."""
    try:
        return detect(text)
    except Exception:
        return "en"


class MultilingualAgent:
    """Wraps an inner agent so users can ask in any language while the
    entire reasoning pipeline runs in English underneath."""

    def __init__(self, inner_agent, llm):
        self.inner = inner_agent
        self.llm = llm
        # Translation calls happen BEFORE the inner run exists to report
        # them (and after its Metrics is already returned), so they're
        # accumulated here and folded into whatever Metrics the inner agent
        # returns — one honest total instead of two partial ones.
        self._translation_metrics = Metrics()

    async def _translate(self, text: str, target: str, step: str) -> str:
        import time
        t0 = time.time()
        result = await self.llm.ainvoke(
            TRANSLATE_PROMPT.format(target=target, text=text)
        )
        self._translation_metrics.llm_calls += 1
        self._translation_metrics.record(f"llm:{step}", time.time() - t0)
        content = result.content if hasattr(result, "content") else str(result)
        # qwen3 leaks <think> blocks even with reasoning=False (see
        # core_agent._strip_thinking); translations must be clean too.
        return _strip_thinking(content)

    async def run(self, question: str):
        """Returns (answer, sql, metrics, language_name). The answer is in
        the question's language; sql and metrics come straight from the
        inner English-language run (with translation cost folded in)."""
        lang_code = detect_language(question)
        lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)

        if lang_code != "en":
            console.print(f"[dim]Detected language: {lang_name} — translating to English...[/dim]")
            english_question = await self._translate(question, "English", "translate_in")
            console.print(f"[dim]English question: {english_question}[/dim]")
        else:
            english_question = question

        result = await self.inner.run(english_question)
        english_answer = result[0]
        sql = result[1] if len(result) > 1 else None
        metrics = result[2] if len(result) > 2 else Metrics()

        if lang_code != "en":
            console.print(f"[dim]Translating answer back to {lang_name}...[/dim]")
            answer = await self._translate(english_answer, lang_name, "translate_out")
            # Figure-integrity check across the translation boundary. The
            # English answer was already verified against the raw query
            # results inside the inner agent; translation is the last place
            # a correct number can still be corrupted or dropped. Uses the
            # same normalized float extraction as answer verification.
            en_figs = {v for v, _dec in _answer_figures(english_answer)}
            tr_figs = {v for v, _dec in _answer_figures(answer)}
            missing = sorted(en_figs - tr_figs)
            if missing:
                console.print(
                    f"[yellow]Warning:[/yellow] figure(s) {sorted(missing)} present in the "
                    f"verified English answer are missing/altered in the {lang_name} "
                    f"translation — showing the English figures alongside."
                )
                answer += (
                    f"\n\n⚠️ Note (figures possibly altered by translation): "
                    f"{english_answer}"
                )
        else:
            answer = english_answer

        # Fold this wrapper's LLM work into the single summary the caller
        # sees — done LAST so the back-translation above is counted too.
        metrics.llm_calls += self._translation_metrics.llm_calls
        for step_name, secs in self._translation_metrics.step_timings.items():
            metrics.record(step_name, secs)
        self._translation_metrics = Metrics()  # fresh per run()

        return answer, sql, metrics, lang_name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_db_url() -> str:
    """DATABASE_URL if set; otherwise the bundled Chinook sample database —
    preserving the original demo experience (ask multilingual questions
    about Chinook) without needing a live Postgres/MySQL server."""
    if os.getenv("DATABASE_URL"):
        return os.getenv("DATABASE_URL")
    here = os.path.dirname(os.path.abspath(__file__))
    return "sqlite:///" + os.path.join(here, "chinook.db")


def _dialect_for(db_url: str) -> str:
    if db_url.startswith("postgresql"):
        return "PostgreSQL"
    if db_url.startswith("mysql"):
        return "MySQL"
    if db_url.startswith("sqlite"):
        return "SQLite"
    return "SQL"


def main():
    parser = argparse.ArgumentParser(
        description="Multilingual front-end for the explicit-control-loop "
                    "text-to-SQL agent (core_agent.SQLAgent).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python multilingual.py "What are the top 5 best-selling artists?"
  python multilingual.py "¿Cuáles son los 5 álbumes más vendidos?"
  python multilingual.py "कनाडा से कितने ग्राहक हैं?"

  python multilingual.py "Top customers by revenue?" \\
      --db-url "postgresql+psycopg2://user:pass@localhost/mydb"
        """,
    )
    parser.add_argument("question", type=str,
                        help="Natural-language question, in any supported language")
    parser.add_argument("--db-url", default=_default_db_url(),
                        help="SQLAlchemy URL (default: $DATABASE_URL, else the "
                             "bundled chinook.db)")
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "ollama"),
                         choices=["ollama", "anthropic"])
    parser.add_argument("--model", default=os.getenv("LLM_MODEL"))
    parser.add_argument("--think", action="store_true",
                        help="Request the model's 'thinking' mode "
                             "(model-dependent; ignored by models without one, "
                             "like Llama 3.2)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--cache-ttl", type=int, default=300)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.getLogger("core_agent").setLevel(logging.INFO)

    console.print(Panel(f"[bold cyan]Question:[/bold cyan] {args.question}",
                        border_style="cyan"))

    try:
        llm = build_llm(args.provider, args.model,
                        reasoning=args.think, max_tokens=args.max_tokens)
        inner = SQLAgent(
            args.db_url, llm, _dialect_for(args.db_url),
            max_retries=args.max_retries,
            cache_ttl=args.cache_ttl,
            use_cache=not args.no_cache,
        )
        agent = MultilingualAgent(inner, llm)
        answer, sql, metrics, language = asyncio.run(agent.run(args.question))

        console.print(Panel(f"[bold green]Answer ({language}):[/bold green]\n\n{answer}",
                            border_style="green"))
        if sql:
            console.print(Panel(f"[dim]{sql}[/dim]", title="SQL used", border_style="dim"))
        if args.metrics:
            console.print(Panel(metrics.summary(), title="Metrics", border_style="blue"))
    except Exception as e:
        console.print(Panel(f"[bold red]Error:[/bold red]\n\n{str(e)}", border_style="red"))
        sys.exit(1)


if __name__ == "__main__":
    main()
