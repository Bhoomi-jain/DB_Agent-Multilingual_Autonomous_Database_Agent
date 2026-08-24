import asyncio
from core_agent import SQLAgent
from multilingual import MultilingualAgent, detect_language
from db_targets import SQLITE_URL


class FakeMsg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Scripted responses consumed in call order; every prompt recorded so
    the tests can assert what each LLM call was actually asked to do."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return FakeMsg(self.responses.pop(0))


GOOD_SQL = "```sql\nSELECT COUNT(*) AS n FROM customers WHERE country = 'Canada'\n```"
EN_ANSWER = "There is 1 customer from Canada."

# Long, unambiguous Spanish sentence — langdetect is unreliable on short or
# punctuation-only inputs, so the test question is deliberately a full one.
ES_QUESTION = "¿Cuántos clientes son de Canadá según la base de datos?"


async def test_nonenglish_roundtrip():
    print("=== Spanish question -> English reasoning -> Spanish answer ===")
    llm = FakeLLM([
        "How many customers are from Canada?",   # translate_in
        GOOD_SQL,                                 # generate_sql
        EN_ANSWER,                                # format_answer
        "Hay 1 cliente de Canadá.",               # translate_out
    ])
    inner = SQLAgent(db_url=SQLITE_URL, llm=llm, dialect="SQLite",
                     max_retries=1, use_cache=False)
    agent = MultilingualAgent(inner, llm)
    answer, sql, metrics, language = await agent.run(ES_QUESTION)

    print("language:", language)
    print("answer:", answer)
    print(metrics.summary())

    assert language == "Spanish", f"expected Spanish detection, got {language!r}"
    assert answer == "Hay 1 cliente de Canadá.", f"unexpected final answer: {answer!r}"
    assert sql is not None and "COUNT" in sql.upper()

    # The pipeline ran entirely on the ENGLISH translation internally.
    assert llm.prompts[0].startswith("Translate") and "English" in llm.prompts[0], (
        "first LLM call should be the translate-to-English step"
    )
    assert ES_QUESTION in llm.prompts[0]
    assert "Schema (only the relevant tables):" in llm.prompts[1] and "customers(" in llm.prompts[1], (
        "SQL generation should have been prompted with the English question's schema"
    )
    assert llm.prompts[-2].splitlines()[0].strip() == f"Question: How many customers are from Canada?", (
        "the inner agent must reason over the TRANSLATED question, not the original"
    )
    # Translate-out targeted the detected language by name.
    assert "into Spanish" in llm.prompts[-1]

    # Honest accounting: translate_in + generate + format + translate_out.
    assert metrics.llm_calls == 4, f"expected 4 LLM calls total, got {metrics.llm_calls}"
    assert metrics.answer_verified is True
    print("PASS\n")


async def test_english_passthrough():
    print("=== English question -> no translation calls at all ===")
    llm = FakeLLM([
        GOOD_SQL,
        EN_ANSWER,
    ])
    inner = SQLAgent(db_url=SQLITE_URL, llm=llm, dialect="SQLite",
                     max_retries=1, use_cache=False)
    agent = MultilingualAgent(inner, llm)
    answer, sql, metrics, language = await agent.run(
        "How many customers are from Canada according to the database?"
    )

    print("language:", language)
    print("answer:", answer)
    assert language == "English", f"expected English passthrough, got {language!r}"
    assert answer == EN_ANSWER
    assert metrics.llm_calls == 2, (
        f"English questions should make NO translation calls "
        f"(generate + format only), got {metrics.llm_calls}"
    )
    assert not any(p.startswith("Translate") for p in llm.prompts), (
        "no prompt should be a translation prompt for an English question"
    )
    print("PASS\n")


async def test_translation_figure_guard():
    print("=== Translation dropping a figure triggers the integrity warning ===")
    llm = FakeLLM([
        "How many customers are from Canada?",
        GOOD_SQL,
        EN_ANSWER,
        # Translate-out mangles/drops the number ("un" instead of "1") —
        # the exact class of silent corruption the guard exists for.
        "Hay un cliente de Canadá.",
    ])
    inner = SQLAgent(db_url=SQLITE_URL, llm=llm, dialect="SQLite",
                     max_retries=1, use_cache=False)
    agent = MultilingualAgent(inner, llm)
    answer, sql, metrics, language = await agent.run(ES_QUESTION)

    print("answer:", answer)
    assert "possibly altered by translation" in answer, (
        "a translation that loses a verified figure must surface a warning, "
        "not silently ship the corrupted text"
    )
    assert EN_ANSWER in answer, "the original verified English figures should be preserved alongside"
    print("PASS\n")


async def main():
    assert detect_language(ES_QUESTION) == "es", (
        "precondition: langdetect should identify the test question as Spanish"
    )
    await test_nonenglish_roundtrip()
    await test_english_passthrough()
    await test_translation_figure_guard()
    print("--- multilingual wrapper verified: round-trip, passthrough, figure-guard ---")


asyncio.run(main())
