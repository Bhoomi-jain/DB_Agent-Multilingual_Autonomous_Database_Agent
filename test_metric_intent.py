import asyncio

import sqlglot
from sqlglot import exp

from core_agent import SQLAgent, validate_metric_intent, SemanticValidationError
from db_targets import PG_URL as DB_URL


class FakeMsg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return FakeMsg(self.responses.pop(0))


# The exact live failure: question asks about MONEY ("spent"), model
# counted invoices instead. Every customer had 7 invoices, so the answer
# was plausible-looking and fully "verified" — wrong measure entirely.
COUNT_SQL = """```sql
SELECT c.name, COUNT(o.order_id) AS cnt
FROM customers c JOIN orders o ON o.customer_id = c.customer_id
GROUP BY c.name ORDER BY cnt DESC LIMIT 5
```"""

GOOD_SQL = """```sql
SELECT c.name, SUM(oi.quantity * oi.unit_price) AS spent
FROM customers c JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON oi.order_id = o.order_id
GROUP BY c.name ORDER BY spent DESC
```"""


async def main():
    llm = FakeLLM([
        COUNT_SQL,   # generate attempt 1: counts instead of summing
        GOOD_SQL,    # generate attempt 2: corrected after intent feedback
        "Alice spent the most.",  # format_answer
    ])
    agent = SQLAgent(db_url=DB_URL, llm=llm, dialect="PostgreSQL",
                     max_retries=1, use_cache=False)
    answer, sql, metrics = await agent.run("Which customer spent the most?")

    print("ANSWER:", answer)
    print("FINAL SQL:", " ".join(sql.split()))
    print(metrics.summary())

    assert metrics.semantic_rejections == 1, (
        f"expected exactly 1 semantic rejection (COUNT instead of SUM), "
        f"got {metrics.semantic_rejections}"
    )
    assert metrics.retries == 1, f"expected 1 retry, got {metrics.retries}"
    # Retry prompt must name the exact confusion and the fix.
    retry_prompt = llm.prompts[1]
    assert "monetary AMOUNT" in retry_prompt and "spent" in retry_prompt, (
        "retry prompt should quote the money term from the question"
    )
    assert "COUNT(" in retry_prompt and "SUM(" in retry_prompt
    # Final SQL must actually aggregate money now.
    ast = sqlglot.parse_one(sql, read="postgres")
    assert any(isinstance(a, exp.Sum) for a in ast.find_all(exp.AggFunc)), (
        f"final query should use SUM(), got: {sql}"
    )
    assert metrics.answer_verified is True
    assert "Alice" in answer
    print("\n--- metric-intent validator caught COUNT-instead-of-SUM and forced a corrective retry ---")

    # ---- Negative controls: must NOT reject legitimate combinations ----
    negatives = [
        ("how-many + COUNT",
         "How many customers are there?",
         "SELECT COUNT(*) FROM customers"),
        ("'total number of' is NOT a money term",
         "What is the total number of invoices?",
         "SELECT COUNT(*) FROM orders"),
        ("average + AVG",
         "What is the average invoice total?",
         "SELECT AVG(order_id) FROM orders"),
        ("no aggregation at all",
         "Which customers had orders shipped via UPS?",
         "SELECT DISTINCT c.name FROM customers c JOIN orders o ON o.customer_id = c.customer_id"),
    ]
    for label, q, s in negatives:
        try:
            validate_metric_intent(q, s, "postgres")
            print(f"negative control passed (not flagged): {label}")
        except SemanticValidationError as e:
            raise AssertionError(f"FALSE POSITIVE on '{label}': {e}")

    # Positive unit control: the inverse confusion (SUM on a how-many question)
    try:
        validate_metric_intent(
            "How many orders are there?",
            "SELECT SUM(customer_id) FROM orders", "postgres")
        raise AssertionError("expected rejection for SUM on a how-many question")
    except SemanticValidationError:
        print("positive unit control passed: SUM rejected on how-many question")


asyncio.run(main())
