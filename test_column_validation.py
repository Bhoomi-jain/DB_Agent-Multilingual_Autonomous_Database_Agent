"""Unit tests for validate_columns_exist — stage-4D column-existence check.
Pure AST-vs-schema logic; no database needed."""

from core_agent import validate_columns_exist, SemanticValidationError

_SCHEMAS = {
    "invoice": [{"name": "invoice_id", "type": "INTEGER", "primary_key": True},
                {"name": "customer_id", "type": "INTEGER", "primary_key": False},
                {"name": "total", "type": "NUMERIC(10,2)", "primary_key": False}],
    "invoiceline": [{"name": "line_id", "type": "INTEGER", "primary_key": True},
                    {"name": "invoice_id", "type": "INTEGER", "primary_key": False},
                    {"name": "quantity", "type": "INTEGER", "primary_key": False},
                    {"name": "unit_price", "type": "NUMERIC(10,2)", "primary_key": False}],
    "customer": [{"name": "customer_id", "type": "INTEGER", "primary_key": True},
                 {"name": "first_name", "type": "NVARCHAR(40)", "primary_key": False}],
}


def expect_reject(label, sql):
    try:
        validate_columns_exist(sql, _SCHEMAS, "postgres")
        raise AssertionError(f"FALSE NEGATIVE on '{label}': {sql!r}")
    except SemanticValidationError as e:
        print(f"rejected as expected: {label} :: {str(e)[:75]}...")


def expect_pass(label, sql):
    validate_columns_exist(sql, _SCHEMAS, "postgres")
    print(f"passed as expected: {label}")


# The exact live failure: InvoiceLine has NO Total column (it's on invoice).
expect_reject("qualified nonexistent column",
              "SELECT SUM(IL.Total) FROM invoiceline IL")

# Close-match suggestion quality: 'totl' should point at 'total'.
try:
    validate_columns_exist("SELECT SUM(totl) FROM invoice", _SCHEMAS, "postgres")
except SemanticValidationError as e:
    assert "total" in str(e), f"suggestion missing: {e}"
    print("rejected as expected: typo column suggests closest real column")

# Ambiguous unqualified column present on BOTH joined tables.
expect_reject("ambiguous unqualified column",
              "SELECT customer_id FROM invoice i "
              "JOIN customer c ON c.customer_id = i.customer_id")

# Unqualified but UNIQUE among the queried tables: perfectly legal SQL.
expect_pass("unqualified unique column passes",
            "SELECT total FROM invoice")
# Same ambiguous name, but only ONE of the two owners is actually joined.
expect_pass("name exists elsewhere, single owner in query passes",
            "SELECT customer_id FROM invoice")

# Column from a table NOT part of the query gets a targeted message.
expect_reject("column belongs to an absent table",
              "SELECT quantity FROM invoice")

# Derived-table qualifiers are out of our jurisdiction: skip silently.
expect_pass("derived-table alias skipped",
            "SELECT x.tot FROM (SELECT total AS tot FROM invoice) x")

# Stars carry no column names: nothing to check.
expect_pass("star select passes",
            "SELECT * FROM invoice")

print("\n--- validate_columns_exist: all existence/ambiguity checks behave ---")
