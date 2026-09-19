"""Unit tests for repair_string_concat — the T-SQL '+' -> '||' dialect fix.
Covers §6.23 plus the unqualified-column gap found live (§6.24): llama3.2
emits `FirstName + ' ' + LastName` with NO table qualifiers, which the first
draft skipped because operands couldn't be alias-resolved."""

from core_agent import repair_string_concat

_SCHEMAS = {
    "customer": [{"name": "customer_id", "type": "INTEGER", "primary_key": True},
                 {"name": "first_name", "type": "NVARCHAR(40)", "primary_key": False},
                 {"name": "last_name", "type": "NVARCHAR(40)", "primary_key": False}],
    "invoice": [{"name": "invoice_id", "type": "INTEGER", "primary_key": True},
                {"name": "customer_id", "type": "INTEGER", "primary_key": False}],
}


def expect_concat(label, sql):
    out = repair_string_concat(sql, _SCHEMAS, "sqlite")
    assert "||" in out, f"'{label}': expected || rewrite, got: {out}"
    assert " + " not in out.replace("++", ""), f"'{label}': + survived: {out}"
    print(f"rewritten: {label} :: {out[:70]}")


def expect_untouched(label, sql):
    out = repair_string_concat(sql, _SCHEMAS, "sqlite")
    assert out == sql, f"'{label}': should be untouched, got: {out}"
    print(f"untouched as expected: {label}")


# Qualified columns (original case).
expect_concat("qualified two-column concat",
              "SELECT c.first_name + ' ' + c.last_name FROM customer c")

# UNQUALIFIED columns — the live failure shape; resolvable by unique name.
expect_concat("unqualified columns resolved by unique schema name",
              "SELECT first_name + ' ' + last_name AS CustomerName FROM customer")

# Mixed column + string literal.
expect_concat("column + literal chain",
              "SELECT first_name + ', ' FROM customer")

# Numeric addition must NEVER be touched.
expect_untouched("numeric addition stays",
                 "SELECT quantity + 1 FROM invoice")
expect_untouched("numeric column + numeric column stays",
                 "SELECT invoice_id + customer_id FROM invoice")

# Ambiguous unqualified name (customer_id lives in BOTH tables): can't
# attribute a type -> conservative bail-out, house "never guess" rule.
expect_untouched("ambiguous unqualified name bails",
                 "SELECT customer_id + 1 FROM invoice i JOIN customer c "
                 "ON c.customer_id = i.customer_id")

print("\n--- repair_string_concat: all dialect-repair behaviors verified ---")
