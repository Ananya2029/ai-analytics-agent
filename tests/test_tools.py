"""Tests for the SQL guard, the read-only warehouse and the agent loop (LLM stubbed).

    python -m pytest -q
"""
import duckdb
import pytest

from agent import core
from agent.llm import Reply, ToolCall
from agent.tools import SQLRejected, Warehouse, guard


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    path = tmp_path_factory.mktemp("wh") / "mini.duckdb"
    with duckdb.connect(str(path)) as con:
        con.sql("CREATE SCHEMA gold")
        con.sql("""CREATE TABLE gold.fact_orders AS SELECT * FROM (VALUES
                   ('o1', 1, 'delivered', 120.0), ('o2', 2, 'delivered', 80.0), ('o3', 1, 'canceled', 50.0))
                   t(order_id, customer_key, order_status, payment_value)""")
        con.sql("CREATE TABLE gold.dim_customer AS SELECT * FROM (VALUES (1, 'SP'), (2, 'RJ')) t(customer_key, state)")
    return Warehouse(path)


@pytest.mark.parametrize("sql", [
    "SELECT 1",
    "with t as (select 1 as x) select x from t",
    "SELECT * FROM gold.fact_orders WHERE order_status = 'update pending'",   # keyword inside a string is fine
    "-- comment\nSELECT 1;",
])
def test_guard_allows_read_only_queries(sql):
    guard(sql)


@pytest.mark.parametrize("sql", [
    "DROP TABLE gold.fact_orders",
    "DELETE FROM gold.fact_orders",
    "SELECT 1; DROP TABLE gold.fact_orders",
    "COPY gold.fact_orders TO 'out.csv'",
    "SELECT * FROM read_csv('C:/secrets.csv')",
    "ATTACH 'other.db'",
    "INSTALL httpfs",
    "PRAGMA database_list",
    "",
])
def test_guard_rejects_everything_else(sql):
    with pytest.raises(SQLRejected):
        guard(sql)


def test_database_is_read_only_even_without_the_guard(warehouse):
    with pytest.raises(duckdb.Error):
        warehouse.con.sql("DELETE FROM gold.fact_orders")


def test_tools(warehouse):
    assert "gold.fact_orders" in warehouse.list_tables()
    assert "payment_value" in warehouse.describe_table("gold.fact_orders")
    assert warehouse.describe_table("main.secret; drop").startswith("Error")
    df, msg = warehouse.run_sql("SELECT count(*) AS n FROM gold.fact_orders")
    assert df["n"][0] == 3
    df, msg = warehouse.run_sql("SELECT nope FROM gold.fact_orders")
    assert df is None and msg.startswith("SQL error")


def test_agent_recovers_from_a_sql_error(warehouse, monkeypatch):
    script = iter([
        Reply("", [ToolCall("run_sql", {"sql": "SELECT sum(revenue) FROM gold.fact_orders"})]),     # wrong column
        Reply("", [ToolCall("run_sql", {"sql": "SELECT sum(payment_value) AS total FROM gold.fact_orders "
                                                "WHERE order_status NOT IN ('canceled', 'unavailable')"})]),
        Reply("Total revenue is R$ 200."),
    ])
    monkeypatch.setattr(core, "chat", lambda messages, tools, provider=None: next(script))
    r = core.run("What is total revenue?", warehouse)
    assert r.sql_errors == 1
    assert r.final_df["total"][0] == 200
    assert "200" in r.answer


def test_agent_is_nudged_back_when_it_stops_without_data(warehouse, monkeypatch):
    script = iter([
        Reply("I will query the orders table to count them."),                    # talks, no tool call
        Reply("", [ToolCall("run_sql", {"sql": "SELECT count(*) AS n FROM gold.fact_orders"})]),
        Reply("There are 3 orders."),
    ])
    monkeypatch.setattr(core, "chat", lambda messages, tools, provider=None: next(script))
    r = core.run("How many orders?", warehouse)
    assert r.nudges == 1 and r.final_df["n"][0] == 3 and r.answer == "There are 3 orders."
