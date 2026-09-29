"""
The agent's tools, and the guard rails around them.

Safety model (defence in depth):
  1. The warehouse is opened READ-ONLY, so writes are impossible at the database level.
  2. A SQL guard rejects anything that is not a single SELECT / WITH query before it runs.
  3. Results are capped at MAX_ROWS and queries are interrupted after QUERY_TIMEOUT_S.
"""
import re
import threading

import duckdb

from agent.config import MAX_ROWS, QUERY_TIMEOUT_S, WAREHOUSE

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|export|import|install|load|"
    r"pragma|set|call|checkpoint|vacuum|grant|truncate|replace)\b|read_csv|read_parquet|read_json|glob\(",
    re.IGNORECASE)


class SQLRejected(ValueError):
    pass


def strip_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    return re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)


def guard(sql: str) -> str:
    """Allow exactly one read-only SELECT/WITH statement; raise SQLRejected otherwise."""
    body = strip_comments(sql).strip().rstrip(";").strip()
    if not body:
        raise SQLRejected("Empty query.")
    if ";" in body:
        raise SQLRejected("Only one statement is allowed.")
    if not re.match(r"^(select|with)\b", body, re.IGNORECASE):
        raise SQLRejected("Only SELECT (or WITH ... SELECT) queries are allowed.")
    # String literals may legitimately contain words like 'set' or 'create'; check outside them
    if FORBIDDEN.search(re.sub(r"'(?:[^']|'')*'", "''", body)):
        raise SQLRejected("The query uses a statement or function that is not allowed.")
    return body


class Warehouse:
    def __init__(self, path=WAREHOUSE):
        if not path.exists():
            raise FileNotFoundError(
                f"Warehouse not found at {path}. Build it with the ecommerce-data-pipeline project "
                "(python -m pipeline.run) or set WAREHOUSE_PATH.")
        self.con = duckdb.connect(str(path), read_only=True)

    # ------------------------------------------------------------ tools
    def list_tables(self) -> str:
        rows = self.con.sql("""
            SELECT schema_name || '.' || table_name, estimated_size
            FROM duckdb_tables() WHERE schema_name = 'gold' ORDER BY 1""").fetchall()
        return "\n".join(f"{t} (~{n:,} rows)" for t, n in rows)

    def schema_summary(self) -> str:
        """Compact 'table(col type, ...)' list of the gold schema, given to the agent up front."""
        rows = self.con.sql("""SELECT table_name, string_agg(column_name || ' ' || lower(data_type), ', '
                                      ORDER BY ordinal_position)
                               FROM information_schema.columns WHERE table_schema = 'gold'
                               GROUP BY 1 ORDER BY 1""").fetchall()
        return "\n".join(f"gold.{t}({cols})" for t, cols in rows)

    def describe_table(self, table: str) -> str:
        if not re.fullmatch(r"gold\.[a-z_]+", table.strip()):
            return "Error: use a gold table name like gold.fact_orders (see list_tables)."
        schema, name = table.strip().split(".")
        cols = self.con.execute("""SELECT column_name, data_type FROM information_schema.columns
                                   WHERE table_schema = ? AND table_name = ? ORDER BY ordinal_position""",
                                [schema, name]).fetchall()
        if not cols:
            return f"Error: table {table} does not exist."
        sample = self.con.sql(f"SELECT * FROM {schema}.{name} LIMIT 3").df().to_string(index=False, max_colwidth=30)
        return "Columns:\n" + "\n".join(f"  {c} ({t})" for c, t in cols) + f"\nSample rows:\n{sample}"

    def run_sql(self, sql: str):
        """Returns (dataframe, message). The dataframe is None when the query was rejected or failed."""
        try:
            body = guard(sql)
        except SQLRejected as e:
            return None, f"Rejected: {e}"
        timer = threading.Timer(QUERY_TIMEOUT_S, self.con.interrupt)
        timer.start()
        try:
            df = self.con.sql(f"SELECT * FROM ({body}) LIMIT {MAX_ROWS + 1}").df()
        except duckdb.Error as e:
            return None, f"SQL error: {str(e).splitlines()[0][:300]}"
        finally:
            timer.cancel()
        note = f" (showing first {MAX_ROWS})" if len(df) > MAX_ROWS else ""
        df = df.head(MAX_ROWS)
        return df, f"{len(df)} rows{note}:\n{df.to_string(index=False, max_colwidth=40)}"


TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "list_tables", "description": "List the warehouse tables (a star schema) with row counts.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "describe_table", "description": "Show the columns, types and 3 sample rows of one table.",
        "parameters": {"type": "object", "properties": {
            "table": {"type": "string", "description": "e.g. gold.fact_orders"}}, "required": ["table"]}}},
    {"type": "function", "function": {
        "name": "run_sql",
        "description": "Run ONE read-only DuckDB SELECT query and get the result rows (max 50).",
        "parameters": {"type": "object", "properties": {
            "sql": {"type": "string", "description": "A single SELECT or WITH query"}}, "required": ["sql"]}}},
]
