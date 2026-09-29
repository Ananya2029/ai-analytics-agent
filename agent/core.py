"""
The agent loop: the LLM plans, calls tools (inspect schema, run SQL), reads the results,
fixes its own SQL errors, and finally answers in plain language.

    python -m agent.core "Which 5 product categories earned the most revenue?"
"""
import sys
import time
from dataclasses import dataclass, field

import pandas as pd

from agent.config import LLM_PROVIDER, MAX_STEPS
from agent.llm import chat, tool_message
from agent.tools import TOOL_SPECS, Warehouse

SYSTEM = """You are a data analyst for Olist, a Brazilian e-commerce marketplace. Answer the user's
business question by querying a DuckDB star-schema warehouse with the tools provided.

Warehouse notes:
- Tables live in schema "gold": fact_orders (one row per order), fact_sales (one row per order item),
  dim_customer, dim_seller, dim_product, dim_date. Join facts to dimensions on *_key columns;
  join fact_sales to dim_date with purchase_date_key = date_key.
- A customer is a person: dim_customer.customer_key (one per real customer).
- Revenue: sum(fact_sales.price) for product revenue, or sum(fact_orders.payment_value) for money paid.
  Unless the user says otherwise, exclude orders with order_status 'canceled' or 'unavailable'.
- Money is in Brazilian reais (R$).

How to work:
1. If unsure about columns, call describe_table first. Do not guess column names.
2. Write ONE DuckDB SELECT query with run_sql. If it errors, read the error, fix the query and retry.
3. When you have the result, answer in 1-3 sentences with the key numbers. Do not invent numbers
   that are not in the query result. If the data cannot answer the question, say so."""


@dataclass
class Step:
    tool: str
    args: dict
    output: str
    ok: bool


@dataclass
class Result:
    question: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    final_sql: str | None = None
    final_df: pd.DataFrame | None = None
    seconds: float = 0.0
    nudges: int = 0          # times the agent stopped early and was sent back to the tools

    @property
    def sql_errors(self):
        return sum(1 for s in self.steps if s.tool == "run_sql" and not s.ok)


def run(question: str, warehouse: Warehouse, provider: str = LLM_PROVIDER, max_steps: int = MAX_STEPS) -> Result:
    t0 = time.time()
    # v2: the schema goes into the prompt up front, so the model does not guess column names
    system = SYSTEM + "\n\nWarehouse schema (table(column type, ...)):\n" + warehouse.schema_summary()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    res = Result(question, "")
    for _ in range(max_steps):
        reply = chat(messages, TOOL_SPECS, provider)
        messages.append(reply.raw or {"role": "assistant", "content": reply.text})
        if not reply.calls:
            last_failed = bool(res.steps) and not res.steps[-1].ok
            if res.final_df is None or last_failed:
                # v2: small models often *say* they will fix the query but stop without calling the
                # tool, or answer without any data. Push them back to the tool instead of accepting it.
                res.nudges += 1
                messages.append({"role": "user", "content":
                                 "You have not answered from a successful query yet. Call run_sql now with a "
                                 "corrected query (check the column names in the schema) instead of describing it."})
                continue
            res.answer = reply.text.strip()
            break
        for call in reply.calls:
            if call.name == "list_tables":
                out, ok = warehouse.list_tables(), True
            elif call.name == "describe_table":
                out = warehouse.describe_table(call.args.get("table", ""))
                ok = not out.startswith("Error")
            elif call.name == "run_sql":
                df, out = warehouse.run_sql(call.args.get("sql", ""))
                ok = df is not None
                if ok:
                    res.final_sql, res.final_df = call.args["sql"], df
            else:
                out, ok = f"Unknown tool {call.name}", False
            res.steps.append(Step(call.name, call.args, out, ok))
            messages.append(tool_message(call, out))
    else:
        res.answer = "I could not finish within the step limit."
    res.seconds = time.time() - t0
    return res


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Which 5 product categories earned the most revenue?"
    r = run(q, Warehouse())
    for s in r.steps:
        print(f"-> {s.tool}({s.args})\n{s.output[:400]}\n")
    print(f"ANSWER ({r.seconds:.0f}s): {r.answer}")
