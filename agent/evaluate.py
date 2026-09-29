"""
Evaluate the agent on eval/questions.json.

    python -m agent.evaluate            # all questions
    python -m agent.evaluate --limit 5

Correct = the expected value appears in the agent's final query result or in its answer
(numbers within 1%; a fraction like 0.068 also matches an expected 6.8 %).
"""
import argparse
import json
import re
import time

import numpy as np
import pandas as pd

from agent.config import EVAL_DIR, LLM_PROVIDER, REPORT_DIR
from agent.core import run
from agent.llm import model_name
from agent.tools import Warehouse


def numbers_in(text: str):
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", text or "")]


def is_correct(expected, answer: str, df: pd.DataFrame | None) -> bool:
    if isinstance(expected, str):
        hay = (answer or "").lower()
        if df is not None:
            hay += " " + df.astype(str).to_string().lower()
        return expected.lower() in hay.replace(" ", "_") or expected.lower() in hay
    candidates = numbers_in(answer)
    if df is not None:
        candidates += [float(v) for v in df.select_dtypes("number").to_numpy().ravel() if pd.notna(v)]
    tol = max(abs(expected) * 0.01, 0.01)
    return any(abs(c - expected) <= tol or abs(c * 100 - expected) <= tol for c in candidates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    REPORT_DIR.mkdir(exist_ok=True)
    questions = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))["questions"][:args.limit]
    wh = Warehouse()
    rows = []
    t0 = time.time()
    for i, q in enumerate(questions, 1):
        try:
            r = run(q["q"], wh)
            ok = is_correct(q["expected"], r.answer, r.final_df)
            rows.append({"question": q["q"], "expected": q["expected"], "correct": ok, "answer": r.answer,
                         "final_sql": r.final_sql, "tool_calls": len(r.steps), "sql_errors": r.sql_errors, "nudges": r.nudges,
                         "seconds": round(r.seconds, 1)})
        except Exception as e:  # count crashes as failures, keep evaluating
            rows.append({"question": q["q"], "expected": q["expected"], "correct": False,
                         "answer": f"ERROR: {e}", "final_sql": None, "tool_calls": 0, "sql_errors": 0, "nudges": 0, "seconds": 0})
        x = rows[-1]
        print(f"[{i:2d}] {'OK  ' if x['correct'] else 'MISS'} {x['seconds']:6.1f}s  calls {x['tool_calls']}  "
              f"errors {x['sql_errors']}  {q['q'][:60]}", flush=True)
    df = pd.DataFrame(rows)
    summary = {
        "llm": f"{LLM_PROVIDER}:{model_name()}",
        "questions": len(df),
        "accuracy": round(float(df["correct"].mean()), 3),
        "avg_tool_calls": round(float(df["tool_calls"].mean()), 2),
        "questions_with_sql_error": int((df["sql_errors"] > 0).sum()),
        "recovered_after_sql_error": int(((df["sql_errors"] > 0) & df["correct"]).sum()),
        "median_seconds": float(np.median(df["seconds"])),
        "total_minutes": round((time.time() - t0) / 60, 1),
    }
    print(json.dumps(summary, indent=2))
    tag = f"{LLM_PROVIDER}_{model_name().replace(':', '-').replace('/', '-')}"
    df.to_csv(REPORT_DIR / f"eval_details_{tag}.csv", index=False)
    (REPORT_DIR / f"eval_summary_{tag}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
