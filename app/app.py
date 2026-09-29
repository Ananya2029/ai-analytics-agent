"""
Ask business questions in plain English; watch the agent inspect the warehouse, write SQL,
recover from errors and answer.

    streamlit run app/app.py
"""
import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.config import LLM_PROVIDER  # noqa: E402
from agent.core import run  # noqa: E402
from agent.llm import LLMError, model_name  # noqa: E402
from agent.tools import Warehouse  # noqa: E402

st.set_page_config(page_title="AI Analytics Agent", page_icon="🤖", layout="wide")


@st.cache_resource
def warehouse():
    return Warehouse()


EXAMPLES = [
    "Which 5 product categories earned the most revenue?",
    "How does the average review score differ between late and on-time deliveries?",
    "What was the monthly number of orders in 2018?",
    "Which customer states have the longest average delivery time?",
]

st.title("🤖 AI Analytics Agent")
st.caption(f"Ask about the Olist e-commerce warehouse. LLM: {LLM_PROVIDER} · {model_name()} · "
           "read-only access, SELECT queries only")

with st.sidebar:
    st.header("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, width="stretch"):
            st.session_state.q = ex
    st.divider()
    st.caption("Tables available")
    st.code(warehouse().list_tables(), language=None)

question = st.chat_input("Ask a business question") or st.session_state.pop("q", None)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking, querying the warehouse..."):
                r = run(question, warehouse())
        except LLMError as e:
            st.error(str(e))
            st.stop()
        st.markdown(r.answer or "_No answer._")
        df = r.final_df
        if df is not None and len(df) > 1 and df.shape[1] >= 2:
            num = df.select_dtypes("number").columns
            if len(num):
                st.plotly_chart(px.bar(df.head(20), x=df.columns[0], y=num[-1]), width="stretch")
        if df is not None:
            st.dataframe(df, hide_index=True)
        with st.expander(f"Agent steps ({len(r.steps)} tool calls, {r.sql_errors} SQL errors fixed, {r.seconds:.0f}s)"):
            for i, s in enumerate(r.steps, 1):
                icon = "✅" if s.ok else "❌"
                st.markdown(f"**{i}. {icon} `{s.tool}`**")
                if s.tool == "run_sql":
                    st.code(s.args.get("sql", ""), language="sql")
                elif s.args:
                    st.code(str(s.args), language=None)
                st.caption(s.output[:700])
