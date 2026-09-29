"""Settings. The warehouse is the DuckDB file built by the ecommerce-data-pipeline project."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

WAREHOUSE = Path(os.environ.get(
    "WAREHOUSE_PATH", ROOT.parent / "ecommerce-data-pipeline" / "warehouse" / "olist.duckdb"))
EVAL_DIR = ROOT / "eval"
REPORT_DIR = ROOT / "reports"

MAX_STEPS = 8          # tool calls per question before the agent must answer
MAX_ROWS = 50          # rows returned to the model per query
QUERY_TIMEOUT_S = 20

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
