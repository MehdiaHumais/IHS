from pathlib import Path
import sys

print("Python:", sys.version)
print("Executable:", sys.executable)
print("Project:", Path(__file__).resolve().parent)

modules = [
    "fastapi", "uvicorn", "pydantic", "dotenv", "dateutil", "pandas",
    "requests", "rapidfuzz", "gspread", "google.oauth2.service_account",
    "fitz", "langchain_core", "langchain_openai", "langchain_groq",
    "langchain_ollama", "langchain_google_genai", "langchain_mcp_adapters",
    "langgraph", "fastmcp",
]
failed = []
for name in modules:
    try:
        __import__(name)
        print("[OK]", name)
    except Exception as exc:
        print("[FAIL]", name, "-", exc)
        failed.append(name)

raise SystemExit(1 if failed else 0)
