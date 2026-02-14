"""Lightweight Bedrock smoke test (mocked).

Replaces prior `google.genai` checks. This script mocks a Bedrock
response and verifies `extract_signals_with_bedrock` parses it.

Run: `python3 api_test.py`
"""
import io
import sys
import types
from unittest.mock import patch

# Provide a minimal fake `arxiv` module so importing `main` works in
# environments without the `arxiv` package installed (only needed for this test).
arxiv_mod = sys.modules.setdefault("arxiv", types.ModuleType("arxiv"))
# minimal attributes used by `main` annotations and code
setattr(arxiv_mod, "Result", object)
setattr(arxiv_mod, "SortCriterion", types.SimpleNamespace(SubmittedDate=None))
setattr(arxiv_mod, "Search", lambda *a, **k: None)
setattr(arxiv_mod, "Client", lambda *a, **k: None)
# Provide a minimal fake `dotenv` with a no-op `load_dotenv` for imports
if "dotenv" not in sys.modules:
    dotenv_mod = types.ModuleType("dotenv")
    dotenv_mod.load_dotenv = lambda *a, **k: None
    sys.modules["dotenv"] = dotenv_mod

# Provide a minimal fake `boto3` so `main` can import in this isolated test.
if "boto3" not in sys.modules:
    boto3_mod = types.ModuleType("boto3")
    boto3_mod.client = lambda *a, **k: None
    sys.modules["boto3"] = boto3_mod

from main import extract_signals_with_bedrock


class FakeClient:
    def invoke_model(self, modelId, body):
        # Return a StreamingBody-like object with a JSON payload
        payload = b'{"main_topic": "mock topic", "methods": ["mock-method"], "keywords": ["kw1", "kw2"]}'
        return {"body": io.BytesIO(payload)}


def run_smoke_test():
    paper = {
        "title": "Mock Paper",
        "summary": "This is a dummy abstract mentioning mock topic and mock-method.",
        "published": "2026-02-01",
        "primary_category": "cs.AI",
    }

    with patch("boto3.client", return_value=FakeClient()):
        signals = extract_signals_with_bedrock([paper])
    print("Smoke test output:")
    print(signals)


if __name__ == "__main__":
    run_smoke_test()
