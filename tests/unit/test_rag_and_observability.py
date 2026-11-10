"""Config, ingestion (load + split, no embeddings), readiness checks and tracing (no LLM)."""

import httpx
import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from copilot.config import Settings, settings
from copilot.observability import health
from copilot.observability.tracing import setup_tracing
from copilot.rag.chain import format_docs
from copilot.rag.ingestion import load_documents, split_documents


def test_settings_defaults_and_env_override(monkeypatch):
    assert Settings().llm_model == "llama3.2:3b"
    monkeypatch.setenv("COPILOT_MAX_REWRITES", "5")
    assert Settings().max_rewrites == 5


def test_load_and_split_the_corpus():
    docs = load_documents()
    assert [d.metadata["source"] for d in docs] == ["centrifugal-pump-maintenance-manual.md"]
    chunks = split_documents(docs)
    assert len(chunks) == 11
    assert all(len(c.page_content) <= settings.chunk_size for c in chunks)
    assert all("start_index" in c.metadata for c in chunks)


def test_format_docs_adds_source_headers():
    docs = [
        Document(page_content="text A", metadata={"source": "a.md"}),
        Document(page_content="text B", metadata={"source": "b.md"}),
    ]
    assert format_docs(docs) == "[Source: a.md]\ntext A\n\n[Source: b.md]\ntext B"


class _FakeResponse:
    def __init__(self, models):
        self._models = models

    def raise_for_status(self):
        return None

    def json(self):
        return {"models": [{"name": m} for m in self._models]}


def test_check_ollama_ok(monkeypatch):
    monkeypatch.setattr(
        httpx, "get", lambda *a, **k: _FakeResponse(["llama3.2:3b", "nomic-embed-text:latest"])
    )
    ok, detail = health.check_ollama()
    assert ok and "llama3.2:3b" in detail


def test_check_ollama_missing_model(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse(["llama3.2:3b"]))
    ok, detail = health.check_ollama()
    assert not ok and "nomic-embed-text:latest" in detail


def test_check_ollama_unreachable(monkeypatch):
    def refuse(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", refuse)
    ok, detail = health.check_ollama()
    assert not ok and "unreachable" in detail


@pytest.mark.parametrize(("count", "ok"), [(11, True), (0, False)])
def test_check_vectorstore(monkeypatch, count, ok):
    class FakeStore:
        class _collection:  # noqa: N801 — mimics Chroma's attribute name
            @staticmethod
            def count():
                return count

    monkeypatch.setattr(health, "get_vectorstore", lambda: FakeStore)
    assert health.check_vectorstore()[0] is ok


def test_tracing_disabled_by_default():
    assert settings.tracing_enabled is False


def test_tracing_produces_nested_spans_without_a_server():
    exporter = InMemorySpanExporter()
    setup_tracing(span_exporter=exporter)
    chain = ChatPromptTemplate.from_messages([("human", "{q}")]) | FakeListChatModel(
        responses=["yes"]
    )
    chain.invoke({"q": "hello"})
    spans = exporter.get_finished_spans()
    kinds = {s.attributes.get("openinference.span.kind") for s in spans}
    assert "LLM" in kinds and "CHAIN" in kinds
    llm_span = next(s for s in spans if s.attributes.get("openinference.span.kind") == "LLM")
    assert llm_span.parent is not None, "the LLM call must be nested in the chain span"
