"""Offline tests for pooling, budgeting, citation formatting, and the
augment_prompt wiring. rewrite_queries/hybrid_search.retrieve/the real
cross-encoder are monkeypatched so this file never calls an LLM, an
embedding API, or loads a torch model.
"""

import json

import pytest

from battery_aar.rag.scripts import augmented_prompt as ap


def _hit(chunk_id, text="text", **extra):
    return {"chunk_id": chunk_id, "text": text, **extra}


# --- _pool_with_rrf ------------------------------------------------------


def test_pool_with_rrf_dedupes_and_ranks_by_cross_query_score():
    per_query_hits = [
        [_hit("a"), _hit("b")],
        [_hit("b"), _hit("c")],
    ]
    pooled = ap._pool_with_rrf(per_query_hits, rrf_k=1)
    order = [r["chunk_id"] for r in pooled]
    assert order == ["b", "a", "c"]  # b appears in both lists, ranks highest
    by_id = {r["chunk_id"]: r for r in pooled}
    assert by_id["a"]["hit_queries"] == [0]
    assert by_id["b"]["hit_queries"] == [0, 1]
    assert by_id["c"]["hit_queries"] == [1]


def test_pool_with_rrf_empty_input():
    assert ap._pool_with_rrf([]) == []


# --- _apply_budget ---------------------------------------------------------


def test_apply_budget_caps_chunk_count():
    ranked = [_hit("a", "x" * 10), _hit("b", "x" * 10), _hit("c", "x" * 10)]
    selected = ap._apply_budget(ranked, k_chunks=2, max_context_chars=1000)
    assert [r["chunk_id"] for r in selected] == ["a", "b"]


def test_apply_budget_skips_oversized_chunks_but_keeps_checking_later_ones():
    ranked = [_hit("a", "x" * 100), _hit("b", "x" * 200), _hit("c", "x" * 50)]
    selected = ap._apply_budget(ranked, k_chunks=3, max_context_chars=200)
    # b alone would exceed the budget (100 + 200 > 200); c still fits after a.
    assert [r["chunk_id"] for r in selected] == ["a", "c"]


def test_apply_budget_empty_input():
    assert ap._apply_budget([], k_chunks=5, max_context_chars=1000) == []


# --- format_context_block ---------------------------------------------------


def test_format_context_block_numbers_and_cites_pages():
    chunks = [
        {"title": "T1", "doc_type": "textbook", "year": 2020, "page_start": 5, "page_end": 5, "text": "Hello"},
        {"title": "T2", "doc_type": "paper", "year": None, "page_start": 10, "page_end": 12, "text": "World"},
    ]
    block = ap.format_context_block(chunks)
    assert block.startswith(ap.CONTEXT_HEADER)
    assert "[1] T1 (textbook, 2020, p. 5)\nHello" in block
    assert "[2] T2 (paper, pp. 10-12)\nWorld" in block
    assert ", ," not in block  # year omission must not leave a stray separator


def test_format_context_block_empty():
    assert ap.format_context_block([]) == ap.CONTEXT_HEADER + "\n"


# --- _rerank_pool (cross-encoder stubbed) -----------------------------------


class _StubCrossEncoder:
    def __init__(self, scores):
        self._scores = scores

    def predict(self, pairs):
        return self._scores


def test_rerank_pool_takes_max_score_over_queries(monkeypatch):
    queries = ["q1", "q2"]
    pooled = [
        {"chunk_id": "a", "text": "ta", "pool_score": 1.0, "hit_queries": [0]},
        {"chunk_id": "b", "text": "tb", "pool_score": 0.5, "hit_queries": [1]},
    ]
    # pairs are built as [(q, text) for record in pooled for q in queries]:
    # (q1,ta) (q2,ta) (q1,tb) (q2,tb)
    stub = _StubCrossEncoder([0.2, 0.9, 0.7, 0.3])
    monkeypatch.setattr(ap.rerank, "_cross_encoder", lambda model_name: stub)

    reranked = ap._rerank_pool(queries, pooled, model_name="fake-model")

    assert [r["chunk_id"] for r in reranked] == ["a", "b"]
    assert reranked[0]["rerank_score"] == pytest.approx(0.9)
    assert reranked[0]["best_query"] == "q2"
    assert reranked[1]["rerank_score"] == pytest.approx(0.7)
    assert reranked[1]["best_query"] == "q1"


def test_rerank_pool_empty_pool_skips_model_call(monkeypatch):
    def _boom(model_name):
        raise AssertionError("should not load a model for an empty pool")

    monkeypatch.setattr(ap.rerank, "_cross_encoder", _boom)
    assert ap._rerank_pool(["q1"], [], model_name="fake-model") == []


# --- augment_prompt end-to-end wiring (everything external stubbed) --------


def _stub_chunk(chunk_id, text):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "title": "Test Book",
        "doc_type": "textbook",
        "year": 2020,
        "page_start": 1,
        "page_end": 1,
    }


def test_augment_prompt_assembles_context_and_writes_trace(monkeypatch, tmp_path):
    monkeypatch.setattr(ap, "rewrite_queries", lambda prompt, n_queries: ["q1", "q2"])

    per_query = {
        "q1": [_stub_chunk("a", "Alpha content")],
        "q2": [_stub_chunk("b", "Beta content")],
    }
    monkeypatch.setattr(
        ap.hybrid_search,
        "retrieve",
        lambda query, k, filter_spec, fusion, beta: per_query.get(query, []),
    )
    monkeypatch.setattr(
        ap.rerank, "_cross_encoder", lambda model_name: _StubCrossEncoder([0.9, 0.1, 0.2, 0.8])
    )

    result = ap.augment_prompt("ORIGINAL PROMPT TEXT", n_queries=2, trace_dir=tmp_path)

    assert result.original_prompt == "ORIGINAL PROMPT TEXT"
    assert result.prompt.endswith("\n\n---\n\nORIGINAL PROMPT TEXT")
    assert "ORIGINAL PROMPT TEXT" in result.prompt
    assert {c["chunk_id"] for c in result.chunks} == {"a", "b"}
    assert result.trace_path is not None and result.trace_path.exists()

    trace = json.loads(result.trace_path.read_text())
    assert trace["original_prompt"] == "ORIGINAL PROMPT TEXT"
    assert trace["queries"] == ["q1", "q2"]


def test_augment_prompt_falls_back_to_original_when_nothing_retrieved(monkeypatch, tmp_path):
    monkeypatch.setattr(ap, "rewrite_queries", lambda prompt, n_queries: ["q1"])
    monkeypatch.setattr(
        ap.hybrid_search, "retrieve", lambda query, k, filter_spec, fusion, beta: []
    )
    monkeypatch.setattr(ap.rerank, "_cross_encoder", lambda model_name: _StubCrossEncoder([]))

    result = ap.augment_prompt("ORIGINAL", n_queries=1, trace_dir=tmp_path)

    assert result.prompt == "ORIGINAL"
    assert result.chunks == []


def test_augment_prompt_skips_trace_when_trace_dir_is_none(monkeypatch):
    monkeypatch.setattr(ap, "rewrite_queries", lambda prompt, n_queries: ["q1"])
    monkeypatch.setattr(
        ap.hybrid_search, "retrieve", lambda query, k, filter_spec, fusion, beta: []
    )
    monkeypatch.setattr(ap.rerank, "_cross_encoder", lambda model_name: _StubCrossEncoder([]))

    result = ap.augment_prompt("ORIGINAL", n_queries=1, trace_dir=None)
    assert result.trace_path is None
