"""Offline tests for fusion math (RRF, beta-weighted) and the retrieve() wiring.

bm25_index/embed_index are monkeypatched so this file never touches the real
indexes, the network, or the embedding API.
"""

import pytest

from battery_aar.rag.scripts import hybrid_search
from battery_aar.rag.scripts.hybrid_search import fuse_beta, fuse_rrf, retrieve


# --- _minmax (private, but its behavior underpins fuse_beta) ---------------


def test_minmax_empty():
    assert hybrid_search._minmax([]) == []


def test_minmax_scales_to_unit_range():
    assert hybrid_search._minmax([1, 2, 3]) == pytest.approx([0.0, 0.5, 1.0])


def test_minmax_constant_scores_all_map_to_one():
    assert hybrid_search._minmax([4, 4, 4]) == [1.0, 1.0, 1.0]


# --- fuse_beta ---------------------------------------------------------


def _hits(*id_score_pairs):
    return [{"chunk_id": cid, "score": score} for cid, score in id_score_pairs]


def test_fuse_beta_zero_is_pure_bm25():
    bm25 = _hits(("a", 10), ("b", 5))
    semantic = _hits(("b", 0.9), ("c", 0.5))
    fused = fuse_beta(bm25, semantic, beta=0.0)
    assert fused == pytest.approx({"a": 1.0, "b": 0.0, "c": 0.0})


def test_fuse_beta_one_is_pure_semantic():
    bm25 = _hits(("a", 10), ("b", 5))
    semantic = _hits(("b", 0.9), ("c", 0.5))
    fused = fuse_beta(bm25, semantic, beta=1.0)
    assert fused == pytest.approx({"a": 0.0, "b": 1.0, "c": 0.0})


def test_fuse_beta_midpoint_blends_both_sides():
    bm25 = _hits(("a", 10), ("b", 5))
    semantic = _hits(("b", 0.9), ("c", 0.5))
    fused = fuse_beta(bm25, semantic, beta=0.5)
    assert fused == pytest.approx({"a": 0.5, "b": 0.5, "c": 0.0})


# --- fuse_rrf ------------------------------------------------------------


def test_fuse_rrf_sums_reciprocal_ranks_across_lists():
    bm25 = _hits(("a", 0), ("b", 0))  # scores irrelevant to RRF, only order matters
    semantic = _hits(("b", 0), ("c", 0))
    fused = fuse_rrf(bm25, semantic, rrf_k=1)
    # a: rank1 in bm25 only -> 1/2
    # b: rank2 in bm25 + rank1 in semantic -> 1/3 + 1/2
    # c: rank2 in semantic only -> 1/3
    assert fused == pytest.approx({"a": 0.5, "b": 1 / 3 + 0.5, "c": 1 / 3})


# --- retrieve() wiring, with the two retrievers stubbed out ---------------


def test_retrieve_rejects_unknown_fusion_method():
    with pytest.raises(ValueError, match="unknown fusion method"):
        retrieve("q", fusion="not-a-method")


@pytest.mark.parametrize("beta", [-0.1, 1.1])
def test_retrieve_rejects_beta_out_of_range(beta):
    with pytest.raises(ValueError, match="beta must be"):
        retrieve("q", fusion="beta", beta=beta)


def test_retrieve_merges_and_annotates_diagnostics(monkeypatch):
    def fake_bm25_search(query, k=5, filter_spec=None):
        return _hits(("x", 5.0))

    def fake_semantic_search(query, k=5, filter_spec=None):
        return _hits(("y", 0.8))

    monkeypatch.setattr(hybrid_search.bm25_index, "search", fake_bm25_search)
    monkeypatch.setattr(hybrid_search.embed_index, "search", fake_semantic_search)

    results = retrieve("battery degradation", k=2, fusion="rrf")

    by_id = {r["chunk_id"]: r for r in results}
    assert set(by_id) == {"x", "y"}
    assert by_id["x"]["bm25_rank"] == 1 and by_id["x"]["semantic_rank"] is None
    assert by_id["y"]["semantic_rank"] == 1 and by_id["y"]["bm25_rank"] is None


def test_retrieve_respects_k_after_fusion(monkeypatch):
    def fake_bm25_search(query, k=5, filter_spec=None):
        return _hits(("a", 3.0), ("b", 2.0), ("c", 1.0))

    def fake_semantic_search(query, k=5, filter_spec=None):
        return []

    monkeypatch.setattr(hybrid_search.bm25_index, "search", fake_bm25_search)
    monkeypatch.setattr(hybrid_search.embed_index, "search", fake_semantic_search)

    results = retrieve("q", k=2, fusion="rrf")
    assert [r["chunk_id"] for r in results] == ["a", "b"]
