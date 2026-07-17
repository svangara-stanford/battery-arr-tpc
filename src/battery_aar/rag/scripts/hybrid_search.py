"""Hybrid retrieval: merge BM25 keyword and semantic (HNSW) search results.

Runs both retrievers with the same query and metadata filter spec, then
fuses the two ranked lists into one final top-k. Two fusion methods are
implemented so retrieval evaluation can compare them:

* ``beta``  -- min-max normalize each retriever's scores to [0, 1] within
  its candidate list, then score = beta * semantic + (1 - beta) * keyword.
  beta=1 is pure semantic, beta=0 pure keyword.
* ``rrf``   -- Reciprocal Rank Fusion: score = sum of 1 / (rrf_k + rank)
  over the lists containing the chunk. Ignores score magnitudes, no
  normalization involved.

Each fused hit keeps per-retriever diagnostics (score and rank under each
method, or None where a retriever did not return the chunk) for traces and
evaluation.

Usage:
    python -m battery_aar.rag.scripts.hybrid_search "your query" [--k 5]
        [--fusion rrf|beta] [--beta 0.5] [--candidates 50] [--filter JSON]
"""

from __future__ import annotations

import argparse
import json

from battery_aar.rag.scripts import bm25_index, embed_index

DEFAULT_FUSION = "rrf"
DEFAULT_BETA = 0.7 # 0.7 is our starting value.
DEFAULT_RRF_K = 60
# How many hits to pull from each retriever before fusing. Fusion can only
# rank chunks that at least one retriever surfaced, so this is deliberately
# larger than the final k.
DEFAULT_CANDIDATES = 50


def _minmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high == low:
        return [1.0] * len(scores)
    return [(s - low) / (high - low) for s in scores]


def fuse_beta(
    bm25_hits: list[dict], semantic_hits: list[dict], beta: float
) -> dict[str, float]:
    """chunk_id -> beta-weighted sum of min-max normalized scores."""
    bm25_norm = dict(
        zip([h["chunk_id"] for h in bm25_hits], _minmax([h["score"] for h in bm25_hits]))
    )
    semantic_norm = dict(
        zip(
            [h["chunk_id"] for h in semantic_hits],
            _minmax([h["score"] for h in semantic_hits]),
        )
    )
    return {
        chunk_id: beta * semantic_norm.get(chunk_id, 0.0)
        + (1.0 - beta) * bm25_norm.get(chunk_id, 0.0)
        for chunk_id in set(bm25_norm) | set(semantic_norm)
    }


def fuse_rrf(
    bm25_hits: list[dict], semantic_hits: list[dict], rrf_k: int
) -> dict[str, float]:
    """chunk_id -> sum of reciprocal ranks across the two lists."""
    fused: dict[str, float] = {}
    for hits in (bm25_hits, semantic_hits):
        for rank, hit in enumerate(hits, 1):
            fused[hit["chunk_id"]] = fused.get(hit["chunk_id"], 0.0) + 1.0 / (rrf_k + rank)
    return fused


def retrieve(
    query: str,
    k: int = 5,
    filter_spec: dict | None = None,
    fusion: str = DEFAULT_FUSION,
    beta: float = DEFAULT_BETA,
    rrf_k: int = DEFAULT_RRF_K,
    candidates: int = DEFAULT_CANDIDATES,
) -> list[dict]:
    """Hybrid top-k for ``query``: both retrievers, one fused ranking.

    Returns chunk records with 'score' (fused), plus 'bm25_score'/'bm25_rank'
    and 'semantic_score'/'semantic_rank' diagnostics (None where a retriever
    did not surface the chunk).
    """
    if fusion not in ("beta", "rrf"):
        raise ValueError(f"unknown fusion method {fusion!r}; use 'beta' or 'rrf'")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")

    bm25_hits = bm25_index.search(query, k=candidates, filter_spec=filter_spec)
    semantic_hits = embed_index.search(query, k=candidates, filter_spec=filter_spec)

    if fusion == "beta":
        fused = fuse_beta(bm25_hits, semantic_hits, beta)
    else:
        fused = fuse_rrf(bm25_hits, semantic_hits, rrf_k)

    by_id = {h["chunk_id"]: h for h in bm25_hits}
    by_id.update({h["chunk_id"]: h for h in semantic_hits})
    bm25_pos = {h["chunk_id"]: (i + 1, h["score"]) for i, h in enumerate(bm25_hits)}
    semantic_pos = {h["chunk_id"]: (i + 1, h["score"]) for i, h in enumerate(semantic_hits)}

    top = sorted(fused, key=lambda chunk_id: -fused[chunk_id])[:k]
    results = []
    for chunk_id in top:
        record = dict(by_id[chunk_id])
        bm25_rank, bm25_score = bm25_pos.get(chunk_id, (None, None))
        semantic_rank, semantic_score = semantic_pos.get(chunk_id, (None, None))
        record.update(
            score=fused[chunk_id],
            bm25_rank=bm25_rank,
            bm25_score=bm25_score,
            semantic_rank=semantic_rank,
            semantic_score=semantic_score,
        )
        results.append(record)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--fusion", choices=["rrf", "beta"], default=DEFAULT_FUSION)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument(
        "--filter", type=json.loads, default=None,
        help='metadata filter spec as JSON, e.g. \'{"tags_any": ["charging-protocols"]}\'',
    )
    args = parser.parse_args()

    hits = retrieve(
        args.query, args.k, args.filter, args.fusion, args.beta, args.rrf_k, args.candidates
    )
    for rank, hit in enumerate(hits, 1):
        pages = (
            f"p{hit['page_start']}"
            if hit["page_start"] == hit["page_end"]
            else f"pp{hit['page_start']}-{hit['page_end']}"
        )
        provenance = (
            f"bm25 #{hit['bm25_rank']}" if hit["bm25_rank"] else "bm25 -",
            f"sem #{hit['semantic_rank']}" if hit["semantic_rank"] else "sem -",
        )
        print(f"[{rank}] {hit['score']:.4f}  {hit['chunk_id']} ({pages}) [{', '.join(provenance)}]")
        print(f"    {hit['text'][:200]}...")


if __name__ == "__main__":
    main()
