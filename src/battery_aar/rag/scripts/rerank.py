"""Cross-encoder re-ranking of hybrid search results (no LLM involved).

Takes the fused candidate list from ``hybrid_search.retrieve`` and re-scores
each (query, chunk text) pair with a local cross-encoder
(cross-encoder/ms-marco-MiniLM-L-6-v2 by default, runs on CPU via
sentence-transformers). Unlike the bi-encoder embedding stage, the
cross-encoder reads query and passage together, giving a sharper relevance
signal; it is applied only to the small candidate set because it is too slow
to score the whole corpus.

Each re-ranked hit keeps its retrieval diagnostics and gains
'rerank_score' and 'fusion_rank'; results are ordered by rerank_score.

Usage:
    python -m battery_aar.rag.scripts.rerank "your query" [--k 5] [--candidates 20]
        [--fusion rrf|beta] [--filter JSON] [--model NAME]

Example:
    python -m battery_aar.rag.scripts.rerank "how is state of charge defined" --k 3 --candidates 20
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache

from battery_aar.rag.scripts import hybrid_search

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CANDIDATES = 20


@lru_cache(maxsize=2)
def _cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rerank(query: str, hits: list[dict], model_name: str = DEFAULT_MODEL) -> list[dict]:
    """Re-order retrieval hits by cross-encoder relevance to ``query``.

    Returns new records sorted by 'rerank_score' (descending), each keeping
    the original fields plus 'fusion_rank' (1-based position before
    re-ranking).
    """
    if not hits:
        return []
    model = _cross_encoder(model_name)
    scores = model.predict([(query, hit["text"]) for hit in hits])
    ranked = sorted(
        (
            {**hit, "rerank_score": float(score), "fusion_rank": position}
            for position, (hit, score) in enumerate(zip(hits, scores), 1)
        ),
        key=lambda record: -record["rerank_score"],
    )
    return ranked


def retrieve_and_rerank(
    query: str,
    k: int = 5,
    candidates: int = DEFAULT_CANDIDATES,
    filter_spec: dict | None = None,
    fusion: str = hybrid_search.DEFAULT_FUSION,
    beta: float = hybrid_search.DEFAULT_BETA,
    model_name: str = DEFAULT_MODEL,
) -> list[dict]:
    """Hybrid retrieval of ``candidates`` chunks, cross-encoder re-ranked, top-k."""
    hits = hybrid_search.retrieve(
        query, k=candidates, filter_spec=filter_spec, fusion=fusion, beta=beta
    )
    return rerank(query, hits, model_name=model_name)[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--fusion", choices=["rrf", "beta"], default=hybrid_search.DEFAULT_FUSION)
    parser.add_argument("--beta", type=float, default=hybrid_search.DEFAULT_BETA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--filter", type=json.loads, default=None,
        help='metadata filter spec as JSON, e.g. \'{"tags_any": ["charging-protocols"]}\'',
    )
    args = parser.parse_args()

    hits = retrieve_and_rerank(
        args.query, args.k, args.candidates, args.filter, args.fusion, args.beta, args.model
    )
    for rank, hit in enumerate(hits, 1):
        pages = (
            f"p{hit['page_start']}"
            if hit["page_start"] == hit["page_end"]
            else f"pp{hit['page_start']}-{hit['page_end']}"
        )
        print(
            f"[{rank}] rerank {hit['rerank_score']:+.3f}  {hit['chunk_id']} ({pages}) "
            f"[was fusion #{hit['fusion_rank']}]"
        )
        print(f"    {hit['text'][:200]}...")


if __name__ == "__main__":
    main()
