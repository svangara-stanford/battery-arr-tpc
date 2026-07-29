"""Retrieval evaluation harness: LLM-as-judge ground truth + parameter sweeps.

Ground truth comes from the real production query source, not synthetic
per-chunk questions: query_rewrite.rewrite_queries() run over the actual
FeatureScientist prompt produces the query set, so evaluation reflects
what the system is actually asked in production.

Those queries aren't tied to a source chunk, so relevance judgments are
built by pooling (TREC-style): for each query, candidates are pulled from
several diverse retrieval strategies (BM25-only, semantic-only, RRF hybrid,
beta hybrid, reranked), deduplicated, and graded for relevance by an LLM
judge in one call per query. Any config's retrieved chunks are then scored
against this judged pool -- chunks outside the pool count as non-relevant,
the standard (accepted) pooling limitation.

Sweeps cover every tunable knob:
    - BM25 (k1, b, method): rebuilds a temp BM25 index per value (~instant).
    - HNSW (M, ef_construction): rebuilds a temp HNSW index per value,
      reusing the cached embeddings.npy (no re-embedding, no API cost).
    - HNSW ef (query-time): no rebuild, varies the search-time parameter.
    - Fusion (method, beta, rrf_k): no rebuild, query-time only.
    - Rerank on/off: no rebuild, query-time only.
Chunk size/overlap is intentionally out of scope for this pass (it requires
re-embedding the whole corpus per variant); revisit as a follow-up.

Usage:
    python -m battery_aar.rag.scripts.eval_retrieval build-queries [--n 15]
    python -m battery_aar.rag.scripts.eval_retrieval build-judgments [--pool-k 8]
    python -m battery_aar.rag.scripts.eval_retrieval sweep --axis fusion|rerank|bm25|hnsw|all [--k 5]

Example:
    python -m battery_aar.rag.scripts.eval_retrieval build-queries --n 15
    python -m battery_aar.rag.scripts.eval_retrieval build-judgments --pool-k 8
    python -m battery_aar.rag.scripts.eval_retrieval sweep --axis all --k 5
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path
from statistics import mean

from battery_aar.rag.scripts import bm25_index, embed_index, hybrid_search, rerank
from battery_aar.rag.scripts.query_rewrite import _chat, feature_scientist_queries

RAG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS_FILE = RAG_DIR / "processed" / "chunks.jsonl"
EVAL_DIR = RAG_DIR / "processed" / "eval"
QUERIES_FILE = EVAL_DIR / "queries.json"
JUDGMENTS_FILE = EVAL_DIR / "judgments.json"

DEFAULT_N_QUERIES = 15
DEFAULT_POOL_K = 8
DEFAULT_EVAL_K = 5
RELEVANCE_THRESHOLD = 1  # grade >= this counts as "relevant" for recall/MRR

JUDGE_SYSTEM_PROMPT = (
    "You grade retrieval relevance for a battery-science RAG system. Given a "
    "query and a numbered list of passages, grade each passage's relevance to "
    "the query: 0 = not relevant / no useful information, 1 = partially or "
    "tangentially relevant, 2 = directly and substantially answers the query. "
    "Return only a JSON array of integers (0, 1, or 2), one per passage, in "
    "the same order, nothing else."
)

CHUNK_PREVIEW_CHARS = 600


# --- 1. Query set: the real production query source --------------------


def build_query_set(n_queries: int = DEFAULT_N_QUERIES) -> list[str]:
    """Retrieval queries from the real FeatureScientist prompt (not synthetic
    per-chunk questions), via the production query rewriter."""
    _, queries = feature_scientist_queries(n_queries=n_queries)
    return queries


def save_query_set(queries: list[str], path: Path = QUERIES_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(queries, indent=2) + "\n", encoding="utf-8")


def load_query_set(path: Path = QUERIES_FILE) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


# --- 2. Judgment pool: diverse retrieval strategies, deduplicated ----------


def _pool_candidates(query: str, pool_k: int = DEFAULT_POOL_K) -> list[dict]:
    pooled: dict[str, dict] = {}
    for hits in (
        bm25_index.search(query, k=pool_k),
        embed_index.search(query, k=pool_k),
        hybrid_search.retrieve(query, k=pool_k, fusion="rrf"),
        hybrid_search.retrieve(query, k=pool_k, fusion="beta"),
        rerank.retrieve_and_rerank(query, k=pool_k, candidates=max(pool_k * 2, 20)),
    ):
        for hit in hits:
            pooled.setdefault(hit["chunk_id"], hit)
    return list(pooled.values())


def _judge_pool(query: str, pooled: list[dict]) -> dict[str, int]:
    """One LLM call grades the whole pool for a query; returns chunk_id -> grade."""
    if not pooled:
        return {}
    listing = "\n\n".join(
        f"[{i}] {chunk['text'][:CHUNK_PREVIEW_CHARS]}" for i, chunk in enumerate(pooled, 1)
    )
    user_prompt = f"Query: {query}\n\nPassages:\n\n{listing}"
    reply = _chat(JUDGE_SYSTEM_PROMPT, user_prompt, temperature=0.0)
    grades = _parse_grade_array(reply, expected_len=len(pooled))
    return {chunk["chunk_id"]: grade for chunk, grade in zip(pooled, grades)}


def _parse_grade_array(text: str, expected_len: int) -> list[int]:
    import re

    match = re.search(r"\[.*\]", text, re.DOTALL)
    candidate = match.group(0) if match else text.strip()
    parsed = json.loads(candidate)
    if not isinstance(parsed, list) or len(parsed) != expected_len:
        raise ValueError(f"expected a JSON array of {expected_len} grades, got: {text}")
    return [max(0, min(2, int(g))) for g in parsed]


def build_judgments(
    queries: list[str], pool_k: int = DEFAULT_POOL_K
) -> dict[str, dict[str, int]]:
    judgments: dict[str, dict[str, int]] = {}
    for i, query in enumerate(queries, 1):
        pooled = _pool_candidates(query, pool_k)
        judgments[query] = _judge_pool(query, pooled)
        n_relevant = sum(g >= RELEVANCE_THRESHOLD for g in judgments[query].values())
        print(f"[{i}/{len(queries)}] {query!r}: {len(pooled)} pooled, {n_relevant} relevant")
    return judgments


def save_judgments(judgments: dict, path: Path = JUDGMENTS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(judgments, indent=2) + "\n", encoding="utf-8")


def load_judgments(path: Path = JUDGMENTS_FILE) -> dict[str, dict[str, int]]:
    return json.loads(path.read_text(encoding="utf-8"))


# --- 3. Metrics (pure functions over a ranked id list + grade dict) -------


def relevant_ids(grades: dict[str, int], threshold: int = RELEVANCE_THRESHOLD) -> set[str]:
    return {cid for cid, g in grades.items() if g >= threshold}


def recall_at_k(ranked_ids: list[str], grades: dict[str, int], k: int) -> float | None:
    relevant = relevant_ids(grades)
    if not relevant:
        return None
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked_ids: list[str], grades: dict[str, int]) -> float | None:
    relevant = relevant_ids(grades)
    if not relevant:
        return None
    for rank, cid in enumerate(ranked_ids, 1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(ranked_ids: list[str], grades: dict[str, int], k: int) -> float:
    return sum(
        (2 ** grades.get(cid, 0) - 1) / math.log2(rank + 1)
        for rank, cid in enumerate(ranked_ids[:k], 1)
    )


def ndcg_at_k(ranked_ids: list[str], grades: dict[str, int], k: int) -> float | None:
    if not relevant_ids(grades):
        return None
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(ideal, 1))
    if idcg == 0:
        return None
    return dcg_at_k(ranked_ids, grades, k) / idcg


def evaluate_config(
    queries: list[str],
    judgments: dict[str, dict[str, int]],
    retrieve_fn,
    k: int = DEFAULT_EVAL_K,
) -> dict:
    """Mean recall@k, MRR, nDCG@k for a retrieve_fn(query) -> list[hit dict]
    config, averaged over queries that have at least one judged-relevant chunk."""
    recalls, rrs, ndcgs = [], [], []
    for query in queries:
        grades = judgments.get(query, {})
        if not relevant_ids(grades):
            continue
        ranked_ids = [hit["chunk_id"] for hit in retrieve_fn(query)]
        recalls.append(recall_at_k(ranked_ids, grades, k))
        rrs.append(reciprocal_rank(ranked_ids, grades))
        ndcgs.append(ndcg_at_k(ranked_ids, grades, k))
    return {
        "n_queries": len(queries),
        "n_scored": len(recalls),
        f"recall@{k}": mean(recalls) if recalls else None,
        "mrr": mean(rrs) if rrs else None,
        f"ndcg@{k}": mean(ndcgs) if ndcgs else None,
    }


# --- 4. Sweeps -------------------------------------------------------------


def sweep_fusion(queries, judgments, k: int = DEFAULT_EVAL_K) -> list[dict]:
    rows = []
    for beta in (0.0, 0.3, 0.5, 0.7, 0.9, 1.0):
        row = evaluate_config(
            queries, judgments,
            lambda q, b=beta: hybrid_search.retrieve(q, k=k, fusion="beta", beta=b), k,
        )
        rows.append({"fusion": "beta", "beta": beta, **row})
    for rrf_k in (10, 30, 60, 100):
        row = evaluate_config(
            queries, judgments,
            lambda q, rk=rrf_k: hybrid_search.retrieve(q, k=k, fusion="rrf", rrf_k=rk), k,
        )
        rows.append({"fusion": "rrf", "rrf_k": rrf_k, **row})
    return rows


def sweep_rerank(queries, judgments, k: int = DEFAULT_EVAL_K) -> list[dict]:
    rows = []
    for fusion in ("rrf", "beta"):
        for use_rerank in (False, True):
            if use_rerank:
                retrieve_fn = lambda q, f=fusion: rerank.retrieve_and_rerank(
                    q, k=k, candidates=20, fusion=f
                )
            else:
                retrieve_fn = lambda q, f=fusion: hybrid_search.retrieve(q, k=k, fusion=f)
            row = evaluate_config(queries, judgments, retrieve_fn, k)
            rows.append({"fusion": fusion, "rerank": use_rerank, **row})
    return rows


def sweep_bm25(
    queries, judgments, chunks_file: Path = DEFAULT_CHUNKS_FILE, k: int = DEFAULT_EVAL_K
) -> list[dict]:
    baseline = {"k1": 1.5, "b": 0.75, "method": "lucene"}
    grid = (
        [("k1", v) for v in (1.0, 1.2, 1.5, 2.0, 2.5)]
        + [("b", v) for v in (0.0, 0.25, 0.5, 0.75, 1.0)]
        + [("method", v) for v in ("lucene", "robertson", "bm25+", "bm25l", "atire")]
    )
    rows = []
    for param, value in grid:
        config = {**baseline, param: value}
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp)
            bm25_index.build(chunks_file, index_dir, config["k1"], config["b"], config["method"], stem=True)
            row = evaluate_config(
                queries, judgments,
                lambda q, d=index_dir: bm25_index.search(q, index_dir=d, k=k), k,
            )
        rows.append({**config, **row})
    return rows


def sweep_hnsw(
    queries, judgments, chunks_file: Path = DEFAULT_CHUNKS_FILE, k: int = DEFAULT_EVAL_K
) -> list[dict]:
    default_index_dir = embed_index.DEFAULT_INDEX_DIR
    default_embeddings = default_index_dir / embed_index.EMBEDDINGS_FILE
    baseline_m, baseline_ef_construction = embed_index.DEFAULT_M, embed_index.DEFAULT_EF_CONSTRUCTION

    rows = []
    for m in (8, 16, 32, 64):
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp)
            shutil.copy(default_embeddings, index_dir / embed_index.EMBEDDINGS_FILE)
            embed_index.build(chunks_file, index_dir, m, baseline_ef_construction, re_embed=False)
            row = evaluate_config(
                queries, judgments,
                lambda q, d=index_dir: embed_index.search(q, index_dir=d, k=k), k,
            )
        rows.append({"param": "M", "M": m, "ef_construction": baseline_ef_construction, **row})

    for ef_construction in (50, 100, 200, 400):
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp)
            shutil.copy(default_embeddings, index_dir / embed_index.EMBEDDINGS_FILE)
            embed_index.build(chunks_file, index_dir, baseline_m, ef_construction, re_embed=False)
            row = evaluate_config(
                queries, judgments,
                lambda q, d=index_dir: embed_index.search(q, index_dir=d, k=k), k,
            )
        rows.append({"param": "ef_construction", "M": baseline_m, "ef_construction": ef_construction, **row})

    for ef_search in (10, 25, 50, 100, 200):
        row = evaluate_config(
            queries, judgments,
            lambda q, ef=ef_search: embed_index.search(q, k=k, ef=ef), k,
        )
        rows.append({"param": "ef (query-time)", "ef": ef_search, **row})
    return rows


SWEEPS = {"fusion": sweep_fusion, "rerank": sweep_rerank, "bm25": sweep_bm25, "hnsw": sweep_hnsw}


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("(no rows)")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_queries = sub.add_parser("build-queries", help="generate the eval query set")
    p_queries.add_argument("--n", type=int, default=DEFAULT_N_QUERIES)

    p_judgments = sub.add_parser("build-judgments", help="pool candidates + LLM-judge relevance")
    p_judgments.add_argument("--pool-k", type=int, default=DEFAULT_POOL_K)

    p_sweep = sub.add_parser("sweep", help="run a parameter sweep against saved judgments")
    p_sweep.add_argument("--axis", choices=[*SWEEPS, "all"], default="all")
    p_sweep.add_argument("--k", type=int, default=DEFAULT_EVAL_K)

    args = parser.parse_args()

    if args.command == "build-queries":
        queries = build_query_set(args.n)
        save_query_set(queries)
        print(f"{len(queries)} queries -> {QUERIES_FILE}")
        for q in queries:
            print(f"  - {q}")
    elif args.command == "build-judgments":
        queries = load_query_set()
        judgments = build_judgments(queries, args.pool_k)
        save_judgments(judgments)
        print(f"judgments -> {JUDGMENTS_FILE}")
    else:
        queries = load_query_set()
        judgments = load_judgments()
        axes = list(SWEEPS) if args.axis == "all" else [args.axis]
        for axis in axes:
            print(f"\n=== {axis} ===")
            print_table(SWEEPS[axis](queries, judgments, k=args.k))


if __name__ == "__main__":
    main()
