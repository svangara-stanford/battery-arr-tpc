"""Semantic index over document chunks: ada-002 embeddings + HNSW (hnswlib).

Embeds the chunk records produced by ``chunk_documents.py`` with OpenAI's
text-embedding-ada-002 (fixed project decision, independent of the agent
model configured in .env) and builds an hnswlib HNSW index for cosine
similarity search. HNSW parameters (M, ef_construction, ef) are exposed for
the retrieval parameter study.

Embeddings are cached to ``embeddings.npy`` next to the index, so rebuilding
with different HNSW parameters does not re-pay the embedding API cost; pass
--re-embed to force a fresh embedding run.

Credentials reuse the agent client env chain (OPEN_BATTERY_AGENTS_* /
STANFORD_AI_*, optionally from .env).

Usage:
    python -m battery_aar.rag.scripts.embed_index build [--M 16]
        [--ef-construction 200] [--re-embed]
    python -m battery_aar.rag.scripts.embed_index search "your query" [--k 5] [--ef 50]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hnswlib
import numpy as np

from battery_aar.agents.llm_client import load_llm_client_config
from battery_aar.rag.scripts.bm25_index import load_chunks
from battery_aar.rag.scripts.filters import allowed_mask, validate_spec

EMBEDDING_MODEL = "text-embedding-ada-002"  # fixed; not the .env agent model
EMBEDDING_DIM = 1536 # default embedding size for ada-002

RAG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS_FILE = RAG_DIR / "processed" / "chunks.jsonl"
DEFAULT_INDEX_DIR = RAG_DIR / "processed" / "hnsw_index"

INDEX_FILE = "index.bin"
EMBEDDINGS_FILE = "embeddings.npy"
PARAMS_FILE = "hnsw_params.json"

DEFAULT_M = 16
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_EF_SEARCH = 50
EMBED_BATCH_SIZE = 64

# Below this fraction of allowed chunks, filtered search switches from the
# HNSW filter callback (graph traversal degrades under selective filters) to
# exact brute-force cosine over the cached embeddings of the allowed subset.
BRUTE_FORCE_FRACTION = 0.1


def _embedding_client():
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    from openai import OpenAI

    config = load_llm_client_config()
    if not config.api_key:
        raise RuntimeError(
            "no API key configured; set OPEN_BATTERY_AGENTS_API_KEY or "
            "STANFORD_AI_API_KEY (see .env.example)"
        )
    kwargs = {"api_key": config.api_key, "base_url": config.base_url}
    if config.default_headers:
        kwargs["default_headers"] = config.default_headers
    return OpenAI(**kwargs)


def embed_texts(texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    client = _embedding_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
        print(f"embedded {min(start + batch_size, len(texts))}/{len(texts)}")
    array = np.asarray(vectors, dtype=np.float32)
    if array.shape != (len(texts), EMBEDDING_DIM):
        raise RuntimeError(f"unexpected embedding shape {array.shape}")
    return array


def build(
    chunks_file: Path,
    index_dir: Path,
    m: int,
    ef_construction: int,
    re_embed: bool,
) -> None:
    chunks = load_chunks(chunks_file)
    index_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = index_dir / EMBEDDINGS_FILE
    if embeddings_path.exists() and not re_embed:
        embeddings = np.load(embeddings_path)
        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"cached embeddings ({len(embeddings)}) do not match chunks "
                f"({len(chunks)}); rerun with --re-embed"
            )
        print(f"reusing cached embeddings from {embeddings_path}")
    else:
        embeddings = embed_texts([c["text"] for c in chunks])
        np.save(embeddings_path, embeddings)

    index = hnswlib.Index(space="cosine", dim=EMBEDDING_DIM)
    index.init_index(max_elements=len(chunks), M=m, ef_construction=ef_construction)
    index.add_items(embeddings, np.arange(len(chunks)))
    index.save_index(str(index_dir / INDEX_FILE))

    params = {
        "model": EMBEDDING_MODEL,
        "dim": EMBEDDING_DIM,
        "M": m,
        "ef_construction": ef_construction,
        "n_chunks": len(chunks),
        "chunks_file": str(chunks_file),
    }
    (index_dir / PARAMS_FILE).write_text(json.dumps(params, indent=2) + "\n")
    print(f"indexed {len(chunks)} chunks -> {index_dir} (params: {params})")


def _brute_force_search(
    query_vector: np.ndarray,
    allowed_labels: np.ndarray,
    embeddings: np.ndarray,
    k: int,
) -> list[tuple[int, float]]:
    """Exact cosine top-k over an allowed subset; returns (label, similarity)."""
    subset = embeddings[allowed_labels]
    subset = subset / np.linalg.norm(subset, axis=1, keepdims=True)
    query = query_vector[0] / np.linalg.norm(query_vector[0])
    similarities = subset @ query
    order = np.argsort(-similarities)[:k]
    return [(int(allowed_labels[i]), float(similarities[i])) for i in order]


def search(
    query: str,
    index_dir: Path = DEFAULT_INDEX_DIR,
    k: int = 5,
    ef: int = DEFAULT_EF_SEARCH,
    filter_spec: dict | None = None,
) -> list[dict]:
    """Return the top-k chunk records for ``query``, each with a 'score' key
    (cosine similarity, higher is better).

    ``filter_spec`` (see ``filters.py``) is applied pre-ranking: broad
    filters use the HNSW filter callback; filters keeping fewer than
    BRUTE_FORCE_FRACTION of chunks fall back to exact cosine search over the
    cached embeddings of the allowed subset.
    """
    params = json.loads((index_dir / PARAMS_FILE).read_text())
    chunks = load_chunks(Path(params["chunks_file"]))
    query_vector = embed_texts([query])

    mask = None
    if filter_spec:
        validate_spec(filter_spec)
        mask = allowed_mask(chunks, filter_spec)
        if not mask.any():
            return []
        k = min(k, int(mask.sum()))

    if mask is not None and mask.sum() < BRUTE_FORCE_FRACTION * len(chunks):
        embeddings = np.load(index_dir / EMBEDDINGS_FILE)
        hits = _brute_force_search(query_vector, np.flatnonzero(mask), embeddings, k)
        return [{**chunks[label], "score": score} for label, score in hits]

    index = hnswlib.Index(space="cosine", dim=params["dim"])
    index.load_index(str(index_dir / INDEX_FILE), max_elements=params["n_chunks"])
    index.set_ef(max(ef, k))
    filter_function = (lambda label: bool(mask[label])) if mask is not None else None
    labels, distances = index.knn_query(query_vector, k=k, filter=filter_function)
    return [
        {**chunks[label], "score": float(1.0 - distance)}
        for label, distance in zip(labels[0], distances[0])
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="embed chunks and build the HNSW index")
    p_build.add_argument("--chunks-file", type=Path, default=DEFAULT_CHUNKS_FILE)
    p_build.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p_build.add_argument("--M", type=int, default=DEFAULT_M, dest="m")
    p_build.add_argument("--ef-construction", type=int, default=DEFAULT_EF_CONSTRUCTION)
    p_build.add_argument(
        "--re-embed", action="store_true",
        help="ignore cached embeddings.npy and call the embedding API again",
    )

    p_search = sub.add_parser("search", help="query an existing index")
    p_search.add_argument("query")
    p_search.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p_search.add_argument("--k", type=int, default=5)
    p_search.add_argument("--ef", type=int, default=DEFAULT_EF_SEARCH)
    p_search.add_argument(
        "--filter", type=json.loads, default=None,
        help='metadata filter spec as JSON, e.g. \'{"tags_any": ["charging-protocols"]}\'',
    )

    args = parser.parse_args()
    if args.command == "build":
        build(args.chunks_file, args.index_dir, args.m, args.ef_construction, args.re_embed)
    else:
        hits = search(args.query, args.index_dir, args.k, args.ef, args.filter)
        for rank, hit in enumerate(hits, 1):
            pages = (
                f"p{hit['page_start']}"
                if hit["page_start"] == hit["page_end"]
                else f"pp{hit['page_start']}-{hit['page_end']}"
            )
            print(f"[{rank}] {hit['score']:.3f}  {hit['chunk_id']} ({pages})")
            print(f"    {hit['text'][:200]}...")


if __name__ == "__main__":
    main()
