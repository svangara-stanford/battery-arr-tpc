"""BM25 lexical index over document chunks (keyword half of hybrid retrieval).

Builds a bm25s index from the chunk records produced by
``chunk_documents.py`` and answers keyword queries against it. Scoring
parameters (k1, b, variant) and stemming are exposed so retrieval
evaluation can sweep them; the build-time tokenization settings are
persisted alongside the index and reused verbatim at query time.

Usage:
    python -m battery_aar.rag.scripts.bm25_index build [--k1 1.5] [--b 0.75]
        [--method lucene] [--no-stem]
    python -m battery_aar.rag.scripts.bm25_index search "your query" [--k 5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import bm25s
import numpy as np
import Stemmer

from battery_aar.rag.scripts.filters import allowed_mask, validate_spec

RAG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS_FILE = RAG_DIR / "processed" / "chunks.jsonl"
DEFAULT_INDEX_DIR = RAG_DIR / "processed" / "bm25_index"

PARAMS_FILE = "bm25_params.json"


def load_chunks(chunks_file: Path) -> list[dict]:
    with chunks_file.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _stemmer(stem: bool):
    return Stemmer.Stemmer("english") if stem else None


def build(
    chunks_file: Path,
    index_dir: Path,
    k1: float,
    b: float,
    method: str,
    stem: bool,
) -> None:
    chunks = load_chunks(chunks_file)
    tokens = bm25s.tokenize(
        [c["text"] for c in chunks], stemmer=_stemmer(stem), show_progress=False
    )
    retriever = bm25s.BM25(k1=k1, b=b, method=method)
    retriever.index(tokens, show_progress=False)
    retriever.save(str(index_dir), corpus=chunks)

    params = {"k1": k1, "b": b, "method": method, "stem": stem, "n_chunks": len(chunks)}
    (index_dir / PARAMS_FILE).write_text(json.dumps(params, indent=2) + "\n")
    print(f"indexed {len(chunks)} chunks -> {index_dir} (params: {params})")


def search(
    query: str,
    index_dir: Path = DEFAULT_INDEX_DIR,
    k: int = 5,
    filter_spec: dict | None = None,
) -> list[dict]:
    """Return the top-k chunk records for ``query``, each with a 'score' key.

    ``filter_spec`` (see ``filters.py``) is applied pre-ranking by masking
    the BM25 scores of ineligible chunks, so results are exact regardless of
    filter selectivity. Zero-score hits are dropped.
    """
    params = json.loads((index_dir / PARAMS_FILE).read_text())
    retriever = bm25s.BM25.load(str(index_dir), load_corpus=True)
    corpus = retriever.corpus

    weight_mask = None
    if filter_spec:
        validate_spec(filter_spec)
        mask = allowed_mask(corpus, filter_spec)
        if not mask.any():
            return []
        weight_mask = mask.astype(np.float32)
        k = min(k, int(mask.sum()))
    else:
        k = min(k, len(corpus))

    tokens = bm25s.tokenize(query, stemmer=_stemmer(params["stem"]), show_progress=False)
    docs, scores = retriever.retrieve(
        tokens, k=k, weight_mask=weight_mask, show_progress=False
    )
    return [
        {**doc, "score": float(score)}
        for doc, score in zip(docs[0], scores[0])
        if score > 0.0
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="build the BM25 index from chunks.jsonl")
    p_build.add_argument("--chunks-file", type=Path, default=DEFAULT_CHUNKS_FILE)
    p_build.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p_build.add_argument("--k1", type=float, default=1.5)
    p_build.add_argument("--b", type=float, default=0.75)
    p_build.add_argument(
        "--method",
        default="lucene",
        choices=["lucene", "robertson", "bm25+", "bm25l", "atire"],
        help="BM25 variant used for scoring",
    )
    p_build.add_argument(
        "--no-stem", dest="stem", action="store_false",
        help="disable English stemming in tokenization",
    )

    p_search = sub.add_parser("search", help="query an existing index")
    p_search.add_argument("query")
    p_search.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p_search.add_argument("--k", type=int, default=5)
    p_search.add_argument(
        "--filter", type=json.loads, default=None,
        help='metadata filter spec as JSON, e.g. \'{"tags_any": ["charging-protocols"]}\'',
    )

    args = parser.parse_args()
    if args.command == "build":
        build(args.chunks_file, args.index_dir, args.k1, args.b, args.method, args.stem)
    else:
        for rank, hit in enumerate(search(args.query, args.index_dir, args.k, args.filter), 1):
            pages = (
                f"p{hit['page_start']}"
                if hit["page_start"] == hit["page_end"]
                else f"pp{hit['page_start']}-{hit['page_end']}"
            )
            print(f"[{rank}] {hit['score']:.3f}  {hit['chunk_id']} ({pages})")
            print(f"    {hit['text'][:200]}...")


if __name__ == "__main__":
    main()
