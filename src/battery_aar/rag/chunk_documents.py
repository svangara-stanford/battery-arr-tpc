"""Split processed page-level JSONL documents into fixed-size overlapping chunks.

Second stage of the vector RAG pipeline: reads the page records produced by
``process_pdfs.py``, concatenates each document's pages into continuous text,
and cuts hard fixed-size chunks (default 1000 chars, 200 overlap). Page
numbers are tracked as spans so every chunk keeps provenance for metadata
filtering and citations. The page-level files are left untouched; chunks are
written to a separate ``chunks.jsonl``.

Usage:
    python -m battery_aar.rag.chunk_documents [--processed-dir DIR]
        [--out FILE] [--chunk-size N] [--overlap N]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

RAG_DIR = Path(__file__).resolve().parent
DEFAULT_PROCESSED_DIR = RAG_DIR / "processed"
DEFAULT_OUT_FILE = DEFAULT_PROCESSED_DIR / "chunks.jsonl"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200

PAGE_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    source: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    n_chars: int
    text: str


@dataclass(frozen=True)
class PageSpan:
    page: int
    start: int
    end: int


def concatenate_pages(page_records: list[dict]) -> tuple[str, list[PageSpan]]:
    """Join a document's pages into one string, recording char offsets."""
    parts: list[str] = []
    spans: list[PageSpan] = []
    offset = 0
    for record in sorted(page_records, key=lambda r: r["page"]):
        text = record["text"]
        if not text:
            continue
        if parts:
            offset += len(PAGE_SEPARATOR)
        parts.append(text)
        spans.append(PageSpan(page=record["page"], start=offset, end=offset + len(text)))
        offset += len(text)
    return PAGE_SEPARATOR.join(parts), spans


def pages_for_range(spans: list[PageSpan], start: int, end: int) -> tuple[int, int]:
    """First and last page overlapping the half-open char range [start, end)."""
    touched = [s.page for s in spans if s.start < end and s.end > start]
    if not touched:
        raise ValueError(f"char range [{start}, {end}) maps to no page")
    return min(touched), max(touched)


def chunk_document(
    page_records: list[dict],
    chunk_size: int,
    overlap: int,
) -> list[ChunkRecord]:
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    text, spans = concatenate_pages(page_records)
    doc_id = page_records[0]["doc_id"]
    source = page_records[0]["source"]

    records: list[ChunkRecord] = []
    step = chunk_size - overlap
    start = 0
    index = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        page_start, page_end = pages_for_range(spans, start, end)
        chunk_text = text[start:end]
        records.append(
            ChunkRecord(
                chunk_id=f"{doc_id}:{index:04d}",
                doc_id=doc_id,
                source=source,
                page_start=page_start,
                page_end=page_end,
                char_start=start,
                char_end=end,
                n_chars=len(chunk_text),
                text=chunk_text,
            )
        )
        if end == len(text):
            break
        start += step
        index += 1
    return records


def load_page_records(jsonl_path: Path) -> list[dict]:
    with jsonl_path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def chunk_all(processed_dir: Path, out_file: Path, chunk_size: int, overlap: int) -> None:
    page_files = sorted(
        p for p in processed_dir.glob("*.jsonl") if p.resolve() != out_file.resolve()
    )
    if not page_files:
        raise FileNotFoundError(f"no page-level JSONL files found in {processed_dir}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out_file.open("w", encoding="utf-8") as handle:
        for page_file in page_files:
            page_records = load_page_records(page_file)
            chunks = chunk_document(page_records, chunk_size, overlap)
            for chunk in chunks:
                handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
            total += len(chunks)
            print(f"{page_file.name}: {len(page_records)} pages -> {len(chunks)} chunks")
    print(f"{total} chunks -> {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    args = parser.parse_args()
    chunk_all(args.processed_dir, args.out, args.chunk_size, args.overlap)


if __name__ == "__main__":
    main()
