"""Offline tests for chunking math, page-span tracking, and catalog validation."""

import json
from pathlib import Path

import pytest

from battery_aar.rag.scripts.chunk_documents import (
    chunk_all,
    chunk_document,
    concatenate_pages,
    load_catalog,
    pages_for_range,
)

CATALOG_ENTRY = {
    "title": "Test Book",
    "doc_type": "textbook",
    "year": 2020,
    "tags": ["degradation-mechanisms"],
}


def _write_catalog(tmp_path, documents: dict) -> Path:
    catalog = {
        "_tag_vocabulary": {"degradation-mechanisms": "desc", "charging-protocols": "desc"},
        "_doc_types": ["textbook", "paper", "review"],
        "documents": documents,
    }
    path = tmp_path / "documents.json"
    path.write_text(json.dumps(catalog))
    return path


def test_concatenate_pages_tracks_offsets_and_sorts_by_page():
    records = [
        {"page": 2, "text": "BBBB"},
        {"page": 1, "text": "AAAA"},
    ]
    text, spans = concatenate_pages(records)
    assert text == "AAAA\n\nBBBB"
    assert spans[0].page == 1 and (spans[0].start, spans[0].end) == (0, 4)
    assert spans[1].page == 2 and (spans[1].start, spans[1].end) == (6, 10)


def test_concatenate_pages_skips_empty_text():
    records = [{"page": 1, "text": ""}, {"page": 2, "text": "content"}]
    text, spans = concatenate_pages(records)
    assert text == "content"
    assert len(spans) == 1 and spans[0].page == 2


def test_pages_for_range_single_page():
    _, spans = concatenate_pages([{"page": 1, "text": "AAAA"}, {"page": 2, "text": "BBBB"}])
    assert pages_for_range(spans, 0, 4) == (1, 1)


def test_pages_for_range_spans_the_page_boundary():
    _, spans = concatenate_pages([{"page": 1, "text": "AAAA"}, {"page": 2, "text": "BBBB"}])
    assert pages_for_range(spans, 3, 7) == (1, 2)


def test_pages_for_range_raises_for_a_gap_only_range():
    _, spans = concatenate_pages([{"page": 1, "text": "AAAA"}, {"page": 2, "text": "BBBB"}])
    with pytest.raises(ValueError):
        pages_for_range(spans, 4, 6)  # falls entirely inside the "\n\n" separator


def test_chunk_document_tiles_without_overlap():
    text = "0123456789ABCDEFGHIJ"  # 20 chars
    records = [{"doc_id": "doc1", "source": "doc1.pdf", "page": 1, "text": text}]
    chunks = chunk_document(records, chunk_size=10, overlap=0, catalog_entry=CATALOG_ENTRY)
    assert [c.text for c in chunks] == ["0123456789", "ABCDEFGHIJ"]
    for a, b in zip(chunks, chunks[1:]):
        assert a.char_end == b.char_start


def test_chunk_document_overlap_invariant_and_reconstruction():
    text = "".join(f"{i:03d}" for i in range(30))  # 90 chars, deterministic content
    records = [{"doc_id": "doc1", "source": "doc1.pdf", "page": 1, "text": text}]
    chunks = chunk_document(records, chunk_size=20, overlap=5, catalog_entry=CATALOG_ENTRY)

    for chunk in chunks:
        assert chunk.text == text[chunk.char_start : chunk.char_end]
        assert chunk.n_chars == len(chunk.text)
    for a, b in zip(chunks, chunks[1:]):
        assert a.text[-5:] == b.text[:5]
    assert chunks[-1].char_end == len(text)
    assert [c.chunk_id for c in chunks] == [f"doc1:{i:04d}" for i in range(len(chunks))]


def test_chunk_document_stamps_catalog_metadata_on_every_chunk():
    records = [{"doc_id": "doc1", "source": "doc1.pdf", "page": 1, "text": "x" * 25}]
    chunks = chunk_document(records, chunk_size=10, overlap=2, catalog_entry=CATALOG_ENTRY)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.title == CATALOG_ENTRY["title"]
        assert chunk.doc_type == CATALOG_ENTRY["doc_type"]
        assert chunk.year == CATALOG_ENTRY["year"]
        assert chunk.tags == CATALOG_ENTRY["tags"]


def test_chunk_document_tracks_page_spans_across_a_boundary():
    # page1 text occupies chars [0, 8), the "\n\n" separator [8, 10), page2 [10, 18).
    # A chunk_size of 15 guarantees the first chunk reaches past the separator
    # into page2's real text (chunk_size=10 would land exactly on the
    # separator boundary and not actually cross into page2).
    records = [
        {"doc_id": "doc1", "source": "doc1.pdf", "page": 1, "text": "A" * 8},
        {"doc_id": "doc1", "source": "doc1.pdf", "page": 2, "text": "B" * 8},
    ]
    chunks = chunk_document(records, chunk_size=15, overlap=0, catalog_entry=CATALOG_ENTRY)
    spanning = [c for c in chunks if c.page_start != c.page_end]
    assert spanning, "expected at least one chunk to cross the page boundary"
    assert spanning[0].page_start == 1 and spanning[0].page_end == 2


@pytest.mark.parametrize("chunk_size,overlap", [(10, 10), (10, 11), (10, -1)])
def test_chunk_document_rejects_invalid_overlap(chunk_size, overlap):
    records = [{"doc_id": "doc1", "source": "doc1.pdf", "page": 1, "text": "x" * 20}]
    with pytest.raises(ValueError):
        chunk_document(records, chunk_size=chunk_size, overlap=overlap, catalog_entry=CATALOG_ENTRY)


def test_load_catalog_valid(tmp_path):
    path = _write_catalog(tmp_path, {"doc1": CATALOG_ENTRY})
    catalog = load_catalog(path)
    assert catalog == {"doc1": CATALOG_ENTRY}


def test_load_catalog_missing_field(tmp_path):
    entry = {k: v for k, v in CATALOG_ENTRY.items() if k != "year"}
    path = _write_catalog(tmp_path, {"doc1": entry})
    with pytest.raises(ValueError, match="missing fields"):
        load_catalog(path)


def test_load_catalog_bad_doc_type(tmp_path):
    entry = {**CATALOG_ENTRY, "doc_type": "blog-post"}
    path = _write_catalog(tmp_path, {"doc1": entry})
    with pytest.raises(ValueError, match="doc_type"):
        load_catalog(path)


def test_load_catalog_unknown_tag(tmp_path):
    entry = {**CATALOG_ENTRY, "tags": ["not-a-real-tag"]}
    path = _write_catalog(tmp_path, {"doc1": entry})
    with pytest.raises(ValueError, match="outside _tag_vocabulary"):
        load_catalog(path)


def test_load_catalog_empty_tags(tmp_path):
    entry = {**CATALOG_ENTRY, "tags": []}
    path = _write_catalog(tmp_path, {"doc1": entry})
    with pytest.raises(ValueError, match="at least one tag"):
        load_catalog(path)


def test_chunk_all_raises_for_uncatalogued_document(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    page_record = {
        "doc_id": "mystery-doc",
        "source": "mystery-doc.pdf",
        "page": 1,
        "n_pages": 1,
        "n_chars": 5,
        "text": "hello",
    }
    (processed / "mystery-doc.jsonl").write_text(json.dumps(page_record) + "\n")
    catalog_file = _write_catalog(tmp_path, {})  # no entries at all

    with pytest.raises(ValueError, match="missing from"):
        chunk_all(processed, processed / "chunks.jsonl", 1000, 200, catalog_file)


def test_chunk_all_writes_chunks_for_catalogued_documents(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    page_record = {
        "doc_id": "doc1",
        "source": "doc1.pdf",
        "page": 1,
        "n_pages": 1,
        "n_chars": 25,
        "text": "x" * 25,
    }
    (processed / "doc1.jsonl").write_text(json.dumps(page_record) + "\n")
    catalog_file = _write_catalog(tmp_path, {"doc1": CATALOG_ENTRY})
    out_file = processed / "chunks.jsonl"

    chunk_all(processed, out_file, chunk_size=10, overlap=2, catalog_file=catalog_file)

    lines = out_file.read_text().splitlines()
    assert len(lines) > 1
    assert all(json.loads(line)["doc_id"] == "doc1" for line in lines)
