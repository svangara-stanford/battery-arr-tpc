"""Offline tests for the BM25 index: bm25s runs fully locally, no network
or API calls involved.

bm25_index.search() validates filter specs via filters.validate_spec, which
defaults to the *real* repo documents.json with no override parameter. To
keep this suite fully self-contained (not dependent on a file that could be
absent, e.g. if it's ever fully untracked from git), the filter tests
monkeypatch bm25_index's bound `validate_spec` name to check against a
temp catalog instead.
"""

import json

import pytest

from battery_aar.rag.scripts import bm25_index
from battery_aar.rag.scripts.bm25_index import build, search
from battery_aar.rag.scripts.filters import validate_spec as real_validate_spec

CHUNKS = [
    {
        "chunk_id": "doc1:0000",
        "doc_id": "doc1",
        "doc_type": "textbook",
        "tags": [],
        "year": 2020,
        "text": "The voltmeter reads the terminal voltage of the cell during discharge.",
    },
    {
        "chunk_id": "doc1:0001",
        "doc_id": "doc1",
        "doc_type": "textbook",
        "tags": [],
        "year": 2020,
        "text": "Battery capacity fades over repeated charge and discharge cycles.",
    },
    {
        "chunk_id": "doc1:0002",
        "doc_id": "doc1",
        "doc_type": "textbook",
        "tags": [],
        "year": 2020,
        "text": "Electrode kinetics follow the Butler-Volmer equation under overpotential.",
    },
]


@pytest.fixture
def index_dir(tmp_path):
    chunks_file = tmp_path / "chunks.jsonl"
    chunks_file.write_text("\n".join(json.dumps(c) for c in CHUNKS) + "\n")
    out_dir = tmp_path / "bm25_index"
    build(chunks_file, out_dir, k1=1.5, b=0.75, method="lucene", stem=True)
    return out_dir


@pytest.fixture
def temp_catalog(tmp_path, monkeypatch):
    """Redirect bm25_index's filter validation to a self-contained temp
    catalog matching CHUNKS, instead of the real repo documents.json."""
    catalog = {
        "_tag_vocabulary": {"degradation-mechanisms": "desc"},
        "_doc_types": ["textbook", "paper", "review"],
        "documents": {"doc1": {}},
    }
    catalog_file = tmp_path / "documents.json"
    catalog_file.write_text(json.dumps(catalog))
    monkeypatch.setattr(
        bm25_index, "validate_spec", lambda spec: real_validate_spec(spec, catalog_file)
    )
    return catalog_file


def test_search_ranks_the_distinctive_match_first(index_dir):
    hits = search("voltmeter", index_dir, k=2)
    assert hits[0]["chunk_id"] == "doc1:0000"


def test_search_respects_k(index_dir):
    hits = search("battery", index_dir, k=1)
    assert len(hits) == 1


def test_search_with_matching_filter_reproduces_unfiltered_ranking(index_dir, temp_catalog):
    unfiltered = [h["chunk_id"] for h in search("battery", index_dir, k=3)]
    filtered = [h["chunk_id"] for h in search("battery", index_dir, k=3, filter_spec={"doc_type": ["textbook"]})]
    assert filtered == unfiltered


def test_search_with_excluding_filter_returns_empty(index_dir, temp_catalog):
    assert search("battery", index_dir, k=3, filter_spec={"doc_type": ["paper"]}) == []


def test_search_with_unknown_tag_raises(index_dir, temp_catalog):
    with pytest.raises(ValueError, match="tag vocabulary"):
        search("battery", index_dir, k=3, filter_spec={"tags_any": ["definitely-not-a-real-tag"]})
