"""Declarative metadata filter spec shared by keyword and semantic retrieval.

A filter spec is a plain JSON-able dict; keys are AND-combined, list values
match if the chunk satisfies any element (OR within a key):

    {"doc_type": ["textbook"], "tags_any": ["degradation-mechanisms"],
     "year_min": 2010}

Supported keys:
    doc_id    list[str]  keep chunks from these documents
    doc_type  list[str]  keep these document types
    tags_any  list[str]  keep chunks whose document has at least one tag
    year_min  int        keep chunks with year >= year_min
    year_max  int        keep chunks with year <= year_max

Chunks whose document has ``year: null`` fail any year constraint
(fail-closed). Specs are validated against documents.json so a typo in a
tag or doc_type raises instead of silently matching nothing. Filtering is
applied pre-ranking in both retrievers, never by discarding ranked results.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RAG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_FILE = RAG_DIR / "documents.json"

ALLOWED_KEYS = {"doc_id", "doc_type", "tags_any", "year_min", "year_max"}
LIST_KEYS = {"doc_id", "doc_type", "tags_any"}
INT_KEYS = {"year_min", "year_max"}


def validate_spec(spec: dict, catalog_file: Path = DEFAULT_CATALOG_FILE) -> None:
    unknown = set(spec) - ALLOWED_KEYS
    if unknown:
        raise ValueError(f"unknown filter keys {sorted(unknown)}; allowed: {sorted(ALLOWED_KEYS)}")
    for key in LIST_KEYS & set(spec):
        value = spec[key]
        if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
            raise ValueError(f"filter key {key!r} must be a non-empty list of strings")
    for key in INT_KEYS & set(spec):
        if not isinstance(spec[key], int):
            raise ValueError(f"filter key {key!r} must be an integer")

    catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
    checks = [
        ("tags_any", set(catalog["_tag_vocabulary"]), "tag vocabulary"),
        ("doc_type", set(catalog["_doc_types"]), "doc types"),
        ("doc_id", set(catalog["documents"]), "catalogued documents"),
    ]
    for key, known, label in checks:
        unknown_values = set(spec.get(key, [])) - known
        if unknown_values:
            raise ValueError(
                f"filter key {key!r} has values not in the {label}: {sorted(unknown_values)}"
            )


def passes_filter(chunk: dict, spec: dict) -> bool:
    if "doc_id" in spec and chunk["doc_id"] not in spec["doc_id"]:
        return False
    if "doc_type" in spec and chunk["doc_type"] not in spec["doc_type"]:
        return False
    if "tags_any" in spec and not set(chunk["tags"]) & set(spec["tags_any"]):
        return False
    if "year_min" in spec and (chunk["year"] is None or chunk["year"] < spec["year_min"]):
        return False
    if "year_max" in spec and (chunk["year"] is None or chunk["year"] > spec["year_max"]):
        return False
    return True


def allowed_mask(chunks: list[dict], spec: dict) -> np.ndarray:
    """Boolean mask over ``chunks`` (index-aligned) for a validated spec."""
    return np.array([passes_filter(chunk, spec) for chunk in chunks], dtype=bool)
