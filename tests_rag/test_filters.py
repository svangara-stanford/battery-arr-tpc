"""Offline tests for the declarative metadata filter spec."""

import json

import numpy as np
import pytest

from battery_aar.rag.scripts.filters import allowed_mask, passes_filter, validate_spec

CHUNK_A = {"doc_id": "docA", "doc_type": "textbook", "tags": ["degradation-mechanisms"], "year": 2020}
CHUNK_B = {"doc_id": "docB", "doc_type": "paper", "tags": ["charging-protocols"], "year": None}


def _write_catalog(tmp_path):
    catalog = {
        "_tag_vocabulary": {"degradation-mechanisms": "desc", "charging-protocols": "desc"},
        "_doc_types": ["textbook", "paper", "review"],
        "documents": {"docA": {}, "docB": {}},
    }
    path = tmp_path / "documents.json"
    path.write_text(json.dumps(catalog))
    return path


# --- passes_filter -----------------------------------------------------


def test_passes_filter_empty_spec_matches_everything():
    assert passes_filter(CHUNK_A, {}) is True


def test_passes_filter_doc_id():
    assert passes_filter(CHUNK_A, {"doc_id": ["docA"]}) is True
    assert passes_filter(CHUNK_A, {"doc_id": ["docB"]}) is False


def test_passes_filter_doc_type():
    assert passes_filter(CHUNK_A, {"doc_type": ["textbook"]}) is True
    assert passes_filter(CHUNK_A, {"doc_type": ["paper"]}) is False


def test_passes_filter_tags_any_is_an_or():
    assert passes_filter(CHUNK_A, {"tags_any": ["degradation-mechanisms", "charging-protocols"]}) is True
    assert passes_filter(CHUNK_A, {"tags_any": ["charging-protocols"]}) is False


def test_passes_filter_year_min_and_max():
    assert passes_filter(CHUNK_A, {"year_min": 2020}) is True
    assert passes_filter(CHUNK_A, {"year_min": 2021}) is False
    assert passes_filter(CHUNK_A, {"year_max": 2020}) is True
    assert passes_filter(CHUNK_A, {"year_max": 2019}) is False


def test_passes_filter_null_year_fails_closed():
    assert passes_filter(CHUNK_B, {"year_min": 1900}) is False
    assert passes_filter(CHUNK_B, {"year_max": 2100}) is False


def test_passes_filter_keys_are_and_combined():
    spec = {"doc_type": ["textbook"], "tags_any": ["degradation-mechanisms"], "year_min": 2020}
    assert passes_filter(CHUNK_A, spec) is True
    assert passes_filter(CHUNK_A, {**spec, "year_min": 2021}) is False


# --- validate_spec -------------------------------------------------------


def test_validate_spec_accepts_a_well_formed_spec(tmp_path):
    catalog_file = _write_catalog(tmp_path)
    validate_spec({"doc_type": ["textbook"], "tags_any": ["degradation-mechanisms"]}, catalog_file)


def test_validate_spec_rejects_unknown_key(tmp_path):
    catalog_file = _write_catalog(tmp_path)
    with pytest.raises(ValueError, match="unknown filter keys"):
        validate_spec({"bogus_key": ["x"]}, catalog_file)


@pytest.mark.parametrize(
    "spec",
    [
        {"doc_type": "textbook"},  # not a list
        {"doc_type": []},  # empty list
        {"doc_type": [1]},  # non-string element
    ],
)
def test_validate_spec_rejects_malformed_list_values(tmp_path, spec):
    catalog_file = _write_catalog(tmp_path)
    with pytest.raises(ValueError, match="non-empty list of strings"):
        validate_spec(spec, catalog_file)


def test_validate_spec_rejects_non_integer_year(tmp_path):
    catalog_file = _write_catalog(tmp_path)
    with pytest.raises(ValueError, match="must be an integer"):
        validate_spec({"year_min": "2020"}, catalog_file)


def test_validate_spec_rejects_unknown_tag(tmp_path):
    catalog_file = _write_catalog(tmp_path)
    with pytest.raises(ValueError, match="tag vocabulary"):
        validate_spec({"tags_any": ["not-a-real-tag"]}, catalog_file)


def test_validate_spec_rejects_unknown_doc_type(tmp_path):
    catalog_file = _write_catalog(tmp_path)
    with pytest.raises(ValueError, match="doc types"):
        validate_spec({"doc_type": ["blog-post"]}, catalog_file)


def test_validate_spec_rejects_unknown_doc_id(tmp_path):
    catalog_file = _write_catalog(tmp_path)
    with pytest.raises(ValueError, match="catalogued documents"):
        validate_spec({"doc_id": ["docZ"]}, catalog_file)


# --- allowed_mask ----------------------------------------------------------


def test_allowed_mask_matches_passes_filter_elementwise():
    chunks = [CHUNK_A, CHUNK_B]
    mask = allowed_mask(chunks, {"doc_type": ["textbook"]})
    assert mask.dtype == np.bool_
    assert mask.tolist() == [True, False]
