"""Offline test for the exact brute-force cosine fallback used under
selective metadata filters. Pure numpy -- no embedding API call involved.
"""

import numpy as np
import pytest

from battery_aar.rag.scripts.embed_index import _brute_force_search


def test_brute_force_search_matches_manual_cosine_ranking():
    embeddings = np.array(
        [
            [1.0, 0.0],  # label 0: parallel to the query
            [0.0, 1.0],  # label 1: orthogonal to the query
            [0.9, 0.1],  # label 2: close to the query
        ],
        dtype=np.float32,
    )
    query_vector = np.array([[1.0, 0.0]], dtype=np.float32)
    allowed_labels = np.array([0, 1, 2])

    results = _brute_force_search(query_vector, allowed_labels, embeddings, k=3)

    order = [label for label, _ in results]
    assert order == [0, 2, 1]  # closest to farthest
    assert results[0][1] == pytest.approx(1.0)  # cosine similarity to itself
    assert results[2][1] == pytest.approx(0.0)  # orthogonal


def test_brute_force_search_respects_the_allowed_subset():
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    query_vector = np.array([[1.0, 0.0]], dtype=np.float32)
    allowed_labels = np.array([1])  # only label 1 is eligible

    results = _brute_force_search(query_vector, allowed_labels, embeddings, k=3)

    assert [label for label, _ in results] == [1]


def test_brute_force_search_respects_k():
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    query_vector = np.array([[1.0, 0.0]], dtype=np.float32)
    allowed_labels = np.array([0, 1, 2])

    results = _brute_force_search(query_vector, allowed_labels, embeddings, k=2)

    assert len(results) == 2
