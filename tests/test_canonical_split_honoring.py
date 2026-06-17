"""Patch F1: ensure ``make_search_split`` honors canonical splits.csv.

Before F1, ``make_search_split`` ignored the canonical
``{train, primary_test, secondary_test}`` tags written by patch E1's
splitter, re-shuffling the full pool. This invalidated surrogate metrics:
b3 (locked secondary test) leaked into surrogate training and validation.

This module verifies that:

* In ``random`` mode on a canonical dataset, neither train nor validation
  contains a ``secondary_test`` row.
* In ``batch`` mode on a canonical dataset, neither train nor validation
  contains a ``secondary_test`` row, and the search pool is limited to the
  canonical ``train`` batches.
* The new ``primary_test`` mode uses canonical ``train`` for fitting and
  canonical ``primary_test`` for validation.
* Legacy splits.csv (``train``/``validation``/``test`` values) preserves
  pre-patch behaviour and logs a warning about possible optimism.
* The hard leak-guard assertion fires when canonical ``secondary_test`` rows
  somehow end up in the synthesized split.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
import pytest

from battery_aar.agents.orchestrator import (
    _classify_split_table,
    _filter_canonical_search_pool,
    make_search_split,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _canonical_severson_like_metadata(
    *,
    n_b1: int = 26,
    n_b2: int = 20,
    n_b3: int = 44,
    seed: int = 1234,
) -> pd.DataFrame:
    """Build a Severson-shaped metadata DataFrame spanning b1, b2, b3.

    Includes the columns ``make_search_split`` looks at (``batch_id``,
    ``source_batch``, ``cell_id``, ``protocol_readable``, ``C1``/``C2``/``C3``/
    ``C4``).
    """
    rng = np.random.default_rng(seed)
    rows: List[dict] = []
    row_id = 0

    def add_batch(date: str, n: int, life_mean: float, protocol_offset: int) -> None:
        nonlocal row_id
        for cell_idx in range(n):
            c1 = 3.6 + 0.4 * ((cell_idx + protocol_offset) % 4)
            c2 = 4.0 + 0.4 * ((cell_idx + protocol_offset) % 3)
            c3 = 4.4 + 0.4 * ((cell_idx + protocol_offset) % 2)
            c4 = 4.16 + 0.001 * cell_idx
            rows.append(
                {
                    "row_id": row_id,
                    "cell_id": f"{date}_cell_{cell_idx:03d}",
                    "source_dataset": "severson_2019_true_life",
                    "source_file": f"{date}_batchdata.mat",
                    "source_batch": date,
                    "batch_id": date,
                    "cycle_life": float(rng.normal(loc=life_mean, scale=80.0)),
                    "label_source": "true_measured_cycle_life",
                    "protocol_readable": f"{c1:.1f}C-{c2:.1f}C-{c3:.1f}C-{c4:.3f}C",
                    "C1": c1,
                    "C2": c2,
                    "C3": c3,
                    "C4": c4,
                }
            )
            row_id += 1

    add_batch("2017-05-12", n_b1, life_mean=820.0, protocol_offset=0)
    add_batch("2017-06-30", n_b2, life_mean=780.0, protocol_offset=1)
    add_batch("2018-04-12", n_b3, life_mean=640.0, protocol_offset=2)
    return pd.DataFrame(rows)


def _labels_from_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    return meta[["row_id", "cycle_life"]].rename(columns={"cycle_life": "y"}).copy()


def _canonical_splits_table(meta: pd.DataFrame) -> pd.DataFrame:
    """Build a canonical splits.csv-like table for the meta fixture.

    Mirrors the production layout: b3 rows -> ``secondary_test``; within
    b1+b2, roughly half -> ``train`` and half -> ``primary_test``.
    """
    rng = np.random.default_rng(0)
    rows = []
    for _, row in meta.iterrows():
        batch = str(row["source_batch"])
        if batch == "2018-04-12":
            split = "secondary_test"
        else:
            # Stable 50/50 split: even cell_idx -> train, odd -> primary_test.
            cell_idx = int(row["row_id"]) % 2
            split = "train" if cell_idx == 0 else "primary_test"
        rows.append(
            {
                "row_id": int(row["row_id"]),
                "cell_id": row["cell_id"],
                "source_file": row["source_file"],
                "source_batch": batch,
                "batch_id": batch,
                "split": split,
                "split_legacy": (
                    "train" if batch == "2017-05-12"
                    else "validation" if batch == "2017-06-30"
                    else "test"
                ),
                "split_source": "canonical_severson",
            }
        )
    # Shuffle for realism.
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=rng.integers(0, 2**31)).reset_index(drop=True)
    return df


def _legacy_splits_table(meta: pd.DataFrame) -> pd.DataFrame:
    """Legacy split table using {train, validation, test} values."""
    rows = []
    for _, row in meta.iterrows():
        batch = str(row["source_batch"])
        split = (
            "train" if batch == "2017-05-12"
            else "validation" if batch == "2017-06-30"
            else "test"
        )
        rows.append(
            {
                "row_id": int(row["row_id"]),
                "cell_id": row["cell_id"],
                "source_batch": batch,
                "split": split,
                "split_source": "source_file_identity_default",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _classify_split_table
# ---------------------------------------------------------------------------


def test_classify_split_table_returns_absent_when_none():
    assert _classify_split_table(None) == "absent"
    assert _classify_split_table(pd.DataFrame()) == "absent"


def test_classify_split_table_detects_canonical():
    meta = _canonical_severson_like_metadata()
    splits = _canonical_splits_table(meta)
    assert _classify_split_table(splits) == "canonical"


def test_classify_split_table_detects_legacy():
    meta = _canonical_severson_like_metadata()
    splits = _legacy_splits_table(meta)
    assert _classify_split_table(splits) == "legacy"


def test_classify_split_table_detects_canonical_without_split_source_column():
    meta = _canonical_severson_like_metadata()
    splits = _canonical_splits_table(meta).drop(columns=["split_source"])
    assert _classify_split_table(splits) == "canonical"


# ---------------------------------------------------------------------------
# _filter_canonical_search_pool
# ---------------------------------------------------------------------------


def test_filter_canonical_search_pool_partitions_by_tag():
    meta = _canonical_severson_like_metadata()
    labels = _labels_from_metadata(meta)
    splits = _canonical_splits_table(meta)

    train_ids, primary_ids, secondary_ids = _filter_canonical_search_pool(labels, splits)

    # Sets are disjoint and their union equals the labeled set.
    assert train_ids.isdisjoint(primary_ids)
    assert train_ids.isdisjoint(secondary_ids)
    assert primary_ids.isdisjoint(secondary_ids)
    assert (train_ids | primary_ids | secondary_ids) == set(int(rid) for rid in labels["row_id"])

    # secondary_test is exactly the b3 rows (44 cells).
    b3_rows = set(int(rid) for rid in meta.loc[meta["source_batch"] == "2018-04-12", "row_id"])
    assert secondary_ids == b3_rows


# ---------------------------------------------------------------------------
# Canonical: random mode
# ---------------------------------------------------------------------------


def test_random_mode_on_canonical_dataset_excludes_secondary_test():
    meta = _canonical_severson_like_metadata()
    labels = _labels_from_metadata(meta)
    splits = _canonical_splits_table(meta)

    train_ids, val_ids, _, manifest, assignments = make_search_split(
        metadata=meta,
        labels=labels,
        split_mode="random",
        validation_fraction=0.25,
        split_seed=1,
        splits_table=splits,
    )

    secondary_rows = set(int(rid) for rid in meta.loc[meta["source_batch"] == "2018-04-12", "row_id"])
    canonical_train = set(int(rid) for rid in splits.loc[splits["split"] == "train", "row_id"])

    # Critical: no b3 (secondary_test) row leaked into either split.
    assert set(train_ids).isdisjoint(secondary_rows), (
        f"random mode leaked {len(set(train_ids) & secondary_rows)} secondary_test rows into train"
    )
    assert set(val_ids).isdisjoint(secondary_rows), (
        f"random mode leaked {len(set(val_ids) & secondary_rows)} secondary_test rows into val"
    )

    # And: the pool was restricted to the canonical train rows.
    assert set(train_ids) | set(val_ids) == canonical_train
    assert manifest["split_table_kind"] == "canonical"
    assert manifest["canonical_secondary_test_pool_size"] == len(secondary_rows)

    # assignments table only covers the search pool (no b3 rows).
    assignment_row_ids = set(assignments["row_id"].astype(int).tolist())
    assert assignment_row_ids.isdisjoint(secondary_rows)


# ---------------------------------------------------------------------------
# Canonical: batch mode
# ---------------------------------------------------------------------------


def test_batch_mode_on_canonical_dataset_excludes_secondary_test():
    meta = _canonical_severson_like_metadata()
    labels = _labels_from_metadata(meta)
    splits = _canonical_splits_table(meta)

    train_ids, val_ids, _, manifest, _ = make_search_split(
        metadata=meta,
        labels=labels,
        split_mode="batch",
        validation_fraction=0.5,  # leave one batch out of two
        split_seed=0,
        splits_table=splits,
    )

    secondary_rows = set(int(rid) for rid in meta.loc[meta["source_batch"] == "2018-04-12", "row_id"])
    canonical_train = set(int(rid) for rid in splits.loc[splits["split"] == "train", "row_id"])

    assert set(train_ids).isdisjoint(secondary_rows)
    assert set(val_ids).isdisjoint(secondary_rows)
    # Search pool came entirely from canonical train.
    assert set(train_ids) | set(val_ids) <= canonical_train
    # Held-out group is one of b1/b2, never b3.
    assert "2018-04-12" not in manifest["heldout_batch_ids"]
    assert set(manifest["heldout_batch_ids"]).issubset({"2017-05-12", "2017-06-30"})


# ---------------------------------------------------------------------------
# Canonical: new primary_test mode
# ---------------------------------------------------------------------------


def test_primary_test_mode_uses_canonical_pools_directly():
    meta = _canonical_severson_like_metadata()
    labels = _labels_from_metadata(meta)
    splits = _canonical_splits_table(meta)

    train_ids, val_ids, _, manifest, _ = make_search_split(
        metadata=meta,
        labels=labels,
        split_mode="primary_test",
        splits_table=splits,
    )

    canonical_train = set(int(rid) for rid in splits.loc[splits["split"] == "train", "row_id"])
    canonical_primary = set(int(rid) for rid in splits.loc[splits["split"] == "primary_test", "row_id"])
    canonical_secondary = set(int(rid) for rid in splits.loc[splits["split"] == "secondary_test", "row_id"])

    assert set(train_ids) == canonical_train
    assert set(val_ids) == canonical_primary
    # No secondary leaks regardless.
    assert set(train_ids).isdisjoint(canonical_secondary)
    assert set(val_ids).isdisjoint(canonical_secondary)
    assert manifest["split_mode"] == "primary_test"
    assert manifest["group_type"] == "canonical_primary_test"


def test_primary_test_mode_rejects_legacy_dataset():
    meta = _canonical_severson_like_metadata()
    labels = _labels_from_metadata(meta)
    splits = _legacy_splits_table(meta)

    with pytest.raises(ValueError, match="requires a canonical splits.csv"):
        make_search_split(
            metadata=meta,
            labels=labels,
            split_mode="primary_test",
            splits_table=splits,
        )


# ---------------------------------------------------------------------------
# Legacy datasets: backwards compatibility + warning
# ---------------------------------------------------------------------------


def test_legacy_dataset_preserves_existing_behaviour_and_warns(caplog):
    meta = _canonical_severson_like_metadata()
    labels = _labels_from_metadata(meta)
    splits = _legacy_splits_table(meta)

    with caplog.at_level(logging.WARNING, logger="battery_aar.agents.orchestrator"):
        train_ids, val_ids, _, manifest, _ = make_search_split(
            metadata=meta,
            labels=labels,
            split_mode="random",
            validation_fraction=0.25,
            split_seed=7,
            splits_table=splits,
        )

    # Warning was logged about legacy mode.
    assert any("Patch F1" in record.message and "legacy" in record.message.lower() for record in caplog.records)
    # All labeled rows are still in play (legacy mode does not filter).
    full_labeled = set(int(rid) for rid in labels["row_id"])
    assert set(train_ids) | set(val_ids) == full_labeled
    assert manifest["split_table_kind"] == "legacy"
    assert manifest["canonical_secondary_test_pool_size"] is None


def test_no_splits_table_still_works():
    """When no splits.csv exists, behaviour matches the pre-patch logic
    (search-pool = full labeled set, no warning, kind = absent)."""
    meta = _canonical_severson_like_metadata(n_b3=0)  # no b3 to keep test honest
    labels = _labels_from_metadata(meta)

    train_ids, val_ids, _, manifest, _ = make_search_split(
        metadata=meta,
        labels=labels,
        split_mode="random",
        validation_fraction=0.25,
        split_seed=0,
        splits_table=None,
    )

    assert manifest["split_table_kind"] == "absent"
    full_labeled = set(int(rid) for rid in labels["row_id"])
    assert set(train_ids) | set(val_ids) == full_labeled


# ---------------------------------------------------------------------------
# Hard leak guard
# ---------------------------------------------------------------------------


def test_leak_guard_fires_if_secondary_test_row_appears_in_search_pool(monkeypatch):
    """If a future regression somehow lets a secondary_test row through, the
    runtime assert at the end of ``make_search_split`` must catch it.

    We simulate the regression by monkey-patching
    ``_filter_canonical_search_pool`` so it advertises b3 row_ids as BOTH
    canonical-train (so the up-front filter step pushes them into the search
    pool) and canonical-secondary (so the final guard knows they should never
    appear in train/val). The guard at the bottom of ``make_search_split``
    must catch the contradiction and raise.
    """
    from battery_aar.agents import orchestrator as orch

    meta = _canonical_severson_like_metadata()
    labels = _labels_from_metadata(meta)
    splits = _canonical_splits_table(meta)

    real_filter = orch._filter_canonical_search_pool

    def fake_filter(labels_df, splits_df):
        train, primary, secondary = real_filter(labels_df, splits_df)
        # Regression simulation: also tag every secondary row as canonical
        # train, so the up-front pool restriction lets them through. The
        # `canonical_secondary` set remains accurate, so the guard at the end
        # has the information it needs to flag the leak.
        return train | secondary, primary, secondary

    monkeypatch.setattr(orch, "_filter_canonical_search_pool", fake_filter)

    with pytest.raises(RuntimeError, match="Patch F1 leak guard"):
        orch.make_search_split(
            metadata=meta,
            labels=labels,
            split_mode="random",
            validation_fraction=0.25,
            split_seed=0,
            splits_table=splits,
        )
