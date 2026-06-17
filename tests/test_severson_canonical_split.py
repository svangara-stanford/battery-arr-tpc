"""Smoke tests for the Severson 2019 canonical train/primary/secondary splitter.

These tests use synthetic in-memory fixtures (no real .mat files required) to
exercise the canonical split logic added in patch E1. They focus on structural
invariants rather than matching BatteryML's exact cell IDs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import pytest

from battery_aar.features.severson_matr import (
    SEVERSON_B1_DATE,
    SEVERSON_B2_DATE,
    SEVERSON_B3_DATE,
    SEVERSON_CANONICAL_TRAIN_FRACTION,
    _canonical_severson_splits,
    _cycle_life_split_balance,
    _extract_batch_date,
    _legacy_split_map,
    _load_canonical_exclusions,
    build_severson_true_life_dataset,
)

from .severson_test_utils import write_toy_severson_mat_dir


def _synthetic_metadata(
    *,
    n_b1: int = 42,
    n_b2: int = 42,
    n_b3: int = 40,
    seed: int = 1234,
) -> pd.DataFrame:
    """Build a synthetic Severson-shaped cell_metadata DataFrame."""

    rng = np.random.default_rng(seed)
    rows: List[dict] = []
    row_id = 0

    def add_batch(date: str, n: int, life_mean: float) -> None:
        nonlocal row_id
        lives = rng.normal(loc=life_mean, scale=120.0, size=n).clip(min=150.0)
        for cell_idx in range(n):
            rows.append(
                {
                    "row_id": row_id,
                    "cell_id": f"{date.replace('-', '')}_cell_{cell_idx:03d}",
                    "source_dataset": "severson_2019_true_life",
                    "source_file": f"{date}_batchdata_updated_struct_errorcorrect.mat",
                    "source_batch": f"{date}_batchdata_updated_struct_errorcorrect",
                    "batch_id": f"{date}_batchdata_updated_struct_errorcorrect",
                    "source_cell_index": cell_idx,
                    "cycle_life": float(lives[cell_idx]),
                    "label_source": "true_measured_cycle_life",
                }
            )
            row_id += 1

    add_batch(SEVERSON_B1_DATE, n_b1, life_mean=820.0)
    add_batch(SEVERSON_B2_DATE, n_b2, life_mean=780.0)
    add_batch(SEVERSON_B3_DATE, n_b3, life_mean=640.0)
    return pd.DataFrame(rows)


def test_extract_batch_date_parses_yyyy_mm_dd_prefix():
    assert _extract_batch_date("2017-05-12_batchdata_updated_struct_errorcorrect") == "2017-05-12"
    assert _extract_batch_date("2018-04-12") == "2018-04-12"
    assert _extract_batch_date("not_a_date") is None
    assert _extract_batch_date("") is None


def test_canonical_split_secondary_test_contains_only_b3_cells():
    meta = _synthetic_metadata()
    splits, diagnostics = _canonical_severson_splits(meta, seed=0)

    secondary = meta.loc[splits.eq("secondary_test")]
    assert not secondary.empty
    secondary_dates = secondary["source_batch"].map(_extract_batch_date).unique().tolist()
    assert secondary_dates == [SEVERSON_B3_DATE]
    # And: every b3 cell ends up in secondary_test.
    assert int((splits.eq("secondary_test")).sum()) == int(
        (meta["source_batch"].map(_extract_batch_date).eq(SEVERSON_B3_DATE)).sum()
    )
    assert diagnostics["n_b3_cells"] == int(splits.eq("secondary_test").sum())


def test_canonical_split_train_and_primary_are_b12_cells_only():
    meta = _synthetic_metadata()
    splits, _ = _canonical_severson_splits(meta, seed=0)

    for split_name in ("train", "primary_test"):
        rows = meta.loc[splits.eq(split_name)]
        dates = set(rows["source_batch"].map(_extract_batch_date).unique().tolist())
        assert dates.issubset({SEVERSON_B1_DATE, SEVERSON_B2_DATE}), (
            f"{split_name} unexpectedly contains non-b1/b2 cells: {dates}"
        )


def test_canonical_split_is_approximately_5050_for_b12():
    meta = _synthetic_metadata(n_b1=42, n_b2=42, n_b3=40)
    splits, _ = _canonical_severson_splits(meta, seed=0)

    n_train = int(splits.eq("train").sum())
    n_primary = int(splits.eq("primary_test").sum())
    total_b12 = n_train + n_primary
    assert total_b12 == 84
    # 0.488 * 84 = 41.0 -> train=41, primary_test=43
    assert n_train == int(round(84 * SEVERSON_CANONICAL_TRAIN_FRACTION))
    assert n_train + n_primary == 84
    # ~50/50 sanity (each within +-5 of half)
    assert abs(n_train - n_primary) <= 6


def test_canonical_split_is_deterministic_under_same_seed():
    meta = _synthetic_metadata()
    splits_a, _ = _canonical_severson_splits(meta, seed=0)
    splits_b, _ = _canonical_severson_splits(meta, seed=0)
    pd.testing.assert_series_equal(splits_a, splits_b)


def test_canonical_split_differs_under_different_seeds():
    meta = _synthetic_metadata()
    splits_seed0, _ = _canonical_severson_splits(meta, seed=0)
    splits_seed7, _ = _canonical_severson_splits(meta, seed=7)
    # Same split counts (deterministic structure) but at least one assignment
    # differs (different shuffle).
    assert (splits_seed0 != splits_seed7).any()


def test_cycle_life_split_balance_warns_on_large_delta():
    meta = _synthetic_metadata()
    # Construct a pathological split: low-life cells -> train, high-life ->
    # primary_test.
    sort_idx = meta.sort_values("cycle_life").index.tolist()
    is_b12 = meta["source_batch"].map(_extract_batch_date).isin({SEVERSON_B1_DATE, SEVERSON_B2_DATE})
    b12_sorted = [idx for idx in sort_idx if is_b12.loc[idx]]
    half = len(b12_sorted) // 2
    splits = pd.Series(index=meta.index, dtype=object)
    splits.loc[b12_sorted[:half]] = "train"
    splits.loc[b12_sorted[half:]] = "primary_test"
    splits.loc[~is_b12] = "secondary_test"

    with pytest.warns(RuntimeWarning, match="differs from primary_test mean"):
        report = _cycle_life_split_balance(meta, splits, tolerance=50.0)
    assert report["balanced"] is False
    assert report["train_primary_abs_delta"] is not None
    assert report["train_primary_abs_delta"] > 50.0


def test_load_canonical_exclusions_handles_comments_and_blank_lines(tmp_path: Path):
    exclusion_path = tmp_path / "ex.txt"
    exclusion_path.write_text(
        """# header comment

# another comment
b1c0
b2c7

b3c37
"""
    )
    ids, comments = _load_canonical_exclusions(exclusion_path)
    assert ids == {"b1c0", "b2c7", "b3c37"}
    # First two non-blank lines are comments.
    assert any("header" in c for c in comments)
    assert any("another" in c for c in comments)


def test_load_canonical_exclusions_returns_empty_when_path_is_none():
    ids, comments = _load_canonical_exclusions(None)
    assert ids == set()
    assert comments == []


def test_legacy_split_map_matches_historical_behaviour():
    files = [
        "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
        "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
        "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
    ]
    mapping = _legacy_split_map(files)
    assert mapping[files[0]] == "train"
    assert mapping[files[1]] == "validation"
    assert mapping[files[2]] == "test"


def test_build_dataset_canonical_split_with_toy_mat_files(tmp_path: Path):
    """End-to-end check on synthetic .mat files spanning b1, b2, b3.

    Verifies: split column values, split_legacy preserved, dataset_card carries
    new fields, and deterministic seed reproduction.
    """

    mat_dir = write_toy_severson_mat_dir(tmp_path, n_files=3, n_cells_per_file=4)
    out_a = tmp_path / "canon_a"
    out_b = tmp_path / "canon_b"

    card_a = build_severson_true_life_dataset(
        mat_dir=mat_dir,
        out_dir=out_a,
        first_n_cycles=100,
        split_mode="canonical_severson",
        split_seed=0,
    )
    card_b = build_severson_true_life_dataset(
        mat_dir=mat_dir,
        out_dir=out_b,
        first_n_cycles=100,
        split_mode="canonical_severson",
        split_seed=0,
    )

    splits_a = pd.read_csv(out_a / "splits.csv")
    splits_b = pd.read_csv(out_b / "splits.csv")

    # Schema checks
    assert "split" in splits_a.columns
    assert "split_legacy" in splits_a.columns
    assert set(splits_a["split"].unique()).issubset({"train", "primary_test", "secondary_test"})
    assert set(splits_a["split_legacy"].unique()).issubset({"train", "validation", "test"})

    # Secondary test == b3 cells only.
    secondary_dates = (
        splits_a.loc[splits_a["split"].eq("secondary_test"), "source_batch"]
        .astype(str)
        .map(_extract_batch_date)
        .unique()
        .tolist()
    )
    assert secondary_dates == [SEVERSON_B3_DATE]

    # Determinism across runs (same seed -> identical split assignments).
    pd.testing.assert_series_equal(
        splits_a.set_index("cell_id")["split"].sort_index(),
        splits_b.set_index("cell_id")["split"].sort_index(),
        check_names=False,
    )

    # Card carries new canonical fields.
    for key in (
        "split_mode",
        "split_seed",
        "train_cells",
        "primary_test_cells",
        "secondary_test_cells",
        "train_mean_cycle_life",
        "primary_test_mean_cycle_life",
        "secondary_test_mean_cycle_life",
        "canonical_exclusions_applied",
    ):
        assert key in card_a, f"missing dataset_card key: {key}"
    assert card_a["split_mode"] == "canonical_severson"
    assert card_a["split_seed"] == 0
    assert card_a["train_cells"] + card_a["primary_test_cells"] + card_a["secondary_test_cells"] == card_a["included_cells"]


def test_build_dataset_source_file_identity_mode_preserves_legacy_values(tmp_path: Path):
    mat_dir = write_toy_severson_mat_dir(tmp_path, n_files=3, n_cells_per_file=2)
    out = tmp_path / "legacy"

    card = build_severson_true_life_dataset(
        mat_dir=mat_dir,
        out_dir=out,
        first_n_cycles=100,
        split_mode="source_file_identity",
    )
    splits = pd.read_csv(out / "splits.csv")

    assert card["split_mode"] == "source_file_identity"
    assert set(splits["split"].unique()).issubset({"train", "validation", "test"})
    # split_legacy column also present and equal to split in legacy mode.
    assert (splits["split"] == splits["split_legacy"]).all()


def test_build_dataset_applies_canonical_exclusions(tmp_path: Path):
    mat_dir = write_toy_severson_mat_dir(tmp_path, n_files=2, n_cells_per_file=2)

    # Build once to discover the auto-generated cell_ids.
    discover = tmp_path / "discover"
    discover_card = build_severson_true_life_dataset(
        mat_dir=mat_dir,
        out_dir=discover,
        first_n_cycles=100,
        split_mode="source_file_identity",
    )
    metadata = pd.read_csv(discover / "cell_metadata.csv")
    assert len(metadata) == discover_card["included_cells"]

    # Drop the first cell.
    drop_id = metadata.iloc[0]["cell_id"]
    exclusions_file = tmp_path / "ex.txt"
    exclusions_file.write_text(f"# test exclusion\n{drop_id}\n")

    out = tmp_path / "with_exclusion"
    card = build_severson_true_life_dataset(
        mat_dir=mat_dir,
        out_dir=out,
        first_n_cycles=100,
        split_mode="canonical_severson",
        split_seed=0,
        exclusion_cell_ids=exclusions_file,
    )
    assert card["canonical_exclusions_count"] == 1
    assert drop_id in card["canonical_exclusions_applied"]
    assert card["included_cells"] == discover_card["included_cells"] - 1

    exclusions_csv = pd.read_csv(out / "exclusions.csv")
    canonical_rows = exclusions_csv[exclusions_csv["exclusion_stage"].eq("canonical_exclusion")]
    assert drop_id in canonical_rows["cell_id"].astype(str).tolist()
