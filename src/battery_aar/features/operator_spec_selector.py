"""Deterministic mapping from feature-operator specs to concrete columns.

This is the Tier B substrate (see docs/structured_feature_specs_plan.md): the
FeatureScientist proposes a list of operator specs (an ``operator_type`` from a
fixed menu plus optional ``y_signal`` / ``aggregation`` params), and this module
resolves those specs to actual columns of an already-materialized feature table
by filtering its ``feature_metadata``. No raw data and no LLM-written code are
involved -- the same spec always resolves to the same columns, which is what
makes agent performance depend on *which operators it chose*, not on how the LLM
happened to phrase code.

Selecting under a hard ``budget`` is the whole point: with a small budget, which
operators the agent picks is a knowledge decision (the regime where retrieved
literature can actually help a weak model), not a "take everything" default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# Columns of the materialized feature_metadata we match specs against. These are
# present in the broad_physics program tables (see program_library.py output).
_OPERATOR_COL = "operator_type"
_SIGNAL_COL = "y_signal"
_AGG_COL = "aggregation"
_NAME_COL = "feature_name"


@dataclass(frozen=True)
class OperatorSelection:
    """Result of resolving specs against a metadata table under a budget."""

    columns: list[str]
    per_spec_counts: list[int]  # columns contributed by each input spec (pre-budget)
    n_requested_specs: int
    n_matched_specs: int
    budget: int | None
    dropped_specs: list[dict[str, Any]] = field(default_factory=list)


def _match_columns_for_spec(metadata: pd.DataFrame, spec: dict[str, Any]) -> list[str]:
    """Columns whose metadata matches a single spec's operator/signal/aggregation."""
    mask = pd.Series(True, index=metadata.index)
    op = spec.get("operator_type") or spec.get("operator_name")
    if op is not None and _OPERATOR_COL in metadata.columns:
        mask &= metadata[_OPERATOR_COL].astype(str) == str(op)
    params = spec.get("params") or {}
    y_signal = spec.get("y_signal", params.get("y_signal"))
    if y_signal is not None and _SIGNAL_COL in metadata.columns:
        mask &= metadata[_SIGNAL_COL].astype(str) == str(y_signal)
    aggregation = spec.get("aggregation", params.get("aggregation"))
    if aggregation is not None and _AGG_COL in metadata.columns:
        mask &= metadata[_AGG_COL].astype(str) == str(aggregation)
    # Deterministic order: sorted feature names.
    return sorted(metadata.loc[mask, _NAME_COL].astype(str).unique().tolist())


def select_columns_for_specs(
    metadata: pd.DataFrame,
    specs: list[dict[str, Any]],
    budget: int | None = None,
) -> OperatorSelection:
    """Resolve operator specs to a concrete, budgeted column subset.

    Args:
        metadata: feature_metadata of a materialized program table; must carry
            at least ``feature_name`` and ``operator_type`` columns.
        specs: ordered list of operator specs (dicts). Order matters: under a
            tight budget, earlier specs are honored first (they are the agent's
            top picks).
        budget: maximum number of columns to return. ``None`` means no cap.

    Returns:
        OperatorSelection with the deterministic column list and bookkeeping.

    Selection is round-robin across matched specs so a budget is spread over the
    agent's distinct ideas rather than exhausted by the first broad operator.
    """
    if _NAME_COL not in metadata.columns:
        raise ValueError(f"metadata must contain a {_NAME_COL!r} column")

    matched: list[list[str]] = []
    per_spec_counts: list[int] = []
    dropped: list[dict[str, Any]] = []
    for spec in specs:
        cols = _match_columns_for_spec(metadata, spec)
        per_spec_counts.append(len(cols))
        if cols:
            matched.append(cols)
        else:
            dropped.append(dict(spec))

    # Round-robin flatten so the budget is shared across the agent's picks.
    ordered: list[str] = []
    seen: set[str] = set()
    if matched:
        max_len = max(len(cols) for cols in matched)
        for i in range(max_len):
            for cols in matched:
                if i < len(cols) and cols[i] not in seen:
                    ordered.append(cols[i])
                    seen.add(cols[i])

    if budget is not None and budget >= 0:
        ordered = ordered[:budget]

    return OperatorSelection(
        columns=ordered,
        per_spec_counts=per_spec_counts,
        n_requested_specs=len(specs),
        n_matched_specs=len(matched),
        budget=budget,
        dropped_specs=dropped,
    )
