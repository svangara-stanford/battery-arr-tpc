from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field


class FeatureOperatorOutput(BaseModel):
    features: Dict[str, float] = Field(default_factory=dict)
    metadata: List[Dict[str, Any]] = Field(default_factory=list)
    status: str = "ok"
    warnings: List[str] = Field(default_factory=list)


FeatureOperatorFn = Callable[..., FeatureOperatorOutput]


class FeatureOperatorRegistry:
    def __init__(self) -> None:
        self._operators: dict[str, FeatureOperatorFn] = {}
        self._descriptors: dict[str, dict[str, Any]] = {}

    def register(self, name: str, fn: FeatureOperatorFn, **descriptor: Any) -> None:
        self._operators[name] = fn
        self._descriptors[name] = {"name": name, **descriptor}

    def get(self, name: str) -> FeatureOperatorFn:
        if name not in self._operators:
            raise KeyError(f"Unknown feature operator: {name}")
        return self._operators[name]

    def available_operators(self) -> List[Dict[str, Any]]:
        return [self._descriptors[name] for name in sorted(self._operators)]


def default_operator_registry() -> FeatureOperatorRegistry:
    from battery_aar.features.operators import (
        cross_cycle_curve_delta,
        cross_cycle_scalar_delta,
        curve_shape,
        cycle_scalar,
        cycle_window_stats,
        generic_timeseries_placeholder,
        learned_embedding_placeholder,
        protocol,
    )

    registry = FeatureOperatorRegistry()
    registry.register("cycle_scalar", cycle_scalar, operator_type="cycle_scalar", family="scalar_summary")
    registry.register("cycle_window_stats", cycle_window_stats, operator_type="cycle_window_stats", family="scalar_summary")
    registry.register("cross_cycle_scalar_delta", cross_cycle_scalar_delta, operator_type="cross_cycle_scalar_delta", family="scalar_delta")
    registry.register("curve_shape", curve_shape, operator_type="curve_shape", family="true_curve_shape")
    registry.register("cross_cycle_curve_delta", cross_cycle_curve_delta, operator_type="cross_cycle_curve_delta", family="true_curve_difference")
    registry.register("protocol", protocol, operator_type="protocol", family="protocol")
    registry.register("learned_embedding_placeholder", learned_embedding_placeholder, operator_type="learned_embedding_placeholder", family="learned_embedding_placeholder")
    registry.register("generic_timeseries_placeholder", generic_timeseries_placeholder, operator_type="generic_timeseries_placeholder", family="generic_timeseries_placeholder")
    return registry


def resolve_operator(registry: Optional[FeatureOperatorRegistry], name: str) -> FeatureOperatorFn:
    return (registry or default_operator_registry()).get(name)
