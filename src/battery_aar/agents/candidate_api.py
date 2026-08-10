from __future__ import annotations

import importlib.util
import io
import multiprocessing as mp
import contextlib
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .sandbox import FORBIDDEN_PATH_PATTERNS, install_open_guard, validate_code_safety


@dataclass
class CandidateRunResult:
    success: bool
    predictions: pd.DataFrame | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    error_type: str | None = None
    failure_reason: str | None = None
    traceback: str | None = None
    syntax_status: str = "unknown"
    features_used: dict[str, Any] | None = None


def load_candidate(path: str | Path):
    candidate_path = Path(path)
    validate_code_safety(candidate_path.read_text())
    spec = importlib.util.spec_from_file_location(f"oba_candidate_{candidate_path.stem}", candidate_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import candidate from {candidate_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "fit") and hasattr(module, "predict"):
        return module
    if hasattr(module, "CandidateModel"):
        return module.CandidateModel()
    raise ValueError("candidate must define fit/predict functions or CandidateModel")


def _install_feature_capture(phase: dict[str, str], calls: list[dict[str, Any]]) -> None:
    """Wrap the shared feature builder so the exact columns each candidate
    trains/predicts on are recorded, regardless of what the generated code does.

    Runs inside the sandbox subprocess before the candidate module is loaded, so
    the candidate's `from battery_aar.features.battery_lifetime_features import
    build_all_battery_features` binds the recording wrapper.
    """
    from battery_aar.features import battery_lifetime_features as _blf

    original = _blf.build_all_battery_features

    def _recording(*args, **kwargs):
        out = original(*args, **kwargs)
        features = out[0] if isinstance(out, tuple) else out
        try:
            columns = [str(col) for col in features.columns]
            n_rows = int(features.shape[0])
        except Exception:
            columns, n_rows = [], 0
        calls.append(
            {
                "phase": phase["current"],
                "n_rows": n_rows,
                "n_features": len(columns),
                "feature_columns": columns,
                "call_kwargs": {
                    key: (list(value) if isinstance(value, tuple) else value)
                    for key, value in kwargs.items()
                    if isinstance(value, (str, int, float, bool, list, tuple, type(None)))
                },
            }
        )
        return out

    _blf.build_all_battery_features = _recording


def _model_state_feature_columns(model: Any) -> list[str] | None:
    """Best-effort extraction of the columns a fitted candidate ACTUALLY used.

    Compiled and baseline candidates return a state dict carrying
    ``feature_columns`` (the final, post-selection column list). This is the
    authoritative count of what the model trained on -- unlike the feature
    builder's output, which is captured pre-filter (e.g. before an operator-spec
    column allowlist narrows 630 built columns down to a budgeted few).
    """
    if isinstance(model, dict):
        cols = model.get("feature_columns")
        if isinstance(cols, (list, tuple)):
            return [str(c) for c in cols]
    return None


def _features_used_summary(
    calls: list[dict[str, Any]], fit_columns: list[str] | None = None
) -> dict[str, Any]:
    fit_calls = [call for call in calls if call["phase"] == "fit"]
    last_fit = fit_calls[-1] if fit_calls else None
    builder_columns = list(last_fit["feature_columns"]) if last_fit else []
    # Prefer the model's actual fit columns; fall back to the builder output
    # (e.g. for freeform candidates that don't return a feature_columns state).
    if fit_columns is not None:
        actual_columns = list(fit_columns)
    else:
        actual_columns = builder_columns
    return {
        "capture_method": (
            "model_state_feature_columns" if fit_columns is not None
            else "runtime_wrap_build_all_battery_features"
        ),
        "library_builder_called": bool(calls),
        "fit_feature_columns": actual_columns,
        "n_fit_features": len(actual_columns),
        # Retain the pre-filter builder output for observability / debugging.
        "n_builder_features": len(builder_columns),
        "calls": calls,
    }


def _worker(queue, path, train_metadata, train_cycle_summary, train_labels, test_metadata, test_cycle_summary, config, forbidden_patterns):
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    phase = {"current": "import"}
    feature_calls: list[dict[str, Any]] = []
    model = None
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            install_open_guard(tuple(forbidden_patterns))
            _install_feature_capture(phase, feature_calls)
            candidate = load_candidate(path)
            phase["current"] = "fit"
            if isinstance(candidate, types.ModuleType):
                model = candidate.fit(train_metadata, train_cycle_summary, train_labels, config)
                phase["current"] = "predict"
                pred = candidate.predict(model, test_metadata, test_cycle_summary, config)
            else:
                model = candidate.fit(train_metadata, train_cycle_summary, train_labels, config)
                phase["current"] = "predict"
                pred = candidate.predict(test_metadata, test_cycle_summary, config)
            if not isinstance(pred, pd.DataFrame):
                pred = pd.DataFrame(pred)
        queue.put(
            CandidateRunResult(
                success=True,
                predictions=pred,
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
                syntax_status="passed",
                features_used=_features_used_summary(
                    feature_calls, _model_state_feature_columns(model)
                ),
            )
        )
    except Exception as exc:
        queue.put(
            CandidateRunResult(
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
                failure_reason=str(exc),
                traceback=traceback.format_exc(),
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
                syntax_status="passed",
                features_used=_features_used_summary(
                    feature_calls, _model_state_feature_columns(model)
                ),
            )
        )


def run_candidate(
    path: str | Path,
    train_metadata: pd.DataFrame,
    train_cycle_summary: pd.DataFrame,
    train_labels: pd.DataFrame,
    test_metadata: pd.DataFrame,
    test_cycle_summary: pd.DataFrame,
    config: dict[str, Any],
    timeout_s: int = 30,
    forbidden_patterns: tuple[str, ...] = FORBIDDEN_PATH_PATTERNS,
) -> CandidateRunResult:
    code = Path(path).read_text()
    try:
        compile(code, str(path), "exec")
    except SyntaxError as exc:
        return CandidateRunResult(
            success=False,
            error=str(exc),
            error_type=type(exc).__name__,
            failure_reason=str(exc),
            traceback=traceback.format_exc(),
            syntax_status="failed",
        )
    try:
        validate_code_safety(code)
    except Exception as exc:
        return CandidateRunResult(
            success=False,
            error=str(exc),
            error_type=type(exc).__name__,
            failure_reason=str(exc),
            traceback=traceback.format_exc(),
            syntax_status="passed",
        )
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_worker,
        args=(queue, str(path), train_metadata, train_cycle_summary, train_labels, test_metadata, test_cycle_summary, config, forbidden_patterns),
    )
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        msg = f"candidate timed out after {timeout_s}s"
        return CandidateRunResult(success=False, error=msg, error_type="TimeoutError", failure_reason=msg, syntax_status="passed")
    if queue.empty():
        msg = "candidate process produced no result"
        return CandidateRunResult(success=False, error=msg, error_type="RuntimeError", failure_reason=msg, syntax_status="passed")
    return queue.get()
