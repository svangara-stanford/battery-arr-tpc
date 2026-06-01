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


def _worker(queue, path, train_metadata, train_cycle_summary, train_labels, test_metadata, test_cycle_summary, config, forbidden_patterns):
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            install_open_guard(tuple(forbidden_patterns))
            candidate = load_candidate(path)
            if isinstance(candidate, types.ModuleType):
                model = candidate.fit(train_metadata, train_cycle_summary, train_labels, config)
                pred = candidate.predict(model, test_metadata, test_cycle_summary, config)
            else:
                model = candidate.fit(train_metadata, train_cycle_summary, train_labels, config)
                pred = candidate.predict(test_metadata, test_cycle_summary, config)
            if not isinstance(pred, pd.DataFrame):
                pred = pd.DataFrame(pred)
        queue.put(CandidateRunResult(success=True, predictions=pred, stdout=stdout_buffer.getvalue(), stderr=stderr_buffer.getvalue(), syntax_status="passed"))
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
