from __future__ import annotations

import importlib.util
import multiprocessing as mp
import traceback
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
    try:
        install_open_guard(tuple(forbidden_patterns))
        candidate = load_candidate(path)
        if hasattr(candidate, "fit") and hasattr(candidate, "predict"):
            model = candidate.fit(train_metadata, train_cycle_summary, train_labels, config)
            pred = candidate.predict(model, test_metadata, test_cycle_summary, config)
        else:
            model = candidate.fit(train_metadata, train_cycle_summary, train_labels, config)
            pred = candidate.predict(test_metadata, test_cycle_summary, config)
        if not isinstance(pred, pd.DataFrame):
            pred = pd.DataFrame(pred)
        queue.put(CandidateRunResult(success=True, predictions=pred))
    except Exception as exc:
        queue.put(CandidateRunResult(success=False, error=str(exc), stderr=traceback.format_exc()))


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
    validate_code_safety(Path(path).read_text())
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
        return CandidateRunResult(success=False, error=f"candidate timed out after {timeout_s}s")
    if queue.empty():
        return CandidateRunResult(success=False, error="candidate process produced no result")
    return queue.get()
