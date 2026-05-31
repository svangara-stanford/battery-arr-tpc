from __future__ import annotations

import builtins
from pathlib import Path

FORBIDDEN_PATH_PATTERNS = (
    "literature_models_and_data",
    "src/battery_aar/paper_reproduction",
    "docs/scientific_baselines.md",
    "reports/attia_reference_reproduction",
    "runs/attia_reference_reproduction",
    "2019-01-24_batch9.zip",
)


def is_forbidden_path(path: str | Path, forbidden_patterns: tuple[str, ...] = FORBIDDEN_PATH_PATTERNS) -> bool:
    text = str(path)
    return any(pattern in text for pattern in forbidden_patterns)


def validate_code_safety(code: str) -> None:
    forbidden_tokens = [
        "requests",
        "urllib",
        "socket",
        "subprocess",
        "http://",
        "https://",
        "literature_models_and_data",
        "paper_reproduction",
        "attia_reference_reproduction",
        "2019-01-24_batch9",
    ]
    hits = [token for token in forbidden_tokens if token in code]
    if hits:
        raise ValueError(f"candidate code contains forbidden token(s): {hits}")


def install_open_guard(forbidden_patterns: tuple[str, ...] = FORBIDDEN_PATH_PATTERNS):
    original_open = builtins.open
    original_path_open = Path.open

    def guarded_open(file, *args, **kwargs):
        if is_forbidden_path(file, forbidden_patterns):
            raise PermissionError(f"candidate attempted to access forbidden path: {file}")
        return original_open(file, *args, **kwargs)

    def guarded_path_open(self, *args, **kwargs):
        if is_forbidden_path(self, forbidden_patterns):
            raise PermissionError(f"candidate attempted to access forbidden path: {self}")
        return original_path_open(self, *args, **kwargs)

    builtins.open = guarded_open
    Path.open = guarded_path_open
    return original_open, original_path_open
