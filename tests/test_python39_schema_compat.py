from __future__ import annotations

import importlib.util
import re
from pathlib import Path


SCHEMA_FILES = [
    Path("src/battery_aar/features/schemas.py"),
    Path("src/battery_aar/features/operator_registry.py"),
    Path("src/battery_aar/tools/schemas.py"),
    Path("src/battery_aar/workflows/schemas.py"),
]


def test_schema_and_feature_program_imports():
    import battery_aar.features.feature_programs  # noqa: F401
    import battery_aar.features.operators  # noqa: F401
    import battery_aar.workflows.schemas  # noqa: F401
    from battery_aar.features.program_library import make_broad_physics_program

    program = make_broad_physics_program()
    assert program.program_id == "broad_physics"

    script_path = Path("scripts/build_battery_feature_program.py")
    spec = importlib.util.spec_from_file_location("build_battery_feature_program_import_check", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")


def test_pydantic_schema_files_do_not_use_pep604_optional_unions():
    optional_union = re.compile(r"(\|\s*None|None\s*\|)")
    runtime_union = re.compile(r"=\s*[A-Za-z_][A-Za-z0-9_]*(\s*\|\s*[A-Za-z_][A-Za-z0-9_]*)+")
    offenders = []
    for path in SCHEMA_FILES:
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if optional_union.search(line) or runtime_union.search(line):
                offenders.append(f"{path}:{lineno}: {line.strip()}")

    assert offenders == []
