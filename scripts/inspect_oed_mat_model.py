#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from battery_aar.paper_reproduction.mat_model_loader import inspect_mat_model_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect an Attia/Chueh OED MATLAB model file.")
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()
    diagnostics = inspect_mat_model_file(args.model)
    print(f"model_file: {diagnostics['path']}")
    print("keys:")
    for key in diagnostics["keys"]:
        var = diagnostics["variables"][key]
        print(f"  {key}: shape={var['shape']} dtype={var['dtype']}")
    print(f"required_variables: {diagnostics['required_variables']}")
    print(f"missing_required_variables: {diagnostics['missing_required_variables']}")
    print(f"load_status: {diagnostics['load_status']}")
    if diagnostics.get("error"):
        print(f"error: {diagnostics['error']}")
    print("json_diagnostics:")
    print(json.dumps(diagnostics, indent=2, default=str))
    return 0 if diagnostics["load_status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
