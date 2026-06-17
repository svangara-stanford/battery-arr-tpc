from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

OED_BATCH_NAMES = (
    "2018-08-28_oed_0",
    "2018-09-02_oed_1",
    "2018-09-06_oed_2",
    "2018-09-10_oed_3",
)
# Attia closed-loop "transfer-only" batch. Still referenced because the optional
# `--attia-transfer` / `--secondary-test-source=attia_batch9` path is preserved.
# As of patch E4 the locked secondary test now points to Severson b3 by default;
# see SEVERSON_MAT_DIR_NAME / SEVERSON_B3_BATCH_DATE below.
VALIDATION_BATCH_NAME = "2019-01-24_batch9"
VALIDATION_BATCH_ZIP = f"{VALIDATION_BATCH_NAME}.zip"

# Severson b3 (locked secondary test under the new default contract).
SEVERSON_MAT_DIR_NAME = "severson_2019_true_life_matr"
SEVERSON_B3_BATCH_DATE = "2018-04-12"
SEVERSON_B3_MAT_FILE_NAME = f"{SEVERSON_B3_BATCH_DATE}_batchdata_updated_struct_errorcorrect.mat"


@dataclass(frozen=True)
class PaperPaths:
    root: Path
    data_dir: Path
    bms_autoanalysis_dir: Path
    optimization_dir: Path


def find_battery_fast_charging_root(root: str | Path | None = None) -> Path:
    """Locate the copied paper-material root.

    The default location is intentionally non-recursive so a batch-9 ZIP is
    never scanned while resolving paths.
    """

    if root is not None:
        candidate = Path(root).expanduser().resolve()
    else:
        candidate = (Path.cwd() / "literature_models_and_data" / "battery-fast-charging").resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"battery-fast-charging root not found: {candidate}")
    return candidate


def resolve_paper_paths(root: str | Path | None = None) -> PaperPaths:
    paper_root = find_battery_fast_charging_root(root)
    return PaperPaths(
        root=paper_root,
        data_dir=paper_root / "data",
        bms_autoanalysis_dir=paper_root / "BMS-autoanalysis",
        optimization_dir=paper_root / "battery-fast-charging-optimization",
    )


def find_oed_batch_paths(data_dir: str | Path, include_validation_batch: bool = False, validation_batch_path: str | Path | None = None) -> list[Path]:
    """Return raw batch folders for Step 1.

    By default this returns only the four OED/CLO folders and does not inspect,
    open, or unzip `2019-01-24_batch9.zip`.
    """

    base = Path(data_dir)
    paths = [base / name for name in OED_BATCH_NAMES if (base / name).is_dir()]
    if include_validation_batch:
        validation = Path(validation_batch_path) if validation_batch_path else base / VALIDATION_BATCH_NAME
        if validation.is_dir():
            paths.append(validation)
    return paths


def find_raw_batch_files(data_dir: str | Path, include_validation_batch: bool = False, validation_batch_path: str | Path | None = None) -> list[Path]:
    files: list[Path] = []
    for batch_dir in find_oed_batch_paths(data_dir, include_validation_batch, validation_batch_path):
        if batch_dir.name.startswith(VALIDATION_BATCH_NAME) and not include_validation_batch:
            continue
        files.extend(sorted(batch_dir.glob("*_structure.json")))
        files.extend(sorted(batch_dir.glob("*.mat")))
    return files


def validation_status(data_dir: str | Path, include_validation_batch: bool = False, validation_batch_path: str | Path | None = None) -> str:
    base = Path(data_dir)
    if include_validation_batch:
        validation = Path(validation_batch_path) if validation_batch_path else base / VALIDATION_BATCH_NAME
        return "included_validation_batch" if validation.is_dir() else "validation_failed"
    if (base / VALIDATION_BATCH_ZIP).exists():
        return "skipped_batch9_zip_present"
    if (base / VALIDATION_BATCH_NAME).exists():
        return "skipped_by_default"
    return "skipped_batch9_missing"


def validation_status_label(status: str) -> str:
    mapping = {
        "skipped_by_default": "validation_skipped_by_default",
        "skipped_batch9_zip_present": "validation_skipped_batch9_zip_present",
        "skipped_batch9_missing": "validation_skipped_batch9_missing",
        "included_validation_batch": "validation_included_batch9",
        "validation_failed": "validation_failed",
    }
    return mapping.get(status, "validation_failed")


# ---------------------------------------------------------------------------
# Secondary-test helpers (patch E4)
#
# The campaign locked secondary test now defaults to Severson b3 instead of
# Attia Batch 9. We expose source-agnostic helpers (`secondary_test_status` /
# `secondary_test_status_label`) that today alias the legacy validation_*
# helpers; a full rename can happen in a future cleanup. The Severson b3
# helper below resolves the `.mat` file path under
# `<bfc>/data/severson_2019_true_life_matr/`.
# ---------------------------------------------------------------------------


def severson_b3_mat_path(bfc_root: str | Path) -> Path:
    """Return the path to the Severson b3 .mat file.

    The b3 batch corresponds to the 2018-04-12 batchdata file inside the
    `severson_2019_true_life_matr/` directory.
    """

    root = Path(bfc_root).expanduser().resolve()
    return root / "data" / SEVERSON_MAT_DIR_NAME / SEVERSON_B3_MAT_FILE_NAME


# Aliases. Today these are equivalent to the legacy validation_* helpers; the
# new names exist so call sites can be migrated without semantic drift. The old
# names continue to work to avoid breaking downstream tooling during the
# transition.
secondary_test_status = validation_status
secondary_test_status_label = validation_status_label


def infer_batch_name_map(batch_paths: list[Path]) -> dict[str, str]:
    """Map folder names to MATLAB-style names used in `apply_model2.m`.

    The four default folders are chronological OED rounds. MATLAB refers to
    them as `oed1`, `oed2`, `oed3`, and `oed4`; this explicit mapping is written
    into reports and can be overridden from the CLI.
    """

    ordered = sorted(batch_paths, key=lambda p: OED_BATCH_NAMES.index(p.name) if p.name in OED_BATCH_NAMES else 999)
    return {path.name: f"oed{idx + 1}" for idx, path in enumerate(ordered) if path.name in OED_BATCH_NAMES}


def parse_batch_name_map(items: list[str] | None) -> dict[str, str]:
    if not items:
        return {}
    mapping: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"batch-name-map item must be FOLDER=oedN, got: {item}")
        folder, batch_name = item.split("=", 1)
        mapping[folder] = batch_name
    return mapping
