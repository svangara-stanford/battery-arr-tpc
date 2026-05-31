from pathlib import Path

from battery_aar.paper_reproduction.paths import OED_BATCH_NAMES, find_oed_batch_paths, find_raw_batch_files, validation_status


def test_batch9_zip_is_skipped_by_default(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for name in OED_BATCH_NAMES:
        (data / name).mkdir()
        (data / name / f"{name}_CH1_structure.json").write_text("{}")
    zip_path = data / "2019-01-24_batch9.zip"
    zip_path.write_text("placeholder")

    batches = find_oed_batch_paths(data)
    files = find_raw_batch_files(data)
    assert [p.name for p in batches] == list(OED_BATCH_NAMES)
    assert all("batch9" not in p.name for p in files)
    assert validation_status(data) == "skipped_batch9_zip_present"
