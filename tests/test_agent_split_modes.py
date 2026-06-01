import json

import pandas as pd

from battery_aar.agents.orchestrator import run_rediscovery


def _write_processed_dataset(base, n_batches=4, n_protocols=6, cells_per_protocol=2):
    base.mkdir(parents=True, exist_ok=True)
    rows_meta = []
    rows_cycles = []
    row_id = 0
    for protocol_idx in range(n_protocols):
        batch_id = f"2018-09-{protocol_idx % n_batches + 1:02d}_oed_{protocol_idx % n_batches}"
        c1 = 3.6 + 0.4 * (protocol_idx % 4)
        c2 = 4.0 + 0.4 * (protocol_idx % 3)
        c3 = 4.4 + 0.4 * (protocol_idx % 2)
        c4 = 4.16 + 0.02 * protocol_idx
        protocol = f"{c1:.1f}C-{c2:.1f}C-{c3:.1f}C-{c4:.3f}C"
        for cell_idx in range(cells_per_protocol):
            rows_meta.append(
                {
                    "row_id": row_id,
                    "cell_id": f"cell_{row_id}",
                    "batch_id": batch_id,
                    "protocol_readable": protocol,
                    "C1": c1,
                    "C2": c2,
                    "C3": c3,
                    "C4": c4,
                    "cycle_life": 700.0 + 20 * protocol_idx + cell_idx,
                    "label_source": "author_model_prediction",
                }
            )
            for cycle in range(1, 21):
                rows_cycles.append(
                    {
                        "row_id": row_id,
                        "cell_id": f"cell_{row_id}",
                        "cycle_index": cycle,
                        "discharge_capacity": 1.1 - 0.001 * cycle - 0.0001 * protocol_idx,
                        "charge_capacity": float("nan"),
                    }
                )
            row_id += 1
    pd.DataFrame(rows_meta).to_csv(base / "cell_metadata.csv", index=False)
    pd.DataFrame(rows_cycles).to_csv(base / "cycle_summary.csv", index=False)
    (base / "dataset_card.json").write_text(json.dumps({"label_source": "author_model_prediction", "batch9_included": False}) + "\n")


def _run_split(tmp_path, split_mode, **kwargs):
    processed = tmp_path / "processed"
    _write_processed_dataset(processed)
    return run_rediscovery(
        processed_dir=processed,
        reference_run=tmp_path / "missing_reference",
        out=tmp_path / f"run_{split_mode}",
        reports_dir=tmp_path / "reports",
        agents=1,
        iterations=1,
        offline=True,
        require_real_data=True,
        split_mode=split_mode,
        validation_fraction=kwargs.pop("validation_fraction", 0.25),
        split_seed=kwargs.pop("split_seed", 0),
        **kwargs,
    )


def test_random_split_keeps_each_cell_wholly_assigned(tmp_path):
    report = _run_split(tmp_path, "random", split_seed=11)
    assignments = pd.read_csv(tmp_path / "run_random" / "split_assignments.csv")
    cycles = pd.read_csv(tmp_path / "processed" / "cycle_summary.csv")
    merged = cycles[["row_id"]].merge(assignments[["row_id", "split"]], on="row_id", how="left")

    assert report["split_manifest"]["group_type"] == "random_cell"
    assert assignments.groupby("row_id")["split"].nunique().max() == 1
    assert merged["split"].notna().all()
    assert merged.groupby("row_id")["split"].nunique().max() == 1


def test_protocol_split_has_no_protocol_overlap_and_writes_artifacts(tmp_path):
    report = _run_split(tmp_path, "protocol", split_seed=2)
    assignments = pd.read_csv(tmp_path / "run_protocol" / "split_assignments.csv")
    manifest = json.loads((tmp_path / "run_protocol" / "split_manifest.json").read_text())
    train_protocols = set(assignments.loc[assignments["split"] == "train", "group_key"])
    val_protocols = set(assignments.loc[assignments["split"] == "validation", "group_key"])

    assert train_protocols.isdisjoint(val_protocols)
    assert manifest["group_type"] == "protocol"
    assert manifest["heldout_protocols"]
    assert report["split_manifest"]["heldout_protocols"] == manifest["heldout_protocols"]
    report_md = (tmp_path / "reports" / "agent_rediscovery.md").read_text()
    assert "## Search Split" in report_md
    assert "Held-Out Protocols" in report_md
    assert "Batch 9 was not used during surrogate search" in report_md


def test_batch_split_has_no_batch_overlap(tmp_path):
    _run_split(tmp_path, "batch", split_seed=3)
    assignments = pd.read_csv(tmp_path / "run_batch" / "split_assignments.csv")
    train_batches = set(assignments.loc[assignments["split"] == "train", "batch_id"])
    val_batches = set(assignments.loc[assignments["split"] == "validation", "batch_id"])

    assert train_batches.isdisjoint(val_batches)
    assert val_batches


def test_batch_split_validation_batch_id_works(tmp_path):
    validation_batch = "2018-09-02_oed_1"
    report = _run_split(tmp_path, "batch", validation_batch_id=validation_batch)
    assignments = pd.read_csv(tmp_path / "run_batch" / "split_assignments.csv")
    val_batches = set(assignments.loc[assignments["split"] == "validation", "batch_id"])

    assert val_batches == {validation_batch}
    assert report["split_manifest"]["heldout_batch_ids"] == [validation_batch]


def test_split_rejects_batch9_in_surrogate_search_dataset(tmp_path):
    processed = tmp_path / "processed"
    _write_processed_dataset(processed, n_batches=2, n_protocols=4)
    metadata = pd.read_csv(processed / "cell_metadata.csv")
    metadata.loc[0, "batch_id"] = "2019-01-24_batch9"
    metadata.to_csv(processed / "cell_metadata.csv", index=False)

    try:
        run_rediscovery(
            processed_dir=processed,
            reference_run=tmp_path / "missing_reference",
            out=tmp_path / "run",
            reports_dir=tmp_path / "reports",
            agents=1,
            iterations=1,
            offline=True,
            require_real_data=True,
            split_mode="batch",
        )
    except ValueError as exc:
        assert "Batch 9 must not be included" in str(exc)
    else:
        raise AssertionError("Batch 9 search dataset was not rejected")
