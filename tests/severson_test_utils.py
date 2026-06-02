from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat


def write_toy_severson_mat_dir(
    base: Path,
    *,
    n_files: int = 2,
    n_cells_per_file: int = 3,
    n_cycles: int = 120,
    ragged_cycles: bool = False,
) -> Path:
    mat_dir = base / "severson_matr"
    mat_dir.mkdir(parents=True, exist_ok=True)
    file_names = [
        "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
        "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
        "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
    ]
    for file_idx in range(n_files):
        dtype = np.dtype([("cycle_life", "O"), ("summary", "O"), ("cycles", "O"), ("policy", "O")])
        batch = np.empty((1, n_cells_per_file), dtype=dtype)
        for cell_idx in range(n_cells_per_file):
            cycles = np.arange(n_cycles)
            qd = 1.12 - 0.001 * cycles - 0.003 * cell_idx - 0.002 * file_idx
            qc = qd + 0.02
            summary = {
                "cycle_index": cycles,
                "QDischarge": qd,
                "QCharge": qc,
                "IR": 0.012 + 0.00005 * cycles + 0.0001 * cell_idx,
                "Tavg": 25.0 + 0.01 * cycles,
            }
            cycle_records = []
            for cycle in range(min(12, n_cycles)):
                n_points = 5 + (cycle % 4) if ragged_cycles else 8
                frac = np.linspace(0.0, 1.0, n_points)
                cycle_records.append(
                    {
                        "cycle_index": cycle,
                        "V": 4.2 - frac,
                        "I": -np.ones_like(frac),
                        "Qd": qd[cycle] * frac,
                        "Qc": np.zeros_like(frac),
                        "t": np.arange(frac.size) + 1000 * cycle,
                        "T": 25.0 + frac,
                    }
                )
            batch[0, cell_idx]["cycle_life"] = np.array([[700 + 50 * file_idx + 5 * cell_idx]])
            batch[0, cell_idx]["summary"] = summary
            batch[0, cell_idx]["cycles"] = np.array(cycle_records, dtype=object)
            batch[0, cell_idx]["policy"] = f"policy_{cell_idx % 2}"
        savemat(mat_dir / file_names[file_idx], {"batch": batch})
    return mat_dir


def _h5_char_dataset(group: h5py.Group, name: str, text: str) -> h5py.Dataset:
    codes = np.asarray([ord(ch) for ch in text], dtype=np.uint16).reshape((-1, 1))
    return group.create_dataset(name, data=codes)


def write_toy_severson_h5_mat_dir(
    base: Path,
    *,
    n_files: int = 1,
    n_cells_per_file: int = 2,
    n_cycles: int = 120,
    include_ragged_cycles: bool = True,
) -> Path:
    mat_dir = base / "severson_matr_h5"
    mat_dir.mkdir(parents=True, exist_ok=True)
    file_names = [
        "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
        "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
        "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
    ]
    ref_dtype = h5py.special_dtype(ref=h5py.Reference)
    for file_idx in range(n_files):
        with h5py.File(mat_dir / file_names[file_idx], "w") as handle:
            refs = handle.create_group("#refs#")
            handle.create_dataset("batch_date", data=np.asarray([[ord(ch)] for ch in "toy"], dtype=np.uint16))
            batch = handle.create_group("batch")
            life_refs = np.empty((1, n_cells_per_file), dtype=ref_dtype)
            summary_refs = np.empty((1, n_cells_per_file), dtype=ref_dtype)
            cycles_refs = np.empty((1, n_cells_per_file), dtype=ref_dtype)
            policy_refs = np.empty((1, n_cells_per_file), dtype=ref_dtype)
            for cell_idx in range(n_cells_per_file):
                cycles = np.arange(n_cycles)
                qd = 1.12 - 0.001 * cycles - 0.003 * cell_idx - 0.002 * file_idx
                qc = qd + 0.02
                life = refs.create_dataset(f"life_{cell_idx}", data=np.asarray([[700 + 50 * file_idx + 5 * cell_idx]], dtype=float))
                life_refs[0, cell_idx] = life.ref
                summary = refs.create_group(f"summary_{cell_idx}")
                summary.create_dataset("cycle", data=cycles.reshape((1, -1)))
                summary.create_dataset("QD", data=qd.reshape((1, -1)))
                summary.create_dataset("QC", data=qc.reshape((1, -1)))
                summary.create_dataset("IR", data=(0.012 + 0.00005 * cycles + 0.0001 * cell_idx).reshape((1, -1)))
                summary.create_dataset("Tavg", data=(25.0 + 0.01 * cycles).reshape((1, -1)))
                summary.create_dataset("chargetime", data=(30.0 + 0.1 * cycles).reshape((1, -1)))
                summary_refs[0, cell_idx] = summary.ref
                cycles_group = refs.create_group(f"cycles_{cell_idx}")
                n_cycle_records = min(12, n_cycles)
                for field in ["V", "I", "Qd", "Qc", "t", "T"]:
                    field_refs = np.empty((1, n_cycle_records), dtype=ref_dtype)
                    for cycle in range(n_cycle_records):
                        n_points = 5 + (cycle % 4) if include_ragged_cycles else 8
                        frac = np.linspace(0.0, 1.0, n_points)
                        if field == "V":
                            values = 4.2 - frac
                        elif field == "I":
                            values = -np.ones_like(frac)
                        elif field == "Qd":
                            values = qd[cycle] * frac
                        elif field == "Qc":
                            values = np.zeros_like(frac)
                        elif field == "t":
                            values = np.arange(frac.size) + 1000 * cycle
                        else:
                            values = 25.0 + frac
                        ds = refs.create_dataset(f"{field}_{cell_idx}_{cycle}", data=values.reshape((1, -1)))
                        field_refs[0, cycle] = ds.ref
                    cycles_group.create_dataset(field, data=field_refs)
                cycles_refs[0, cell_idx] = cycles_group.ref
                policy = _h5_char_dataset(refs, f"policy_{cell_idx}", f"policy_{cell_idx % 2}")
                policy_refs[0, cell_idx] = policy.ref
            batch.create_dataset("cycle_life", data=life_refs)
            batch.create_dataset("summary", data=summary_refs)
            batch.create_dataset("cycles", data=cycles_refs)
            batch.create_dataset("policy", data=policy_refs)
    return mat_dir
