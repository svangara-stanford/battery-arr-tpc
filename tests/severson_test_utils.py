from __future__ import annotations

import numpy as np
from pathlib import Path
from scipy.io import savemat


def write_toy_severson_mat_dir(base: Path, *, n_files: int = 2, n_cells_per_file: int = 3, n_cycles: int = 120) -> Path:
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
                frac = np.linspace(0.0, 1.0, 8)
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
