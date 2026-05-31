from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

C1_VALUES = [3.6, 4.0, 4.4, 4.8, 5.2, 5.6, 6.0, 7.0, 8.0]
C2_VALUES = [3.6, 4.0, 4.4, 4.8, 5.2, 5.6, 6.0, 7.0]
C3_VALUES = [3.6, 4.0, 4.4, 4.8, 5.2, 5.6]
C4_LIMITS = (0.1, 4.81)


def compute_c4(c1: float, c2: float, c3: float) -> float:
    return 0.2 / (1 / 6 - (0.2 / c1 + 0.2 / c2 + 0.2 / c3))


def generate_policy_space() -> pd.DataFrame:
    rows: list[tuple[float, float, float, float]] = []
    for c1 in C1_VALUES:
        for c2 in C2_VALUES:
            for c3 in C3_VALUES:
                c4 = compute_c4(c1, c2, c3)
                if C4_LIMITS[0] <= c4 <= C4_LIMITS[1]:
                    if c1 == 4.8 and c2 == 4.8 and c3 == 4.8:
                        continue
                    rows.append((c1, c2, c3, c4))
    df = pd.DataFrame(rows, columns=["C1", "C2", "C3", "C4"])
    return df.round({"C1": 3, "C2": 3, "C3": 3, "C4": 3})


def save_policy_space(path: str | Path) -> pd.DataFrame:
    df = generate_policy_space()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def assert_author_policy_count(df: pd.DataFrame) -> None:
    if len(df) != 224:
        raise AssertionError(f"Expected 224 valid policies, found {len(df)}")
    if np.any((df["C4"] < C4_LIMITS[0]) | (df["C4"] > C4_LIMITS[1])):
        raise AssertionError("Generated policy has C4 outside author limits")
