import numpy as np

from battery_aar.paper_reproduction.policy_space import compute_c4, generate_policy_space


def test_policy_space_has_author_count_and_excludes_baseline():
    df = generate_policy_space()
    assert len(df) == 224
    assert not ((df.C1 == 4.8) & (df.C2 == 4.8) & (df.C3 == 4.8)).any()
    assert ((df.C4 >= 0.1) & (df.C4 <= 4.81)).all()


def test_policy_space_known_protocol():
    df = generate_policy_space()
    expected = round(compute_c4(5.6, 6.0, 4.8), 3)
    row = df[(df.C1 == 5.6) & (df.C2 == 6.0) & (df.C3 == 4.8)]
    assert len(row) == 1
    assert np.isclose(row.iloc[0].C4, expected)
