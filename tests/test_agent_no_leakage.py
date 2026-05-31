import pytest
import pandas as pd

from battery_aar.agents.evaluator import sanitize_metadata
from battery_aar.agents.sandbox import validate_code_safety


def test_sanitize_metadata_removes_identifiers_and_protocol_by_default():
    df = pd.DataFrame({"row_id": [1], "cell_id": ["c"], "batch_id": ["b"], "cc1": [4.0], "feature": [1.2]})
    out = sanitize_metadata(df)
    assert "row_id" in out
    assert "feature" in out
    assert "cell_id" not in out
    assert "batch_id" not in out
    assert "cc1" not in out


def test_candidate_code_cannot_reference_reference_paths():
    with pytest.raises(ValueError):
        validate_code_safety("open('literature_models_and_data/battery-fast-charging/BMS-autoanalysis/oed_model.mat')")
