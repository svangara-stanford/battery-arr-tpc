import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from battery_aar.paper_reproduction.bms_apply_model import (
    apply_oed_model,
    transform_prediction,
    validate_bayesgap_prediction_scale,
    write_bayesgap_input,
)
from tests.test_bms_features import synthetic_cell


def _save_model(path, mse=0.001):
    savemat(
        path,
        {
            "mu": np.zeros(15),
            "sigma": np.ones(15),
            "feat_ind": np.array([1, 2]),
            "B1": np.array([0.1, 0.2]),
            "y_mu": 2.0,
            "MSE": mse,
            "t_val": 2.0,
            "des_mat": np.eye(3),
        },
    )


def test_apply_oed_model_and_bayesgap_columns(tmp_path):
    model = tmp_path / "toy.mat"
    _save_model(model)
    df = apply_oed_model([synthetic_cell()], model, cutoff_cycle=100, batch_name="oed2")
    assert df.loc[0, "Prediction"] > 0
    assert df.loc[0, "target_transform"] == "log10_cycle_life"
    assert "raw_output" in df.columns
    assert "ci_width" in df.columns
    csv = tmp_path / "pred.csv"
    bg = write_bayesgap_input(df, csv)
    assert list(bg.columns) == ["C1", "C2", "C3", "C4", "Prediction"]
    assert list(csv.read_text().splitlines()[0].split(",")) == ["C1", "C2", "C3", "C4", "Prediction"]


def test_apply_oed_model_flags_anomalous_intervals(tmp_path):
    model = tmp_path / "wide.mat"
    _save_model(model, mse=10.0)
    df = apply_oed_model([synthetic_cell()], model, cutoff_cycle=100)
    assert df.loc[0, "anomaly_flag"]
    assert df.loc[0, "Prediction"] == -1


def test_inverse_lifetime_transform_returns_cycle_life():
    pred, lo, hi = transform_prediction(np.log10(1 / 1000), 0.0, "log10_inverse_cycle_life")
    assert pred == pytest.approx(1000.0)
    assert lo == pytest.approx(1000.0)
    assert hi == pytest.approx(1000.0)


def test_apply_oed_model_auto_detects_inverse_lifetime(tmp_path):
    model = tmp_path / "inverse.mat"
    savemat(
        model,
        {
            "mu": np.zeros(15),
            "sigma": np.ones(15),
            "feat_ind": np.array([1]),
            "B1": np.array([0.0]),
            "y_mu": np.log10(1 / 1000),
            "MSE": 0.0,
            "t_val": 2.0,
            "des_mat": np.eye(2),
        },
    )
    df = apply_oed_model([synthetic_cell()], model, cutoff_cycle=100, target_transform="auto")
    assert df.loc[0, "target_transform"] == "log10_inverse_cycle_life"
    assert df.loc[0, "Prediction"] == pytest.approx(1000.0)


def test_bayesgap_prediction_scale_safety_rejects_sub_10_cycle_outputs():
    low_scale = pd.DataFrame({"C1": [4.0, 4.4], "C2": [4.0, 4.4], "C3": [4.0, 4.4], "C4": [4.0, 4.0], "Prediction": [0.001, 0.002]})
    with pytest.raises(ValueError):
        validate_bayesgap_prediction_scale(low_scale)


def test_bayesgap_prediction_scale_safety_accepts_cycle_life_units():
    cycle_scale = pd.DataFrame({"C1": [4.0, 4.4], "C2": [4.0, 4.4], "C3": [4.0, 4.4], "C4": [4.0, 4.0], "Prediction": [700.0, 1100.0]})
    validate_bayesgap_prediction_scale(cycle_scale)


def test_write_bayesgap_input_filters_anomalies(tmp_path):
    df = pd.DataFrame(
        {
            "C1": [4.0, 4.4],
            "C2": [4.0, 4.4],
            "C3": [4.0, 4.4],
            "C4": [4.0, 4.0],
            "Prediction": [1000.0, -1.0],
            "anomaly_flag": [False, True],
        }
    )
    out = write_bayesgap_input(df, tmp_path / "pred.csv")
    assert len(out) == 1
    assert out["Prediction"].iloc[0] == 1000.0
