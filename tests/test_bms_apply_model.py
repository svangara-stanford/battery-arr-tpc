import numpy as np
from scipy.io import savemat

from battery_aar.paper_reproduction.bms_apply_model import apply_oed_model, write_bayesgap_input
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
