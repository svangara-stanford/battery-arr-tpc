import numpy as np
import pytest
from scipy.io import savemat

from battery_aar.paper_reproduction.mat_model_loader import ModelLoadError, load_oed_mat_model


def test_mat_model_loader_converts_matlab_indices(tmp_path):
    path = tmp_path / "toy.mat"
    savemat(
        path,
        {
            "mu": np.zeros(3),
            "sigma": np.ones(3),
            "feat_ind": np.array([1, 3]),
            "B1": np.array([0.5, -0.25]),
            "y_mu": 2.0,
            "MSE": 0.1,
            "t_val": 2.0,
            "des_mat": np.eye(3),
        },
    )
    model = load_oed_mat_model(path)
    assert model.feat_ind_matlab.tolist() == [1, 3]
    assert model.feat_ind_python.tolist() == [0, 2]
    assert model.B1.shape == (2,)


def test_mat_model_loader_rejects_missing_variables(tmp_path):
    path = tmp_path / "bad.mat"
    savemat(path, {"mu": np.zeros(3)})
    with pytest.raises(ModelLoadError):
        load_oed_mat_model(path)
