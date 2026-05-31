from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io

REQUIRED_MODEL_VARIABLES = ("mu", "sigma", "feat_ind", "B1", "y_mu", "MSE", "t_val", "des_mat")


class ModelLoadError(ValueError):
    pass


@dataclass
class OEDMatModel:
    path: Path
    mu: np.ndarray
    sigma: np.ndarray
    feat_ind_python: np.ndarray
    feat_ind_matlab: np.ndarray
    B1: np.ndarray
    y_mu: float
    MSE: float
    t_val: float
    des_mat: np.ndarray

    def diagnostics(self) -> dict[str, Any]:
        out = asdict(self)
        out["path"] = str(self.path)
        for key, value in list(out.items()):
            if isinstance(value, np.ndarray):
                out[key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "values": value.tolist() if value.size <= 20 else value.ravel()[:20].tolist(),
                }
        return out


def _load_raw_mat(path: Path) -> dict[str, Any]:
    try:
        return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        pass
    except ValueError as exc:
        # scipy raises ValueError for some v7.3/HDF5 files.
        if "Unknown mat file type" not in str(exc) and "Please use HDF reader" not in str(exc):
            raise
    data: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        for key in handle.keys():
            data[key] = np.array(handle[key])
    return data


def _as_1d_float(data: dict[str, Any], key: str) -> np.ndarray:
    arr = np.asarray(data[key], dtype=float).squeeze()
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ModelLoadError(f"{key} must be a 1D float array, got shape {arr.shape}")
    return arr


def _as_scalar_float(data: dict[str, Any], key: str) -> float:
    arr = np.asarray(data[key], dtype=float).squeeze()
    if arr.ndim != 0:
        raise ModelLoadError(f"{key} must be a scalar float, got shape {arr.shape}")
    return float(arr)


def _as_2d_float(data: dict[str, Any], key: str) -> np.ndarray:
    arr = np.asarray(data[key], dtype=float).squeeze()
    if arr.ndim != 2:
        raise ModelLoadError(f"{key} must be a 2D float array, got shape {arr.shape}")
    return arr


def load_oed_mat_model(path: str | Path) -> OEDMatModel:
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Author model file not found: {model_path}")
    raw = _load_raw_mat(model_path)
    missing = [key for key in REQUIRED_MODEL_VARIABLES if key not in raw]
    if missing:
        raise ModelLoadError(f"Missing required variables in {model_path}: {missing}")

    mu = _as_1d_float(raw, "mu")
    sigma = _as_1d_float(raw, "sigma")
    feat_ind_matlab = np.asarray(raw["feat_ind"], dtype=int).squeeze()
    if feat_ind_matlab.ndim == 0:
        feat_ind_matlab = feat_ind_matlab.reshape(1)
    if feat_ind_matlab.ndim != 1:
        raise ModelLoadError(f"feat_ind must be a 1D int array, got shape {feat_ind_matlab.shape}")
    if np.any(feat_ind_matlab < 1):
        raise ModelLoadError("feat_ind must contain MATLAB 1-based positive indices")
    feat_ind_python = feat_ind_matlab - 1

    B1 = _as_1d_float(raw, "B1")
    y_mu = _as_scalar_float(raw, "y_mu")
    mse = _as_scalar_float(raw, "MSE")
    t_val = _as_scalar_float(raw, "t_val")
    des_mat = _as_2d_float(raw, "des_mat")

    if mu.shape != sigma.shape:
        raise ModelLoadError(f"mu and sigma shape mismatch: {mu.shape} vs {sigma.shape}")
    if np.max(feat_ind_python) >= mu.size:
        raise ModelLoadError(f"feat_ind references feature outside mu/sigma length {mu.size}")
    if B1.size != feat_ind_python.size:
        raise ModelLoadError(f"B1 length {B1.size} does not match feat_ind length {feat_ind_python.size}")
    expected_des = feat_ind_python.size + 1
    if des_mat.shape != (expected_des, expected_des):
        raise ModelLoadError(f"des_mat shape {des_mat.shape} must be {(expected_des, expected_des)}")
    if np.any(sigma == 0):
        raise ModelLoadError("sigma contains zeros")

    return OEDMatModel(
        path=model_path,
        mu=mu,
        sigma=sigma,
        feat_ind_python=feat_ind_python,
        feat_ind_matlab=feat_ind_matlab,
        B1=B1,
        y_mu=y_mu,
        MSE=mse,
        t_val=t_val,
        des_mat=des_mat,
    )


def inspect_mat_model_file(path: str | Path) -> dict[str, Any]:
    model_path = Path(path)
    raw = _load_raw_mat(model_path)
    variables: dict[str, Any] = {}
    for key, value in raw.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(value)
        variables[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    diagnostics = {
        "path": str(model_path),
        "keys": sorted(variables),
        "variables": variables,
        "required_variables": list(REQUIRED_MODEL_VARIABLES),
        "missing_required_variables": [key for key in REQUIRED_MODEL_VARIABLES if key not in raw],
        "load_status": "unvalidated",
    }
    try:
        diagnostics["model"] = load_oed_mat_model(model_path).diagnostics()
        diagnostics["load_status"] = "ok"
    except Exception as exc:  # diagnostic script should report, not mask.
        diagnostics["load_status"] = "failed"
        diagnostics["error"] = str(exc)
    return diagnostics
